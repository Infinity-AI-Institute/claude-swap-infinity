"""One shared local endpoint that rotates setup-token accounts on rate limits.

Every Claude Code process on the machine points at it (``ANTHROPIC_BASE_URL``
plus a fixed local capability as ``CLAUDE_CODE_OAUTH_TOKEN``). The proxy keeps
using one account until the provider answers 429, then marks that account as
cooling down and replays the same request on the next available account, so a
rate limit is absorbed once for every running agent instead of per session.

Only setup-token accounts are pooled: they never refresh, so the proxy holds no
refresh tokens. The upstream, forwarded headers and request handling are the
native adapter's (``vision_proxy.InferenceProxy``).

Setup, limits and the security trade-offs of one shared endpoint are in
``docs/rotate-proxy.md``.
"""

from __future__ import annotations

import json
import math
import os
import secrets
import threading
import time
from pathlib import Path

from claude_swap.exceptions import ClaudeSwitchError, NoUsableLogin
from claude_swap.vision_proxy import InferenceProxy

DEFAULT_PORT = 41871
PORT_ENVIRONMENT_VARIABLE = "CSWAP_ROTATE_PORT"
DEFAULT_COOLDOWN_S = 15 * 60
MAX_COOLDOWN_S = 6 * 3600
# How long a reader waits for another process that has created the capability
# file to finish writing it.
CAPABILITY_WRITE_WAIT_S = 1.0


def _retry_seconds(retry_after) -> float:
    try:
        value = float(retry_after)
    except (TypeError, ValueError):
        return DEFAULT_COOLDOWN_S
    return min(max(value, 30.0), MAX_COOLDOWN_S)


def port_from_environment(environ=os.environ) -> int:
    """The proxy port: ``CSWAP_ROTATE_PORT`` when set, otherwise the default."""
    raw = environ.get(PORT_ENVIRONMENT_VARIABLE)
    if raw is None:
        return DEFAULT_PORT
    if raw.isascii() and raw.isdecimal() and 1 <= int(raw) <= 65535:
        return int(raw)
    raise ClaudeSwitchError(
        f"{PORT_ENVIRONMENT_VARIABLE} must be a port number from 1 to 65535, not {raw!r}."
    )


def _setup_token(raw_credentials: str) -> str | None:
    """The setup token in an account's stored credentials, or ``None``.

    A setup token is an OAuth access token stored without a refresh token.
    The other kinds are never pooled: an API key (stored as a bare string)
    bills per token, and a browser login carries a refresh token that Claude
    Code owns and rotates.
    """
    try:
        oauth = json.loads(raw_credentials)["claudeAiOauth"]
        token = oauth["accessToken"]
        has_refresh_token = bool(oauth.get("refreshToken"))
    except (ValueError, KeyError, TypeError):
        return None
    if has_refresh_token or not isinstance(token, str):
        return None
    return token if token.startswith("sk-ant-oat") else None


def load_token_accounts(switcher, last=("sweep",)) -> list[dict]:
    """Enabled setup-token accounts in slot order, aliases in ``last`` at the end.

    Accounts are chosen by what is stored, not by their label: ``add-token
    --email`` registers a setup token under a real address.
    """
    try:
        seq = json.loads(switcher.sequence_file.read_text())
    except FileNotFoundError:
        return []  # no account has ever been registered on this machine
    accounts = seq.get("accounts", {})
    order = [str(n) for n in seq.get("sequence", [])] or sorted(accounts, key=int)
    pool = []
    for num in order:
        account = accounts.get(num) or {}
        email = account.get("email", "")
        if account.get("disabled"):
            continue
        token = _setup_token(switcher.read_account_credentials(num, email))
        if token is None:
            continue
        pool.append({"num": num, "name": account.get("alias") or email, "token": token})
    return [a for a in pool if a["name"] not in last] + [a for a in pool if a["name"] in last]


class LocalTokenPool:
    """Sticky rotation: stay on one account until it is rate-limited."""

    def __init__(self, accounts, *, clock=time.monotonic, log=None):
        if not accounts:
            raise ClaudeSwitchError(
                "No setup-token accounts to rotate. Register one with `cswap add-token`."
            )
        self.accounts = accounts
        self.cooling = {}  # num -> monotonic time it becomes usable again
        self.dead = set()  # nums whose token the provider rejected (401)
        self.index = 0
        self.lock = threading.Lock()
        self.clock = clock
        self.log = log or (lambda _msg: None)

    def _credential(self, account):
        return {"accessToken": account["token"], "num": account["num"], "name": account["name"]}

    def _next_available(self, start):
        now = self.clock()
        n = len(self.accounts)
        for step in range(n):
            i = (start + step) % n
            account = self.accounts[i]
            if account["num"] in self.dead:
                continue
            if self.cooling.get(account["num"], 0) <= now:
                return i
        return None

    def _no_usable_account(self) -> NoUsableLogin:
        """Why no account can serve right now.

        Call with the lock held, and only after ``_next_available`` found nothing.
        """
        live = [a for a in self.accounts if a["num"] not in self.dead]
        if not live:
            return NoUsableLogin(
                "Every setup token in the rotation pool was rejected by the provider."
            )
        # Every live account is cooling, so each has an entry in the future.
        soonest = min(self.cooling[a["num"]] for a in live)
        wait = max(1, math.ceil(soonest - self.clock()))
        return NoUsableLogin(
            "Every remaining setup token in the rotation pool is rate-limited; "
            f"the first is expected back in {wait} seconds.",
            retry_after_seconds=wait,
        )

    def get(self, *, model=None):
        with self.lock:
            i = self._next_available(self.index)
            if i is None:
                live = [a for a in self.accounts if a["num"] not in self.dead]
                if not live:
                    raise self._no_usable_account()
                # all cooling: use the one that frees up first rather than fail
                i = self.accounts.index(min(live, key=lambda a: self.cooling.get(a["num"], 0)))
            if i != self.index:
                self.log(f"using {self.accounts[i]['name']}")
            self.index = i
            return self._credential(self.accounts[i])

    def rate_limited(self, credential, retry_after, *, model=None):
        with self.lock:
            wait = _retry_seconds(retry_after)
            self.cooling[credential["num"]] = self.clock() + wait
            i = self._next_available(self.index)
            if i is None:
                self.log(f"{credential['name']} rate-limited; every account is cooling")
                return None
            self.index = i
            self.log(f"{credential['name']} rate-limited for {wait:.0f}s; switched to {self.accounts[i]['name']}")
            return self._credential(self.accounts[i])

    def record_rate_limit(self, credential, retry_after):
        with self.lock:
            self.cooling[credential["num"]] = self.clock() + _retry_seconds(retry_after)

    def recover(self, rejected, *, model=None):
        """Drop a token the provider rejected and move to the next account.

        When no other account can serve now, this raises instead of returning
        ``None``. ``None`` would send the agent the rejected token's 401, which
        reads as "log in again" even when another account is only rate-limited.
        """
        with self.lock:
            self.dead.add(rejected["num"])
            self.log(f"{rejected['name']} rejected (401); dropped from rotation")
            i = self._next_available(self.index)
            if i is None:
                raise self._no_usable_account()
            self.index = i
            return self._credential(self.accounts[i])

    def status(self):
        with self.lock:
            now = self.clock()
            return [
                {
                    "name": a["name"],
                    "active": i == self.index,
                    "dead": a["num"] in self.dead,
                    "cooling_s": max(0, round(self.cooling.get(a["num"], 0) - now)),
                }
                for i, a in enumerate(self.accounts)
            ]


class SharedRotateProxy(InferenceProxy):
    """The native adapter on a fixed port with a capability every agent shares."""

    # The per-session adapter's advice is to relaunch `cswap run`, which does
    # not apply to agents that share this endpoint.
    no_usable_login_advice = (
        "Register a working setup token with `cswap add-token`, then restart "
        "`cswap rotate-proxy`."
    )

    def __init__(self, credentials, *, port=DEFAULT_PORT, capability, **kw):
        super().__init__(credentials, port=port, **kw)
        self.capability = capability

    def __enter__(self):
        try:
            return super().__enter__()
        except OSError as error:
            raise ClaudeSwitchError(
                f"Cannot listen on 127.0.0.1:{self.port}: {error.strerror or error}. "
                "Another rotate-proxy may already be running; stop it, or choose a "
                f"free port with {PORT_ENVIRONMENT_VARIABLE}."
            ) from error


def capability_path(switcher) -> Path:
    return Path(switcher.backup_dir) / "rotate-capability"


def load_or_create_capability(switcher) -> str:
    """The capability local agents present to the proxy, created on first use.

    Creation is exclusive, so the proxy and a ``rotate-env`` started at the
    same moment agree on one value instead of each writing its own.
    """
    path = capability_path(switcher)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return _read_capability(path)
    except FileNotFoundError:
        raise ClaudeSwitchError(
            "No claude-swap account store exists yet. Register a setup token "
            "with `cswap add-token` first."
        ) from None
    value = secrets.token_urlsafe(32)
    with os.fdopen(fd, "w") as f:
        f.write(value)
    return value


def _read_capability(path: Path) -> str:
    """Read an existing capability, waiting out a creator that is mid-write."""
    deadline = time.monotonic() + CAPABILITY_WRITE_WAIT_S
    while True:
        value = path.read_text().strip()
        if value:
            return value
        if time.monotonic() >= deadline:
            raise ClaudeSwitchError(
                f"{path} is empty. Delete it and run the command again to create "
                "a new capability."
            )
        time.sleep(0.01)


def serve(switcher, *, port=DEFAULT_PORT, log=print) -> None:
    accounts = load_token_accounts(switcher)
    pool = LocalTokenPool(accounts, log=lambda m: log(time.strftime("%H:%M:%S ") + m, flush=True))
    capability = load_or_create_capability(switcher)
    with SharedRotateProxy(pool, port=port, capability=capability) as proxy:
        log(f"rotate-proxy on {proxy.url} with {len(accounts)} accounts: "
            + ", ".join(a["name"] for a in accounts), flush=True)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
