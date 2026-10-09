"""One shared local endpoint that rotates setup-token accounts on rate limits.

Every Claude Code process on the machine points at it (``ANTHROPIC_BASE_URL``
plus a fixed local capability as ``CLAUDE_CODE_OAUTH_TOKEN``). The proxy keeps
using one account until the provider answers 429, then marks that account as
cooling down and replays the same request on the next available account, so a
rate limit is absorbed once for every running agent instead of per session.

Only setup-token accounts are pooled: they never refresh, so the proxy holds no
refresh tokens. The upstream, forwarded headers and request handling are the
native adapter's (``vision_proxy.InferenceProxy``).
"""

from __future__ import annotations

import http.server
import json
import os
import secrets
import threading
import time
from pathlib import Path

from claude_swap.exceptions import ClaudeSwitchError
from claude_swap.vision_proxy import InferenceProxy

DEFAULT_PORT = 41871
DEFAULT_COOLDOWN_S = 15 * 60
MAX_COOLDOWN_S = 6 * 3600
TOKEN_DOMAIN = "@token.local"


def _retry_seconds(retry_after) -> float:
    try:
        value = float(retry_after)
    except (TypeError, ValueError):
        return DEFAULT_COOLDOWN_S
    return min(max(value, 30.0), MAX_COOLDOWN_S)


def load_token_accounts(switcher, last=("sweep",)) -> list[dict]:
    """Enabled setup-token accounts in slot order, aliases in ``last`` at the end."""
    seq = json.loads(switcher.sequence_file.read_text())
    accounts = seq.get("accounts", {})
    order = [str(n) for n in seq.get("sequence", [])] or sorted(accounts, key=int)
    pool = []
    for num in order:
        account = accounts.get(num) or {}
        email = account.get("email", "")
        if not email.endswith(TOKEN_DOMAIN) or account.get("disabled"):
            continue
        raw = switcher.read_account_credentials(num, email)
        try:
            token = json.loads(raw)["claudeAiOauth"]["accessToken"]
        except (ValueError, KeyError, TypeError):
            continue
        if not token.startswith("sk-ant-oat"):
            continue  # API keys bill per token; never pool them
        pool.append({"num": num, "name": account.get("alias") or email, "token": token})
    return [a for a in pool if a["name"] not in last] + [a for a in pool if a["name"] in last]


class LocalTokenPool:
    """Sticky rotation: stay on one account until it is rate-limited."""

    def __init__(self, accounts, *, clock=time.monotonic, log=None):
        if not accounts:
            raise ClaudeSwitchError("No setup-token accounts to rotate.")
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

    def get(self, *, model=None):
        with self.lock:
            i = self._next_available(self.index)
            if i is None:
                live = [a for a in self.accounts if a["num"] not in self.dead]
                if not live:
                    raise ClaudeSwitchError("Every setup token was rejected.")
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
        with self.lock:
            self.dead.add(rejected["num"])
            self.log(f"{rejected['name']} rejected (401); dropped from rotation")
            i = self._next_available(self.index)
            if i is None:
                return None
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

    def __init__(self, credentials, *, port=DEFAULT_PORT, capability, **kw):
        super().__init__(credentials, **kw)
        self.capability = capability
        self.port = port

    def __enter__(self):
        original = http.server.ThreadingHTTPServer
        port = self.port

        class FixedPortServer(original):
            allow_reuse_address = True

            def __init__(self, address, handler):
                super().__init__((address[0], port), handler)

        http.server.ThreadingHTTPServer = FixedPortServer
        try:
            return super().__enter__()
        finally:
            http.server.ThreadingHTTPServer = original


def capability_path(switcher) -> Path:
    return Path(switcher.backup_dir) / "rotate-capability"


def load_or_create_capability(switcher) -> str:
    path = capability_path(switcher)
    if path.exists():
        return path.read_text().strip()
    value = secrets.token_urlsafe(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(value)
    return value


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
