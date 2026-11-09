"""Keep a Vision-proxied session's cached feature flags fresh (issue #40).

Claude Code reads its GrowthBook feature flags from a cache in the global
config (``.claude.json``) and refreshes that cache from
``https://api.anthropic.com/api/eval/<sdk key>``. The host is hard-coded
(2.1.321: the flag client's ``apiHost`` is the constant
``https://api.anthropic.com/``, and no setting or variable replaces it; only
``/v1/messages`` follows ``ANTHROPIC_BASE_URL``), and the bearer is
``CLAUDE_CODE_OAUTH_TOKEN``. Under the Vision adapter that variable holds the
adapter's capability, which Anthropic refuses, so a proxied session never
refreshes its flags and keeps whatever a direct login last cached, for good.
People then see notices the flags would have suppressed (a stale
``tengu_usage_overage_included_models`` said "Fable 5.2 now uses usage
credits").

The adapter never hands native the real token, so the refresh happens beside
the session: in a background thread of the adapter's process, at most every
``REFRESH_AFTER_SECONDS``, Claude Code itself runs ``doctor`` with a real
access token in a throwaway profile. ``doctor`` initializes the flag client and
saves its payload like any start does, and makes no model request (measured on
2.1.321). Then only the flag keys are copied into the session's
``.claude.json``, under Claude Code's own config lock, by an atomic replace
that keeps the file's mode. Everything else the throwaway run wrote (the pool
login's identity, policy files) is discarded with it.

Every failure is soft: the session never waits for a refresh, never sees its
output, and an attempt that produced no flags is not retried for
``RETRY_AFTER_SECONDS``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import stat
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from claude_swap.claude_locks import proper_lockfile
from claude_swap.exceptions import ClaudeCodeLockTimeout
from claude_swap.fsutil import replace_with_retry

# The keys Claude Code saves after a flag fetch, all four together (2.1.321
# ``syncRemoteEvalToDisk``). Nothing else is ever copied.
FLAG_PAYLOAD_KEYS = (
    "cachedGrowthBookFeatures",
    "cachedExperimentFeatures",
    "cachedExperimentData",
)
FETCHED_AT_KEY = "cachedGrowthBookFeaturesAt"  # epoch milliseconds
FLAG_KEYS = (*FLAG_PAYLOAD_KEYS, FETCHED_AT_KEY)
# Identity copied into the throwaway profile so percentage rollouts evaluate
# for this machine, as they would for a direct login on it.
SEED_KEYS = ("userID",)

# The reporter's private router refreshed every 6h; Claude Code itself
# refreshes far more often, but each refresh here costs a credential issuance
# and a native start (about 5 s of CPU-light work).
REFRESH_AFTER_SECONDS = 6 * 3600
# After an attempt, whatever its outcome, launches leave the flags alone this
# long, so a refresh that cannot succeed is not retried by every launch.
RETRY_AFTER_SECONDS = 3600
# A long session re-checks staleness this often.
CHECK_INTERVAL_SECONDS = 15 * 60
# ``doctor`` took about 5 s; it is stopped if it takes much longer.
TIMEOUT_SECONDS = 60.0
DOCTOR_ARGUMENTS = ("doctor",)

# Never written to stderr: native is drawing on that terminal.
_log = logging.getLogger("claude-swap.flag_refresh")
_log.addHandler(logging.NullHandler())


class FlagRefresher:
    """Refresh one native profile's cached flags out of band.

    ``native_command`` runs Claude Code (``[path]``); ``base_env`` is the
    environment native was launched with, adapter variables included, which
    are replaced here; ``token_source`` returns a real access token and is
    called only when a refresh is due.
    """

    def __init__(
        self,
        native_command: Sequence[str],
        config_path: Path,
        base_env: Mapping[str, str],
        token_source: Callable[[], str],
        *,
        marker_dir: Path,
        clock: Callable[[], float] = time.time,
        timeout_seconds: float = TIMEOUT_SECONDS,
    ):
        self.native_command = list(native_command)
        self.config_path = Path(config_path)
        self.base_env = dict(base_env)
        self.token_source = token_source
        digest = hashlib.sha256(str(self.config_path).encode()).hexdigest()[:16]
        self.marker = Path(marker_dir) / f"{digest}.attempt"
        self.clock = clock
        self.timeout_seconds = timeout_seconds
        self.stopping = threading.Event()
        self.thread: threading.Thread | None = None

    # -- background --------------------------------------------------------

    def start(self) -> None:
        self.thread = threading.Thread(
            target=self._loop, name="cswap-flag-refresh", daemon=True
        )
        self.thread.start()

    def stop(self, timeout: float) -> None:
        """Stop checking; a run in progress is ended, its result discarded."""
        self.stopping.set()
        if self.thread is not None:
            self.thread.join(timeout)

    def _loop(self) -> None:
        while not self.stopping.is_set():
            try:
                outcome = self.refresh_if_stale()
                _log.info("Feature flag refresh check: %s", outcome)
            except Exception:
                # A bug here must cost the refresh, never the session.
                _log.warning("Feature flag refresh failed unexpectedly.", exc_info=True)
            self.stopping.wait(CHECK_INTERVAL_SECONDS)

    # -- one check ---------------------------------------------------------

    def refresh_if_stale(self) -> str:
        """Refresh if due; returns the outcome, for the log and tests.

        ``fresh``, ``backoff``, ``no_config``: nothing was due or possible.
        ``refreshed``: the flags were replaced. ``superseded``: a newer
        payload arrived meanwhile and was kept. ``no_flags``: native ran but
        saved no flags. ``failed``: no token, or native failed or timed out.
        ``stopped``: the session ended first.
        """
        config = self._read_config()
        if config is None:
            return "no_config"
        if not self._is_stale(config):
            return "fresh"
        if self._attempted_recently():
            return "backoff"
        self._mark_attempt()
        try:
            token = self.token_source()
        except Exception as error:
            _log.warning("No access token to refresh feature flags with: %r", error)
            return "failed"
        fresh = self._run_native(token)
        if fresh is None:
            return "failed" if not self.stopping.is_set() else "stopped"
        if not fresh:
            _log.warning(
                "Claude Code's `doctor` saved no feature flags; cached flags stay "
                "as they are. A newer Claude Code may no longer fetch them there."
            )
            return "no_flags"
        return self._write_flags(fresh)

    def _is_stale(self, config: dict) -> bool:
        fetched_at = config.get(FETCHED_AT_KEY)
        if type(fetched_at) is not int:
            return True
        return self.clock() - fetched_at / 1000 >= REFRESH_AFTER_SECONDS

    def _attempted_recently(self) -> bool:
        try:
            attempted_at = self.marker.stat().st_mtime
        except OSError:
            return False
        return 0 <= self.clock() - attempted_at < RETRY_AFTER_SECONDS

    def _mark_attempt(self) -> None:
        try:
            self.marker.parent.mkdir(parents=True, exist_ok=True)
            self.marker.touch()
            now = self.clock()
            os.utime(self.marker, (now, now))
        except OSError as error:
            _log.warning("Could not record a feature flag refresh attempt: %s", error)

    def _read_config(self) -> dict | None:
        """The target config, or None when it must not be touched.

        A missing file is native's to create; a symlink or anything but a
        JSON object is left alone rather than replaced.
        """
        path = self.config_path
        try:
            if path.is_symlink() or not path.is_file():
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def _run_native(self, token: str) -> dict | None:
        """Run ``doctor`` in a throwaway profile; return the flags it saved.

        None when it failed or was stopped; ``{}`` when it saved none.
        """
        workdir = Path(tempfile.mkdtemp(prefix="cswap-flags-"))
        try:
            profile = workdir / "profile"
            secure = workdir / "secure"
            profile.mkdir()
            secure.mkdir()
            current = self._read_config() or {}
            seed = {key: current[key] for key in SEED_KEYS if key in current}
            (profile / ".claude.json").write_text(json.dumps(seed), encoding="utf-8")
            # The adapter's URL would also change the flags' evaluation
            # (native reports a non-default API host as an attribute).
            env = dict(self.base_env)
            env.pop("ANTHROPIC_BASE_URL", None)
            env["CLAUDE_CONFIG_DIR"] = str(profile)
            # Its own secure storage, so native never reads or saves a login
            # where the person's own lives.
            env["CLAUDE_SECURESTORAGE_CONFIG_DIR"] = str(secure)
            env["CLAUDE_CODE_OAUTH_TOKEN"] = token
            try:
                child = subprocess.Popen(
                    [*self.native_command, *DOCTOR_ARGUMENTS],
                    env=env,
                    cwd=str(workdir),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError as error:
                _log.warning("Could not start Claude Code to refresh flags: %s", error)
                return None
            if not self._wait(child):
                return None
            try:
                saved = json.loads((profile / ".claude.json").read_text("utf-8"))
            except (OSError, ValueError):
                return {}
            if not isinstance(saved, dict):
                return {}
            fresh = {key: saved[key] for key in FLAG_KEYS if key in saved}
            valid = (
                len(fresh) == len(FLAG_KEYS)
                and isinstance(fresh["cachedGrowthBookFeatures"], dict)
                and bool(fresh["cachedGrowthBookFeatures"])
                and type(fresh[FETCHED_AT_KEY]) is int
            )
            return fresh if valid else {}
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def _wait(self, child: subprocess.Popen) -> bool:
        """Wait for ``child``; False (after ending it) on timeout or stop."""
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                child.wait(timeout=0.1)
                return True
            except subprocess.TimeoutExpired:
                pass
            if self.stopping.is_set() or time.monotonic() >= deadline:
                if not self.stopping.is_set():
                    _log.warning(
                        "Claude Code's `doctor` took over %.0f s; stopped it.",
                        self.timeout_seconds,
                    )
                child.kill()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                return False

    def _write_flags(self, fresh: dict) -> str:
        """Copy the flag keys into the config, under native's config lock."""
        path = self.config_path
        lock_dir = path.parent / (path.name + ".lock")
        try:
            with proper_lockfile(lock_dir):
                current = self._read_config()
                if current is None:
                    return "no_config"
                held_at = current.get(FETCHED_AT_KEY)
                if type(held_at) is int and held_at >= fresh[FETCHED_AT_KEY]:
                    return "superseded"
                current.update(fresh)
                _replace_json(path, current)
        except (ClaudeCodeLockTimeout, OSError) as error:
            _log.warning("Could not save refreshed feature flags: %s", error)
            return "failed"
        return "refreshed"


def _replace_json(path: Path, data: dict) -> None:
    """Atomically replace ``path`` with ``data``, keeping its mode.

    Not ``settings.atomic_write_json``: that narrows the parent directory to
    0700, and this file's parent may be the home directory.
    """
    mode = stat.S_IMODE(path.stat().st_mode)
    fd, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        replace_with_retry(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
