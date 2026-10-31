"""Per-request account ledger: which shared account served each model request.

Implements account ledger format v1, a contract shared with codex-swap and
infy (which uploads each session's ledger beside its transcript, so Vision can
show the account per message). Do not change the format here alone.

- ``${XDG_STATE_HOME:-~/.local/state}/agent-account-ledger/claude/``
- ``<session_id>.jsonl``; ``_unknown.jsonl`` when the request named no session
- one JSON object per line, appended with O_APPEND in a single write, so lines
  from concurrent threads and processes never interleave; files 0600, dirs 0700
- fields ``v, ts, harness, session_id, request_id, account_alias,
  account_email, vision_login_id, status, key_owner_email``, all present,
  null when unknown (``key_owner_email`` is a v1-compatible addition)
- never tokens, keys, prompts or bodies
- files whose mtime is over 30 days old are deleted once per process start

The ledger is bookkeeping: it must never fail or slow a request. Every I/O
error is swallowed after one warning in cswap's log.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

LEDGER_VERSION = 1
HARNESS = "claude"
RETENTION_SECONDS = 30 * 24 * 3600
UNKNOWN_SESSION_FILE = "_unknown.jsonl"
ACCOUNT_DOMAIN = "@infinity.inc"

# A session id becomes a file name, so only plain tokens qualify: Claude sends
# a UUID. The leading alphanumeric keeps ``_unknown`` and dotfiles out of reach.
_SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
_REQUEST_ID = re.compile(r"[\x21-\x7e]{1,256}")
_VISION_LOGIN_ID = re.compile(r"ail_[A-Za-z0-9-]{1,64}")
_MAX_ALIAS_LENGTH = 128
_MAX_EMAIL_LENGTH = 320

_OPEN_FLAGS = (
    os.O_WRONLY
    | os.O_APPEND
    | os.O_CREAT
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
    # Windows: no newline translation, so one line is exactly one write.
    | getattr(os, "O_BINARY", 0)
)

# Runs inside the native adapter, whose terminal native Claude draws on: log to
# cswap's file only, never stderr (see vision_proxy._log).
_log = logging.getLogger("claude-swap.account_ledger")
_log.addHandler(logging.NullHandler())

# POSIX O_APPEND makes each write land whole at the end of the file, across
# threads and processes. Windows' C runtime emulates O_APPEND as a seek then a
# write, so two threads can overwrite each other (measured in CI: a 16-thread
# test read back NUL-filled lines). Serializing appends in this process closes
# that; across processes on Windows, lines can still collide (see README).
_append_lock = threading.Lock()

_pruned_lock = threading.Lock()
_pruned_directories: set[Path] = set()


def default_ledger_dir(harness: str = HARNESS) -> Path:
    """``$XDG_STATE_HOME/agent-account-ledger/<harness>``, else ``~/.local/state/...``."""
    state_home = os.environ.get("XDG_STATE_HOME", "")
    # The XDG Base Directory spec: a relative path is invalid and ignored.
    base = (
        Path(state_home)
        if state_home and os.path.isabs(state_home)
        else Path.home() / ".local" / "state"
    )
    return base / "agent-account-ledger" / harness


@dataclass(frozen=True)
class LedgerEntry:
    """One upstream attempt. Values are as observed; ``record`` validates them."""

    sent_at: datetime
    session_id: object
    request_id: object
    account_alias: object
    account_email: object
    vision_login_id: object
    status: object
    # The Vision email of the person whose API key the adapter uses. Unlike
    # account_email it is a person's address, kept as Vision returns it.
    key_owner_email: object = None


def _match(pattern: re.Pattern[str], value: object) -> str | None:
    if isinstance(value, str) and pattern.fullmatch(value):
        return value
    return None


def person_email(value: object) -> str | None:
    """A person's address as given, if it is one string that can be one."""
    if (
        isinstance(value, str)
        and 3 <= len(value) <= _MAX_EMAIL_LENGTH
        and "@" in value
        and value.isprintable()
        and not any(c.isspace() for c in value)
    ):
        return value
    return None


def _timestamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.") + (
        f"{moment.microsecond // 1000:03d}Z"
    )


class AccountLedger:
    """Appends one line per upstream attempt; never raises from I/O."""

    def __init__(self, directory: Path, *, harness: str = HARNESS) -> None:
        self.directory = Path(directory)
        self.harness = harness
        self._lock = threading.Lock()
        self._io_warned = False
        self._warned_emails: set[str] = set()

    def session_id(self, value: object) -> str | None:
        """The session id a header names, or None when it cannot name a file."""
        return _match(_SESSION_ID, value)

    def record(self, entry: LedgerEntry) -> None:
        session_id = self.session_id(entry.session_id)
        status = entry.status
        line = {
            "v": LEDGER_VERSION,
            "ts": _timestamp(entry.sent_at),
            "harness": self.harness,
            "session_id": session_id,
            "request_id": _match(_REQUEST_ID, entry.request_id),
            "account_alias": (
                entry.account_alias
                if isinstance(entry.account_alias, str)
                and 0 < len(entry.account_alias) <= _MAX_ALIAS_LENGTH
                else None
            ),
            "account_email": self._email(entry.account_email),
            "vision_login_id": _match(_VISION_LOGIN_ID, entry.vision_login_id),
            # bool is an int subclass; True is not an HTTP status.
            "status": (
                status
                if type(status) is int and 100 <= status <= 599
                else None
            ),
            "key_owner_email": person_email(entry.key_owner_email),
        }
        data = (json.dumps(line, ensure_ascii=True) + "\n").encode("ascii")
        name = f"{session_id}.jsonl" if session_id else UNKNOWN_SESSION_FILE
        try:
            self._append(self.directory / name, data)
        except OSError as error:
            self._warn_io(f"could not append to {self.directory / name}: {error}")

    def prune_once(self) -> None:
        """Delete ledger files older than the retention, once per process per dir."""
        with _pruned_lock:
            if self.directory in _pruned_directories:
                return
            _pruned_directories.add(self.directory)
        cutoff = time.time() - RETENTION_SECONDS
        try:
            candidates = list(self.directory.glob("*.jsonl"))
        except OSError as error:
            self._warn_io(f"could not list {self.directory}: {error}")
            return
        for path in candidates:
            try:
                info = path.lstat()
                if info.st_mtime < cutoff and path.is_file() and not path.is_symlink():
                    path.unlink()
            except FileNotFoundError:
                continue  # another process pruned it first
            except OSError as error:
                self._warn_io(f"could not prune {path}: {error}")

    def _email(self, value: object) -> str | None:
        if not isinstance(value, str) or not value:
            return None
        local, at, domain = value.rpartition("@")
        if local and at and "@" + domain.lower() == ACCOUNT_DOMAIN:
            return value
        with self._lock:
            first = value not in self._warned_emails
            self._warned_emails.add(value)
        if first:
            _log.warning(
                "Account ledger: %s is not an %s address; recorded as null.",
                value,
                ACCOUNT_DOMAIN,
            )
        return None

    def _append(self, path: Path, data: bytes) -> None:
        with _append_lock:
            self._append_unlocked(path, data)

    def _append_unlocked(self, path: Path, data: bytes) -> None:
        try:
            descriptor = os.open(path, _OPEN_FLAGS, 0o600)
        except FileNotFoundError:
            self._make_directories()
            descriptor = os.open(path, _OPEN_FLAGS, 0o600)
        try:
            written = os.write(descriptor, data)
        finally:
            os.close(descriptor)
        if written != len(data):
            raise OSError(f"short write ({written} of {len(data)} bytes)")

    def _make_directories(self) -> None:
        # The harness directory and its agent-account-ledger parent are both
        # created private; anything above them is the user's own state home.
        self.directory.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.directory.mkdir(mode=0o700, exist_ok=True)

    def _warn_io(self, message: str) -> None:
        with self._lock:
            if self._io_warned:
                return
            self._io_warned = True
        _log.warning(
            "Account ledger write failed: %s (requests are unaffected; later "
            "ledger errors in this process are not logged).",
            message,
        )
