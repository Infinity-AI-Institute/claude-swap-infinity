"""Local per-hour token totals, shaped as Vision's usage report rows.

Vision's report contract v1 (``POST /api/ai-accounts/usage/reports``) takes
hourly running totals per ``(bucket_start, login_id, model)``, upserted under
the reporting key and a stable ``machine_id``. This module keeps exactly those
rows on disk, plus three local fields: the key that served them
(``api_key_id``), a ``version`` that grows with every change, and the
``sent_version`` Vision last accepted. A row whose version is ahead of its
sent version is due for (re)sending; ``usage_report`` sends them.

The file is shared by every cswap process on the machine. Each update takes a
cross-process lock, re-reads, adds, and atomically replaces the file, so
concurrent sessions never overwrite each other's counts. Rows are partitioned
by the Vision API key that served them (``api_key_id``, from Vision's
api-keys/self; null until that lookup answers), because the server upserts per
key: one key's report must not carry another key's usage.

Like the ledger, this is bookkeeping: ``add`` never raises, and a failure costs
only the count, after one warning in cswap's log.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from claude_swap.account_ledger import default_ledger_dir
from claude_swap.exceptions import ClaudeSwitchError
from claude_swap.fsutil import read_text_with_retry, replace_with_retry
from claude_swap.locking import FileLock
from claude_swap.response_usage import TokenUsage

STATE_VERSION = 1
REPORTER_TOOL = "claude-swap"
# Vision rejects buckets older than 35 days, so older ones are dropped.
RETENTION_SECONDS = 35 * 24 * 3600
LOCK_TIMEOUT_SECONDS = 5.0
COUNT_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)
# A row's identity: the report's upsert key (minus machine, which is global
# here) plus the key that served it.
ROW_KEY = ("api_key_id", "bucket_start", "login_id", "model")
# A report row, exactly as the contract names its fields, in its order.
REPORT_FIELDS = (
    "bucket_start",
    "account_id",
    "login_id",
    "harness",
    "model",
    "requests",
    *COUNT_FIELDS,
    "reasoning_output_tokens",
)
# The report contract's model rule; other names cannot be reported.
_MODEL = re.compile(r"[A-Za-z0-9._:/-]{1,128}")
_ACCOUNT_ID = re.compile(r"aia_[A-Za-z0-9-]{1,64}")
_LOGIN_ID = re.compile(r"ail_[A-Za-z0-9-]{1,64}")
_MACHINE_ID = re.compile(
    r"mch_[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)

# Runs inside the native adapter: cswap's log file only, never stderr.
_log = logging.getLogger("claude-swap.usage_totals")
_log.addHandler(logging.NullHandler())


def default_totals_path(harness: str = "claude") -> Path:
    """Beside, not inside, the per-session ledger directory infy uploads."""
    return default_ledger_dir(harness).parent / f"usage-hourly-{harness}.json"


def default_machine_id_path() -> Path:
    return default_ledger_dir().parent / "machine_id"


@dataclass(frozen=True)
class UsageRecord:
    """One finished request that reported usage."""

    sent_at: datetime
    api_key_id: str | None
    account_id: str | None
    login_id: str | None
    harness: str
    model: str | None
    usage: TokenUsage


def row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in ROW_KEY)


def _version(row: dict[str, Any], field: str, default: int) -> int:
    # Rows written before versions existed count as changed and never sent.
    value = row.get(field, default)
    return value if type(value) is int else default


def bucket_start(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:00:00Z")


def load_machine_id(path: Path) -> str:
    """This machine's ``mch_<uuid4>``, created on first use and kept.

    Concurrent first uses agree: the id is published with a hard link, which
    fails if another process published first, and then that one is read.
    """
    existing = _read_machine_id(path)
    if existing is not None:
        return existing
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    candidate = f"mch_{uuid.uuid4()}"
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, (candidate + "\n").encode("ascii"))
    finally:
        os.close(descriptor)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            # Another process published first (use its id), or the file is
            # malformed (replace it).
            if _read_machine_id(path) is None:
                replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    published = _read_machine_id(path)
    if published is None:
        raise OSError(f"could not create a machine id at {path}")
    return published


def _read_machine_id(path: Path) -> str | None:
    try:
        text = read_text_with_retry(path).strip()
    except FileNotFoundError:
        return None
    return text if _MACHINE_ID.fullmatch(text) else None


def _match(pattern: re.Pattern[str], value: object) -> str | None:
    if isinstance(value, str) and pattern.fullmatch(value):
        return value
    return None


def _empty_state(machine_id: str) -> dict[str, Any]:
    return {"v": STATE_VERSION, "machine_id": machine_id, "rows": []}


class UsageTotals:
    """The hourly totals file; ``add`` folds finished requests into it."""

    def __init__(
        self,
        path: Path,
        machine_id_path: Path,
        *,
        clock: Any = time.time,
    ) -> None:
        self.path = Path(path)
        self.machine_id_path = Path(machine_id_path)
        self.clock = clock
        self._thread_lock = threading.Lock()
        self._warn_lock = threading.Lock()
        self._warned = False

    def add(self, records: Iterable[UsageRecord]) -> None:
        """Count each record that has usage and reportable ids; never raises."""
        rows = [row for row in map(self._row, records) if row is not None]
        if not rows:
            return
        try:
            with self._thread_lock, FileLock(
                self.path.with_name(self.path.name + ".lock"),
                timeout=LOCK_TIMEOUT_SECONDS,
            ):
                state = self._load_for_update()
                self._merge(state, rows)
                self._write(state)
        except (OSError, ClaudeSwitchError) as error:
            self._warn(f"could not update {self.path}: {error}")

    def read(self) -> dict[str, Any]:
        """The current totals (empty if none yet); for reporting."""
        return self._load(set_aside_corrupt=False) or _empty_state(
            load_machine_id(self.machine_id_path)
        )

    def _row(self, record: UsageRecord) -> dict[str, Any] | None:
        model = _match(_MODEL, record.model)
        account_id = _match(_ACCOUNT_ID, record.account_id)
        login_id = _match(_LOGIN_ID, record.login_id)
        if model is None or account_id is None or login_id is None:
            return None
        usage = record.usage
        return {
            "api_key_id": record.api_key_id,
            "bucket_start": bucket_start(record.sent_at),
            "account_id": account_id,
            "login_id": login_id,
            "harness": record.harness,
            "model": model,
            "requests": 1,
            # The contract's counts are integers; a count the provider left
            # out adds nothing.
            **{field: getattr(usage, field) or 0 for field in COUNT_FIELDS},
            # Anthropic gives no reasoning breakdown (thinking is in output).
            "reasoning_output_tokens": None,
            "version": 1,
            "sent_version": 0,
        }

    def pending(self, api_key_id: str) -> list[dict[str, Any]]:
        """``api_key_id``'s rows changed since Vision last accepted them."""
        state = self._load(set_aside_corrupt=False)
        if state is None:
            return []
        cutoff = self._cutoff()
        return [
            row
            for row in state["rows"]
            if isinstance(row, dict)
            and row.get("api_key_id") == api_key_id
            and str(row.get("bucket_start", "")) >= cutoff
            and _version(row, "version", 1) > _version(row, "sent_version", 0)
        ]

    def settle(
        self,
        sent: dict[tuple[Any, ...], int],
        rejected: set[tuple[Any, ...]],
    ) -> None:
        """Record what Vision accepted (row key -> version sent) and drop what it rejected.

        A row that changed after it was sent keeps its newer version, so it
        stays due. Rejected rows are dropped: no resend can change a verdict.
        """
        with self._thread_lock, FileLock(
            self.path.with_name(self.path.name + ".lock"), timeout=LOCK_TIMEOUT_SECONDS
        ):
            state = self._load_for_update()
            kept = []
            for row in state["rows"]:
                key = row_key(row)
                if key in rejected:
                    continue
                if key in sent and _version(row, "sent_version", 0) < sent[key]:
                    row["sent_version"] = sent[key]
                kept.append(row)
            state["rows"] = kept
            self._write(state)

    def _cutoff(self) -> str:
        return bucket_start(datetime.fromtimestamp(self.clock() - RETENTION_SECONDS, UTC))

    def _merge(self, state: dict[str, Any], rows: list[dict[str, Any]]) -> None:
        cutoff = self._cutoff()
        kept = {
            row_key(row): row
            for row in state["rows"]
            if isinstance(row, dict) and str(row.get("bucket_start", "")) >= cutoff
        }
        for row in rows:
            if row["bucket_start"] < cutoff:
                continue
            key = row_key(row)
            existing = kept.get(key)
            if existing is None:
                kept[key] = row
                continue
            existing["version"] = _version(existing, "version", 1) + 1
            existing["requests"] = int(existing.get("requests", 0)) + 1
            for field in COUNT_FIELDS:
                existing[field] = int(existing.get(field) or 0) + row[field]
        state["rows"] = sorted(
            kept.values(), key=lambda row: tuple(str(row.get(f)) for f in ROW_KEY)
        )

    def _load_for_update(self) -> dict[str, Any]:
        machine_id = load_machine_id(self.machine_id_path)
        state = self._load(set_aside_corrupt=True) or _empty_state(machine_id)
        state["machine_id"] = machine_id
        return state

    def _load(self, *, set_aside_corrupt: bool) -> dict[str, Any] | None:
        try:
            text = read_text_with_retry(self.path)
        except FileNotFoundError:
            return None
        try:
            state = json.loads(text)
            if (
                not isinstance(state, dict)
                or state.get("v") != STATE_VERSION
                or not isinstance(state.get("rows"), list)
            ):
                raise ValueError("not a usage totals file")
        except ValueError:
            if not set_aside_corrupt:
                return None
            # Keep the evidence; counting restarts from zero.
            aside = self.path.with_name(f"{self.path.name}.corrupt-{int(self.clock())}")
            replace_with_retry(self.path, aside)
            self._warn(f"{self.path} was unreadable; moved it to {aside.name}")
            return None
        return state

    def _write(self, state: dict[str, Any]) -> None:
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, (json.dumps(state, indent=1) + "\n").encode("ascii"))
        finally:
            os.close(descriptor)
        try:
            replace_with_retry(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def _warn(self, message: str) -> None:
        with self._warn_lock:
            if self._warned:
                return
            self._warned = True
        _log.warning(
            "Usage totals: %s (requests are unaffected; later errors in this "
            "process are not logged).",
            message,
        )
