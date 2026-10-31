"""Send the hourly usage totals to Vision (usage report contract v1).

``POST /api/ai-accounts/usage/reports`` with the process's own Vision API key
(capability ``agent_accounts:use``, which every swap key holds). Vision
upserts each row under ``(key, machine_id, bucket_start, login_id, model)``
and the row REPLACES what it stored, so a row is sent as its running total
whenever it changed since Vision last accepted it, the current hour included.

Schedule, on a background thread: a first pass at start, then every
``interval`` seconds, earlier after a burst of finished requests, and one
last pass when the adapter exits (bounded by the caller's timeout; the thread
is a daemon, so a hung Vision never holds the process). Nothing here runs on a
request's thread: a request only calls ``nudge`` or ``hold``.

Answers:

- 200: each row's verdict. Accepted rows are marked sent; rejected rows are
  dropped (no resend can change a verdict), their codes logged once each.
- 429, 5xx, a network failure, or an answer that is not the contract's: the
  whole batch is retried after a backoff (60 s doubling to 30 min, or the
  Retry-After Vision names).
- 401, 403: the key may not report. Reporting stops for this process, and
  ``notice`` holds one line for the person, printed when Claude exits.

Limits (``limits`` in the answer, and ``GET /api/ai-accounts/usage/limits/self``)
are provisional in the contract and null today. They are read and kept on
``LocalLimitGate``, which refuses nothing until the limits contract is fixed:
that is where local refusal goes.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Iterable
from typing import Any, Protocol

from claude_swap.exceptions import ClaudeSwitchError
from claude_swap.usage_totals import (
    REPORT_FIELDS,
    REPORTER_TOOL,
    UsageRecord,
    UsageTotals,
    load_machine_id,
    row_key,
)
from claude_swap.vision import VisionClient, VisionError
from claude_swap.vision_key_owner import KeyIdentity

REPORTS_PATH = "/api/ai-accounts/usage/reports"
LIMITS_SELF_PATH = "/api/ai-accounts/usage/limits/self"
MAX_ROWS_PER_REPORT = 500
MAX_REPORT_BYTES = 512 * 1024
REPORT_INTERVAL_SECONDS = 300
# Finished requests with usage that bring the next report forward, and the
# least time between two reports.
BURST_RECORDS = 50
MIN_GAP_SECONDS = 30
BACKOFF_INITIAL_SECONDS = 60
BACKOFF_MAX_SECONDS = 1800

REVOKED_KEY_NOTICE = (
    "cswap: Vision refused this session's usage reports because its Vision API "
    "key was revoked or is unknown. Create a new key at https://vision.infinity.inc "
    "(gear → API keys) and save it with `cswap --set-vision-token`."
)
NOT_PERMITTED_NOTICE = (
    "cswap: Vision refused this session's usage reports: the Vision API key lacks "
    "agent_accounts:use, or your access changed. Mint a key with it at "
    "https://vision.infinity.inc (gear → API keys), or ask a Vision admin."
)

# Runs inside the native adapter: cswap's log file only, never stderr.
_log = logging.getLogger("claude-swap.usage_report")
_log.addHandler(logging.NullHandler())


class KeyIdentitySource(Protocol):
    def current(self) -> KeyIdentity | None: ...


def report_batches(
    rows: Iterable[dict[str, Any]],
    reporter: dict[str, str],
    *,
    max_rows: int = MAX_ROWS_PER_REPORT,
    max_bytes: int = MAX_REPORT_BYTES,
) -> list[dict[str, Any]]:
    """Request bodies of at most ``max_rows`` rows and ``max_bytes`` bytes.

    Each row carries exactly the contract's fields; local fields stay home.
    """
    batches: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    # json.dumps of {"reporter": ..., "rows": []}, plus a comma per extra row.
    base = len(json.dumps({"reporter": reporter, "rows": []}).encode())
    size = base
    for row in rows:
        wire = {field: row.get(field) for field in REPORT_FIELDS}
        row_size = len(json.dumps(wire).encode()) + 2
        if current and (len(current) >= max_rows or size + row_size > max_bytes):
            batches.append({"reporter": reporter, "rows": current})
            current, size = [], base
        current.append(wire)
        size += row_size
    if current:
        batches.append({"reporter": reporter, "rows": current})
    return batches


def fetch_limits(client: VisionClient) -> object:
    """The caller's limits from ``GET .../usage/limits/self``; None today."""
    answer = client.request("GET", LIMITS_SELF_PATH)
    if not isinstance(answer, dict) or "limits" not in answer:
        raise VisionError("service_unavailable")
    return answer["limits"]


class LocalLimitGate:
    """Where cswap will refuse requests locally once a key is over its limit.

    The contract's ``limits`` shape is provisional (null until Vision's limits
    release), so this keeps the latest value and refuses nothing. When the
    shape is fixed, ``refusal`` returns the message to show instead of
    forwarding the request; the adapter already asks it before each request.
    """

    def __init__(self) -> None:
        self.limits: object = None

    def update(self, limits: object) -> None:
        self.limits = limits

    def refusal(self) -> str | None:
        return None


class _Retry(Exception):
    """A failure after which the whole batch is sent again later."""

    def __init__(self, reason: str, retry_after: int | None = None) -> None:
        super().__init__(reason)
        self.retry_after = retry_after


class UsageReporter:
    def __init__(
        self,
        totals: UsageTotals,
        client: VisionClient,
        key_owner: KeyIdentitySource,
        *,
        version: str,
        clock: Callable[[], float] = time.time,
        interval: float = REPORT_INTERVAL_SECONDS,
        burst: int = BURST_RECORDS,
    ) -> None:
        self.totals = totals
        self.client = client
        self.key_owner = key_owner
        self.version = version
        self.clock = clock
        self.interval = interval
        self.burst = burst
        self.limit_gate = LocalLimitGate()
        # One line for the person when Vision refuses the key; None otherwise.
        self.notice: str | None = None
        self.next_attempt_at: float | None = None  # set while backing off
        self._lock = threading.Lock()
        self._report_lock = threading.Lock()
        self._wake = threading.Event()
        self._stopping = False
        self._stopped = False  # Vision refused the key
        self._failures = 0
        self._last_report: float | None = None
        self._since_report = 0
        self._held: list[UsageRecord] = []
        self._logged_codes: set[str] = set()
        self._thread: threading.Thread | None = None

    # -- called from request threads: cheap and non-blocking ------------------

    def nudge(self, count: int = 1) -> None:
        """Note finished requests with usage; a burst brings the report forward."""
        with self._lock:
            self._since_report += count
            burst = self._since_report >= self.burst
        if burst:
            self._wake.set()

    def hold(self, records: list[UsageRecord]) -> None:
        """Keep records whose key is not yet identified, to file once it is."""
        with self._lock:
            self._held.extend(records)

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="cswap-usage-report", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float) -> None:
        """One last report, waiting at most ``timeout`` seconds for it."""
        self._stopping = True
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
        else:
            self._file_held(at_exit=True)

    def _run(self) -> None:
        while not self._stopping:
            self._guarded(self.tick)
            self._wake.wait(timeout=self._seconds_until_due())
            self._wake.clear()
        self._guarded(self._final_pass)

    def _final_pass(self) -> None:
        self._file_held(at_exit=True)
        if not self._stopped:
            self.report_now()

    def _guarded(self, step: Callable[[], None]) -> None:
        try:
            step()
        except Exception:
            # An uncaught thread exception prints to stderr, native's screen.
            _log.warning("Usage report pass failed unexpectedly.", exc_info=True)

    # -- scheduling -----------------------------------------------------------

    def _seconds_until_due(self) -> float:
        now = self.clock()
        if self.next_attempt_at is not None:
            return max(1.0, self.next_attempt_at - now)
        if self._last_report is None:
            return 1.0
        return max(1.0, self._last_report + self.interval - now)

    def _due(self) -> bool:
        now = self.clock()
        if self.next_attempt_at is not None:
            return now >= self.next_attempt_at
        if self._last_report is None:
            return True
        elapsed = now - self._last_report
        with self._lock:
            burst = self._since_report >= self.burst
        return elapsed >= self.interval or (burst and elapsed >= MIN_GAP_SECONDS)

    def tick(self) -> None:
        """Report if one is due (first pass, interval, burst, end of a backoff)."""
        if self._stopped or not self._due():
            return
        self.report_now()

    # -- one pass -------------------------------------------------------------

    def report_now(self) -> None:
        """Send every changed row of the current key now, batch by batch."""
        with self._report_lock:
            if self._stopped:
                return
            self._last_report = self.clock()
            with self._lock:
                self._since_report = 0
            identity = self.key_owner.current()
            key_id = None if identity is None else identity.key_id
            if key_id is None:
                return  # unidentified key: nothing can be attributed yet
            self._file_held(key_id=key_id)
            rows = self.totals.pending(key_id)
            if not rows:
                self._succeeded()
                return
            reporter = {
                "tool": REPORTER_TOOL,
                "version": self.version,
                "machine_id": load_machine_id(self.totals.machine_id_path),
            }
            versions = {row_key(row): row.get("version", 1) for row in rows}
            keys = [row_key(row) for row in rows]
            offset = 0
            try:
                for body in report_batches(rows, reporter):
                    batch_keys = keys[offset : offset + len(body["rows"])]
                    offset += len(body["rows"])
                    self._send(body, batch_keys, versions)
            except _Retry as retry:
                self._backoff(str(retry), retry.retry_after)
                return
            except _Refused as refused:
                self._stop_reporting(refused.notice, str(refused))
                return
            except (OSError, ClaudeSwitchError) as error:
                # The totals file could not be read or settled; try later.
                self._backoff(f"local totals unavailable: {error}", None)
                return
            self._succeeded()

    def _send(
        self,
        body: dict[str, Any],
        keys: list[tuple[Any, ...]],
        versions: dict[tuple[Any, ...], Any],
    ) -> None:
        try:
            answer = self.client.request("POST", REPORTS_PATH, body)
        except VisionError as error:
            if error.status == 401 or error.code == "unauthorized":
                raise _Refused(REVOKED_KEY_NOTICE, f"HTTP {error.status} {error.code}") from None
            if error.status == 403:
                raise _Refused(NOT_PERMITTED_NOTICE, f"HTTP {error.status} {error.code}") from None
            if error.status is None or error.status == 429 or error.status >= 500:
                raise _Retry(
                    f"HTTP {error.status} {error.code}", error.retry_after_seconds
                ) from None
            # 400/413/other: a request this client built wrongly. Resending
            # the same batch cannot help soon, so it waits the longest backoff.
            _log.warning(
                "Vision refused a usage report as malformed (HTTP %s %s).",
                error.status,
                error.code,
            )
            raise _Retry(f"HTTP {error.status} {error.code}", BACKOFF_MAX_SECONDS) from None
        results = _row_results(answer, len(keys))
        if results is None:
            raise _Retry("an answer that is not the contract's")
        sent: dict[tuple[Any, ...], int] = {}
        rejected: set[tuple[Any, ...]] = set()
        for index, result in enumerate(results):
            if result["status"] == "accepted":
                sent[keys[index]] = int(versions[keys[index]])
            else:
                rejected.add(keys[index])
                self._log_rejection(str(result.get("code")))
        self.totals.settle(sent, rejected)
        self.limit_gate.update(answer.get("limits"))

    def _log_rejection(self, code: str) -> None:
        if code in self._logged_codes:
            return
        self._logged_codes.add(code)
        _log.warning(
            "Vision rejected usage report rows (%s); they were dropped.", code
        )

    def _file_held(self, *, key_id: str | None = None, at_exit: bool = False) -> None:
        """File held records under the key now known, or unattributed at exit."""
        if key_id is None and not at_exit:
            return
        if key_id is None:
            identity = self.key_owner.current()
            key_id = None if identity is None else identity.key_id
        with self._lock:
            held, self._held = self._held, []
        if held:
            self.totals.add(
                UsageRecord(
                    record.sent_at, key_id, record.account_id, record.login_id,
                    record.harness, record.model, record.usage,
                )
                for record in held
            )

    def _succeeded(self) -> None:
        self._failures = 0
        self.next_attempt_at = None

    def _backoff(self, reason: str, retry_after: int | None) -> None:
        delay = min(BACKOFF_INITIAL_SECONDS * 2**self._failures, BACKOFF_MAX_SECONDS)
        if retry_after is not None:
            delay = min(max(retry_after, 1), BACKOFF_MAX_SECONDS)
        self._failures += 1
        self.next_attempt_at = self.clock() + delay
        _log.warning("Usage report failed (%s); retrying in %d s.", reason, delay)

    def _stop_reporting(self, notice: str, reason: str) -> None:
        self._stopped = True
        self.notice = notice
        _log.warning("Usage reporting stopped for this session: Vision refused the key (%s).", reason)


class _Refused(Exception):
    def __init__(self, notice: str, reason: str) -> None:
        super().__init__(reason)
        self.notice = notice


def _row_results(answer: object, expected: int) -> list[dict[str, Any]] | None:
    """The per-row verdicts, one per row sent in index order, else None."""
    if not isinstance(answer, dict):
        return None
    rows = answer.get("rows")
    if not isinstance(rows, list) or len(rows) != expected:
        return None
    ordered: list[dict[str, Any] | None] = [None] * expected
    for result in rows:
        if (
            not isinstance(result, dict)
            or type(result.get("index")) is not int
            or not 0 <= result["index"] < expected
            or result.get("status") not in ("accepted", "rejected")
            or ordered[result["index"]] is not None
        ):
            return None
        ordered[result["index"]] = result
    return [result for result in ordered if result is not None]
