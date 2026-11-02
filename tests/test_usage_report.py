"""Uploading the hourly usage totals to Vision (usage report contract v1).

A synthetic Vision on loopback answers ``POST /api/ai-accounts/usage/reports``
and ``GET /api/ai-accounts/usage/limits/self`` the way vision PR #417 does
(``{"data": {"rows": [...], "limits": null}}``), so the real VisionClient,
its retries and its errors are exercised.
"""

import http.server
import json
import logging
import threading
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest

from claude_swap.response_usage import TokenUsage
from claude_swap.usage_report import (
    LocalLimitGate,
    UsageReporter,
    fetch_limits,
    report_batches,
)
from claude_swap.usage_totals import UsageRecord, UsageTotals
from claude_swap.vision import VisionClient
from claude_swap.vision_key_owner import KeyIdentity

KEY = "vsk_" + "1" * 40
KEY_ID = "vkey_00000000-0000-4000-8000-000000000001"
OTHER_KEY_ID = "vkey_00000000-0000-4000-8000-000000000002"
LOGIN_A = "ail_00000000-0000-4000-8000-000000000001"
LOGIN_B = "ail_00000000-0000-4000-8000-000000000002"
ACCOUNT_A = "aia_00000000-0000-4000-8000-000000000001"
ACCOUNT_B = "aia_00000000-0000-4000-8000-000000000002"
CONTRACT_FIELDS = [
    "bucket_start",
    "account_id",
    "login_id",
    "harness",
    "model",
    "requests",
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "reasoning_output_tokens",
]


class SyntheticVision:
    """Records each report; ``answers`` scripts the replies, then it accepts."""

    def __init__(self):
        self.reports: list[dict] = []
        self.answers: list[tuple[int, dict | None, dict]] = []
        self.reject: dict[str, str] = {}  # login_id -> code
        self.limits_calls = 0
        self.hold = threading.Event()
        self.hold.set()
        vision = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def _send(self, status, payload, headers=None):
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                for key, value in (headers or {}).items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/api/ai-accounts/usage/limits/self":
                    vision.limits_calls += 1
                    return self._send(200, {"data": {"limits": None}})
                self._send(404, {"error": {"code": "not_found"}})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                vision.hold.wait(10)
                vision.reports.append(body)
                if vision.answers:
                    status, payload, headers = vision.answers.pop(0)
                    if payload is not None:
                        return self._send(status, payload, headers)
                results = []
                for index, row in enumerate(body["rows"]):
                    code = vision.reject.get(row["login_id"])
                    if code:
                        results.append(
                            {"index": index, "status": "rejected", "code": code,
                             "message": "synthetic"}
                        )
                    else:
                        results.append(
                            {"index": index, "status": "accepted",
                             "permitted_at_report": True}
                        )
                self._send(200, {"data": {"rows": results, "limits": None}})

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def sent_rows(self):
        return [row for report in self.reports for row in report["rows"]]


@pytest.fixture
def vision():
    instance = SyntheticVision()
    try:
        yield instance
    finally:
        instance.hold.set()
        instance.server.shutdown()
        instance.server.server_close()
        instance.thread.join()


class Clock:
    def __init__(self):
        self.now = time.time()

    def __call__(self):
        return self.now


def identity(key_id=KEY_ID):
    lookup = Mock()
    lookup.current.return_value = KeyIdentity("owner-1@infinity.inc", key_id)
    return lookup


def record(login=LOGIN_A, account=ACCOUNT_A, model="claude-opus-4-6", *, key=KEY_ID,
           when=None, usage=TokenUsage(10, 20, 30, 40)):
    return UsageRecord(when or datetime.now(UTC), key, account, login, "claude", model, usage)


def setup(tmp_path, vision, *, key_owner=None, clock=None, **options):
    totals = UsageTotals(tmp_path / "totals.json", tmp_path / "machine_id")
    reporter = UsageReporter(
        totals,
        VisionClient(vision.url, KEY),
        key_owner or identity(),
        version="0.26.0b1+infinity.8",
        clock=clock or Clock(),
        **options,
    )
    return totals, reporter


def file_rows(tmp_path):
    return json.loads((tmp_path / "totals.json").read_text())["rows"]


def test_a_report_sends_the_keys_rows_in_the_contract_shape(tmp_path, vision):
    totals, reporter = setup(tmp_path, vision)
    totals.add([record(), record(), record(LOGIN_B, ACCOUNT_B)])
    reporter.report_now()
    [report] = vision.reports
    assert report["reporter"] == {
        "tool": "claude-swap",
        "version": "0.26.0b1+infinity.8",
        "machine_id": totals.read()["machine_id"],
    }
    rows = sorted(report["rows"], key=lambda row: row["login_id"])
    assert [list(row) for row in rows] == [CONTRACT_FIELDS] * 2
    assert (rows[0]["login_id"], rows[0]["requests"], rows[0]["output_tokens"]) == (
        LOGIN_A,
        2,
        40,
    )
    assert rows[0]["bucket_start"] == datetime.now(UTC).strftime("%Y-%m-%dT%H:00:00Z")


def test_sent_rows_are_not_resent_until_they_change(tmp_path, vision):
    totals, reporter = setup(tmp_path, vision)
    totals.add([record()])
    reporter.report_now()
    reporter.report_now()
    assert len(vision.reports) == 1
    # The hour's running total grows: the whole new total replaces the old.
    totals.add([record()])
    reporter.report_now()
    assert len(vision.reports) == 2
    [row] = vision.reports[1]["rows"]
    assert (row["requests"], row["output_tokens"]) == (2, 40)


def test_sent_marks_survive_a_restart(tmp_path, vision):
    totals, reporter = setup(tmp_path, vision)
    totals.add([record()])
    reporter.report_now()
    _, restarted = setup(tmp_path, vision)
    restarted.report_now()
    assert len(vision.reports) == 1


def test_only_the_current_keys_rows_are_sent(tmp_path, vision):
    totals, reporter = setup(tmp_path, vision)
    totals.add([record(key=OTHER_KEY_ID), record(LOGIN_B, ACCOUNT_B)])
    reporter.report_now()
    assert [row["login_id"] for row in vision.sent_rows()] == [LOGIN_B]
    # The other key's row stays for a process using that key.
    assert {row["api_key_id"] for row in file_rows(tmp_path)} == {KEY_ID, OTHER_KEY_ID}


def test_nothing_is_sent_while_the_key_is_unidentified(tmp_path, vision):
    lookup = Mock()
    lookup.current.return_value = None
    totals, reporter = setup(tmp_path, vision, key_owner=lookup)
    totals.add([record(key=None)])
    reporter.report_now()
    assert vision.reports == []


def test_records_held_until_the_key_is_known_are_then_filed_and_sent(tmp_path, vision):
    lookup = Mock()
    lookup.current.return_value = None
    totals, reporter = setup(tmp_path, vision, key_owner=lookup)
    reporter.hold([record(key=None), record(key=None)])
    reporter.report_now()
    assert vision.reports == []
    assert not (tmp_path / "totals.json").exists()
    lookup.current.return_value = KeyIdentity("owner-1@infinity.inc", KEY_ID)
    reporter.report_now()
    [row] = vision.sent_rows()
    assert row["requests"] == 2
    assert file_rows(tmp_path)[0]["api_key_id"] == KEY_ID


def test_held_records_are_kept_unattributed_at_exit(tmp_path, vision):
    lookup = Mock()
    lookup.current.return_value = None
    totals, reporter = setup(tmp_path, vision, key_owner=lookup)
    reporter.start()
    reporter.hold([record(key=None)])
    reporter.stop(timeout=2)
    [row] = file_rows(tmp_path)
    assert (row["api_key_id"], row["requests"]) == (None, 1)


def test_rejected_rows_are_dropped_and_their_codes_logged_once(tmp_path, vision, caplog):
    vision.reject[LOGIN_B] = "login_not_permitted"
    totals, reporter = setup(tmp_path, vision)
    with caplog.at_level(logging.INFO, logger="claude-swap"):
        totals.add([record(), record(LOGIN_B, ACCOUNT_B)])
        reporter.report_now()
        totals.add([record(LOGIN_B, ACCOUNT_B)])
        reporter.report_now()
    assert [row["login_id"] for row in file_rows(tmp_path)] == [LOGIN_A]
    logged = [r.getMessage() for r in caplog.records if "login_not_permitted" in r.getMessage()]
    assert len(logged) == 1
    assert "@" not in logged[0]


@pytest.mark.parametrize(
    "status,headers",
    [(503, {}), (500, {}), (429, {"Retry-After": "120"})],
)
def test_a_retryable_failure_resends_the_whole_batch_after_a_backoff(
    tmp_path, vision, status, headers
):
    clock = Clock()
    totals, reporter = setup(tmp_path, vision, clock=clock)
    totals.add([record(), record(LOGIN_B, ACCOUNT_B)])
    vision.answers.append((status, {"error": {"code": "usage_unavailable"}}, headers))
    reporter.tick()
    assert len(vision.reports) == 1
    reporter.tick()  # still backing off
    assert len(vision.reports) == 1
    clock.now += 125 if headers else 61
    reporter.tick()
    assert len(vision.reports) == 2
    assert vision.reports[1]["rows"] == vision.reports[0]["rows"]
    assert all(row.get("sent_version") for row in file_rows(tmp_path))


def test_backoff_grows_and_is_capped(tmp_path, vision):
    clock = Clock()
    totals, reporter = setup(tmp_path, vision, clock=clock)
    totals.add([record()])
    delays = []
    for _ in range(8):
        vision.answers.append((503, {"error": {"code": "x"}}, {}))
        before = clock.now
        reporter.report_now()
        delays.append(reporter.next_attempt_at - before)
    assert delays[:3] == [60, 120, 240]
    assert max(delays) == 1800


def test_vision_unreachable_is_retried(tmp_path):
    totals = UsageTotals(tmp_path / "totals.json", tmp_path / "machine_id")
    clock = Clock()
    # A closed port: connection refused.
    reporter = UsageReporter(
        totals, VisionClient("http://127.0.0.1:9", KEY), identity(),
        version="1", clock=clock,
    )
    totals.add([record()])
    reporter.report_now()
    assert reporter.next_attempt_at == clock.now + 60
    assert reporter.notice is None


@pytest.mark.parametrize(
    "status,code,needle",
    [
        (401, "unauthorized", "--set-vision-token"),
        (403, "capability_required", "agent_accounts:use"),
        (403, "not_permitted", "agent_accounts:use"),
    ],
)
def test_a_refused_key_stops_reporting_and_tells_the_user_once(
    tmp_path, vision, status, code, needle
):
    clock = Clock()
    totals, reporter = setup(tmp_path, vision, clock=clock)
    totals.add([record()])
    vision.answers.append((status, {"error": {"code": code}}, {}))
    reporter.report_now()
    sent = len(vision.reports)
    assert reporter.notice is not None and needle in reporter.notice
    assert reporter.notice.startswith("cswap: ")
    assert len(reporter.notice) < 300
    clock.now += 10_000
    reporter.tick()
    reporter.report_now()
    assert len(vision.reports) == sent
    # Rows stay, unsent, for a later process with a working key.
    assert file_rows(tmp_path)[0].get("sent_version", 0) == 0


def test_an_unexpected_answer_is_retried_not_trusted(tmp_path, vision):
    totals, reporter = setup(tmp_path, vision)
    totals.add([record()])
    vision.answers.append((200, {"data": {"rows": "nope", "limits": None}}, {}))
    reporter.report_now()
    assert file_rows(tmp_path)[0].get("sent_version", 0) == 0
    assert reporter.next_attempt_at is not None


def test_batches_hold_at_most_500_rows_and_512_kib():
    rows = [
        {
            "bucket_start": "2026-10-31T19:00:00Z",
            "account_id": ACCOUNT_A,
            "login_id": f"ail_00000000-0000-4000-8000-{index:012d}",
            "harness": "claude",
            "model": "m" * 128,
            "requests": 1,
            "input_tokens": 1,
            "output_tokens": 1,
            "cache_read_input_tokens": 1,
            "cache_creation_input_tokens": 1,
            "reasoning_output_tokens": None,
            "api_key_id": KEY_ID,
            "version": 1,
            "sent_version": 0,
        }
        for index in range(1201)
    ]
    reporter_info = {"tool": "claude-swap", "version": "1", "machine_id": "mch_x"}
    batches = report_batches(rows, reporter_info)
    assert [len(batch["rows"]) for batch in batches] == [500, 500, 201]
    assert all(list(row) == CONTRACT_FIELDS for b in batches for row in b["rows"])
    small = report_batches(rows[:50], reporter_info, max_bytes=4096)
    assert all(len(json.dumps(b).encode()) <= 4096 for b in small)
    assert sum(len(b["rows"]) for b in small) == 50


def test_a_large_backlog_goes_out_in_several_reports(tmp_path, vision):
    totals, reporter = setup(tmp_path, vision)
    totals.add(
        [record(f"ail_00000000-0000-4000-8000-{index:012d}") for index in range(1100)]
    )
    reporter.report_now()
    assert [len(report["rows"]) for report in vision.reports] == [500, 500, 100]
    assert all(row["sent_version"] == 1 for row in file_rows(tmp_path))


def test_the_schedule_reports_every_interval_and_early_after_a_burst(tmp_path, vision):
    clock = Clock()
    totals, reporter = setup(tmp_path, vision, clock=clock, interval=300, burst=5)
    totals.add([record()])
    reporter.tick()
    assert len(vision.reports) == 1  # first pass is due at once
    totals.add([record()])
    reporter.tick()
    assert len(vision.reports) == 1
    for _ in range(5):
        reporter.nudge()
    clock.now += 31  # a burst still waits a short minimum gap
    reporter.tick()
    assert len(vision.reports) == 2
    totals.add([record()])
    clock.now += 300
    reporter.tick()
    assert len(vision.reports) == 3


def test_stop_flushes_once_and_never_waits_long(tmp_path, vision):
    totals, reporter = setup(tmp_path, vision)
    reporter.start()
    totals.add([record()])
    started = time.monotonic()
    reporter.stop(timeout=2)
    assert time.monotonic() - started < 2.5
    assert len(vision.sent_rows()) == 1


def test_stop_gives_up_on_a_hung_vision_within_the_bound(tmp_path, vision):
    totals, reporter = setup(tmp_path, vision)
    reporter.start()
    totals.add([record()])
    vision.hold.clear()
    started = time.monotonic()
    reporter.stop(timeout=1)
    assert time.monotonic() - started < 1.5


def test_buckets_too_old_to_report_are_pruned_not_sent(tmp_path, vision):
    totals, reporter = setup(tmp_path, vision)
    old = datetime.now(UTC) - timedelta(days=36)
    totals.add([record(when=old), record()])
    reporter.report_now()
    assert len(vision.sent_rows()) == 1


def test_the_limits_in_a_report_answer_are_kept_for_the_gate(tmp_path, vision):
    totals, reporter = setup(tmp_path, vision)
    totals.add([record()])
    provisional = [{"window": "day", "state": "ok"}]
    vision.answers.append(
        (
            200,
            {
                "data": {
                    "rows": [{"index": 0, "status": "accepted", "permitted_at_report": True}],
                    "limits": provisional,
                }
            },
            {},
        )
    )
    reporter.report_now()
    assert reporter.limit_gate.limits == provisional
    # Refusing locally waits for the limits contract.
    assert reporter.limit_gate.refusal() is None


def test_the_limits_read_returns_null_today(vision):
    assert fetch_limits(VisionClient(vision.url, KEY)) is None
    assert vision.limits_calls == 1


def test_the_gate_refuses_nothing_until_limits_are_specified():
    gate = LocalLimitGate()
    assert gate.refusal() is None
    gate.update([{"window": "day", "state": "exceeded"}])
    assert gate.refusal() is None
