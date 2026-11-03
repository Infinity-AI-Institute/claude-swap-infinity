"""Uploading the hourly usage totals to Vision (usage report contract v1).

A synthetic Vision on loopback answers ``POST /api/ai-accounts/usage/reports``
and ``GET /api/ai-accounts/usage/limits/self`` the way vision PRs #417 and
#421 do (``{"data": {"rows": [...], "limits": ...}}``), so the real
VisionClient, its retries and its errors are exercised.
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
    LIMITS_RETRY_SECONDS,
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
        # A report holding any of these logins is refused whole, as malformed (400).
        self.malformed: set[str] = set()
        self.limits_calls = 0
        # What both the limits read and every report answer say (PR #421).
        self.limits: list | None = None
        self.limits_status = 200
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
                    if vision.limits_status != 200:
                        return self._send(
                            vision.limits_status, {"error": {"code": "service_unavailable"}}
                        )
                    return self._send(200, {"data": {"limits": vision.limits}})
                self._send(404, {"error": {"code": "not_found"}})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                vision.hold.wait(10)
                vision.reports.append(body)
                if any(row["login_id"] in vision.malformed for row in body["rows"]):
                    return self._send(400, {"error": {"code": "invalid_request"}})
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
                self._send(200, {"data": {"rows": results, "limits": vision.limits}})

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


def setup(tmp_path, vision, *, key_owner=None, clock=None, key=KEY, **options):
    totals = UsageTotals(tmp_path / "totals.json", tmp_path / "machine_id")
    reporter = UsageReporter(
        totals,
        VisionClient(vision.url, key),
        key_owner or identity(),
        version="0.26.0b1+infinity.8",
        clock=clock or Clock(),
        **options,
    )
    return totals, reporter


def eventually(condition, seconds=5.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return condition()


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


def unknown_key():
    lookup = Mock()
    lookup.current.return_value = None
    return lookup


def test_records_held_until_the_key_is_known_are_then_filed_and_sent(tmp_path, vision):
    lookup = unknown_key()
    totals, reporter = setup(tmp_path, vision, key_owner=lookup)
    reporter.hold([record(key=None), record(key=None)])
    reporter.report_now()
    assert vision.reports == []
    assert file_rows(tmp_path) == []  # held, not filed under any key
    lookup.current.return_value = KeyIdentity("owner-1@infinity.inc", KEY_ID)
    reporter.report_now()
    [row] = vision.sent_rows()
    assert row["requests"] == 2
    assert file_rows(tmp_path)[0]["api_key_id"] == KEY_ID
    assert totals.read()["held"] == []


def test_held_usage_is_on_disk_and_survives_a_restart(tmp_path, vision):
    # A process killed before its key was named loses nothing: the next one files it.
    _, reporter = setup(tmp_path, vision, key_owner=unknown_key())
    reporter.hold([record(key=None), record(LOGIN_B, ACCOUNT_B, key=None)])
    _, restarted = setup(tmp_path, vision)
    restarted.report_now()
    assert sorted(row["login_id"] for row in vision.sent_rows()) == [LOGIN_A, LOGIN_B]


def test_held_usage_of_another_key_is_never_filed_under_this_one(tmp_path, vision):
    _, reporter = setup(tmp_path, vision, key_owner=unknown_key())
    reporter.hold([record(key=None)])
    totals, other = setup(
        tmp_path, vision, key="vsk_" + "2" * 40, key_owner=identity(OTHER_KEY_ID)
    )
    other.report_now()
    assert vision.reports == []
    [held] = totals.read()["held"]
    assert "vsk_" not in json.dumps(held)  # a fingerprint, never the key
    # The key it was recorded with still files it.
    _, same = setup(tmp_path, vision)
    same.report_now()
    assert len(vision.sent_rows()) == 1


def test_held_usage_stays_held_at_exit_when_the_key_is_still_unknown(tmp_path, vision):
    totals, reporter = setup(tmp_path, vision, key_owner=unknown_key())
    reporter.start()
    reporter.hold([record(key=None)])
    reporter.stop(timeout=2)
    assert file_rows(tmp_path) == []
    [held] = totals.read()["held"]
    assert held["requests"] == 1
    assert vision.reports == []


def test_a_pass_without_a_known_key_does_not_reset_the_timer(tmp_path, vision):
    lookup = unknown_key()
    totals, reporter = setup(tmp_path, vision, key_owner=lookup)
    totals.add([record()])
    reporter.tick()  # the first pass, before api-keys/self has answered
    assert vision.reports == []
    lookup.current.return_value = KeyIdentity("owner-1@infinity.inc", KEY_ID)
    reporter.tick()  # no clock advance: the first real report is still due
    assert len(vision.reports) == 1


def test_a_burst_inside_the_minimum_gap_is_deferred_not_dropped(tmp_path, vision):
    clock = Clock()
    totals, reporter = setup(tmp_path, vision, clock=clock, interval=300, burst=5)
    totals.add([record()])
    reporter.tick()
    clock.now += 10
    reporter.nudge(5)
    # The loop waits until the minimum gap has passed, not the whole interval.
    assert reporter._seconds_until_due() == pytest.approx(20)


def test_the_running_loop_reports_a_burst_once_the_gap_has_passed(tmp_path, vision):
    totals, reporter = setup(
        tmp_path, vision, clock=time.time, interval=300, burst=2, min_gap=1
    )
    totals.add([record()])
    reporter.start()
    try:
        assert eventually(lambda: len(vision.reports) == 1)
        totals.add([record(LOGIN_B, ACCOUNT_B)])
        reporter.nudge(2)
        assert eventually(lambda: len(vision.reports) == 2)  # well inside 300 s
    finally:
        reporter.stop(timeout=2)


def test_a_batch_refused_as_malformed_is_split_so_good_rows_still_go(tmp_path, vision):
    bad = "ail_00000000-0000-4000-8000-0000000000ff"
    vision.malformed.add(bad)
    clock = Clock()
    totals, reporter = setup(tmp_path, vision, clock=clock)
    totals.add([record(), record(bad), record(LOGIN_B, ACCOUNT_B)])
    reporter.report_now()
    sent = {row["login_id"]: row["sent_version"] for row in file_rows(tmp_path)}
    assert sent == {LOGIN_A: 1, bad: 0, LOGIN_B: 1}
    # The refused row stays, unsent, and is tried again after the longest backoff.
    assert reporter.next_attempt_at == clock.now + 1800


def many_rows(tmp_path, count):
    state = json.loads((tmp_path / "totals.json").read_text())
    [template] = state["rows"]
    logins = [f"ail_00000000-0000-4000-8000-{index:012d}" for index in range(count)]
    state["rows"] = [{**template, "login_id": login} for login in logins]
    (tmp_path / "totals.json").write_text(json.dumps(state))
    return logins


def test_a_refused_row_does_not_hold_back_later_batches(tmp_path, vision):
    totals, reporter = setup(tmp_path, vision)
    totals.add([record()])
    logins = many_rows(tmp_path, 700)
    vision.malformed.add(logins[3])  # in the first batch of 500
    reporter.report_now()
    unsent = [row["login_id"] for row in file_rows(tmp_path) if row["sent_version"] == 0]
    assert unsent == [logins[3]]
    assert len(vision.reports) < 40  # isolating one row costs a few requests


def test_isolating_refused_rows_spends_a_bounded_number_of_requests(tmp_path, vision):
    totals, reporter = setup(tmp_path, vision)
    totals.add([record()])
    logins = many_rows(tmp_path, 600)
    vision.malformed.update(logins)  # every row: not a row problem at all
    reporter.report_now()
    assert len(vision.reports) <= 2 + 32
    assert all(row["sent_version"] == 0 for row in file_rows(tmp_path))  # nothing dropped


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


def exceeded_day():
    return [
        {
            "window": "day",
            "resets_at": (datetime.now(UTC) + timedelta(hours=3)).isoformat(),
            "limit": {"unit": "usd_estimated", "value": 50},
            "used": {"value": 52.5, "as_of": datetime.now(UTC).isoformat()},
            "state": "exceeded",
        }
    ]


def test_the_limits_in_a_report_answer_refuse_locally(tmp_path, vision):
    totals, reporter = setup(tmp_path, vision)
    totals.add([record()])
    vision.limits = exceeded_day()
    reporter.report_now()
    assert "daily usage limit (52.50 of 50.00 USD at API list price" in (
        reporter.limit_gate.refusal() or ""
    )


def test_an_answer_without_a_limits_field_keeps_the_known_limits(tmp_path, vision):
    totals, reporter = setup(tmp_path, vision)
    reporter.limit_gate.update(exceeded_day())
    totals.add([record()])
    vision.answers.append(
        (200, {"data": {"rows": [{"index": 0, "status": "accepted"}]}}, {})
    )
    reporter.report_now()
    assert reporter.limit_gate.refusal() is not None


def test_usage_is_still_reported_while_the_limit_refuses(tmp_path, vision):
    totals, reporter = setup(tmp_path, vision)
    reporter.limit_gate.update(exceeded_day())
    totals.add([record()])
    reporter.report_now()
    assert len(vision.sent_rows()) == 1


def test_the_limits_read_follows_the_contract(vision):
    assert fetch_limits(VisionClient(vision.url, KEY)) is None
    vision.limits = exceeded_day()
    assert fetch_limits(VisionClient(vision.url, KEY)) == vision.limits
    assert vision.limits_calls == 2


def test_limits_are_read_at_start_before_the_key_is_known_then_each_interval(
    tmp_path, vision
):
    clock = Clock()
    _, reporter = setup(tmp_path, vision, key_owner=unknown_key(), clock=clock)
    vision.limits = exceeded_day()
    reporter.tick()
    assert vision.limits_calls == 1
    assert reporter.limit_gate.refusal() is not None
    reporter.tick()
    assert vision.limits_calls == 1  # not on every pass
    clock.now += reporter.interval
    vision.limits = None
    reporter.tick()
    assert vision.limits_calls == 2
    assert reporter.limit_gate.refusal() is None


def test_a_report_answer_counts_as_a_limits_read(tmp_path, vision):
    clock = Clock()
    totals, reporter = setup(tmp_path, vision, clock=clock)
    reporter.tick()  # start: reads limits, reports nothing
    totals.add([record()])
    clock.now += reporter.interval
    reporter.tick()  # the report's answer carries the limits
    assert (vision.limits_calls, len(vision.reports)) == (1, 1)


def test_a_failed_limits_read_is_retried_soon_and_keeps_what_was_known(
    tmp_path, vision
):
    clock = Clock()
    _, reporter = setup(tmp_path, vision, key_owner=unknown_key(), clock=clock)
    reporter.limit_gate.update(exceeded_day())
    vision.limits_status = 503
    reporter.tick()
    assert vision.limits_calls == 1
    assert reporter.limit_gate.refusal() is not None  # never lifted by an outage
    assert reporter._seconds_until_due() <= LIMITS_RETRY_SECONDS
    clock.now += LIMITS_RETRY_SECONDS
    vision.limits_status = 200
    reporter.tick()
    assert vision.limits_calls == 2
    assert reporter.limit_gate.refusal() is None


def test_limits_are_still_read_after_vision_refused_the_reports(tmp_path, vision):
    clock = Clock()
    totals, reporter = setup(tmp_path, vision, clock=clock)
    totals.add([record()])
    vision.answers.append((403, {"error": {"code": "capability_required"}}, {}))
    reporter.tick()
    assert reporter.notice is not None
    clock.now += reporter.interval
    vision.limits = exceeded_day()
    reporter.tick()
    assert reporter.limit_gate.refusal() is not None
