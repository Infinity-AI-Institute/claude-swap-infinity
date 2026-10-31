"""Token usage reaches the ledger line and the local hourly totals.

Lines are written when a request finishes, so they carry the provider's usage;
the hourly totals have the row shape of Vision's usage report contract v1.
"""

import http.client
import http.server
import json
import os
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest

from claude_swap.account_ledger import AccountLedger
from claude_swap.usage_totals import (
    RETENTION_SECONDS,
    UsageTotals,
    load_machine_id,
    report_body,
)
from claude_swap.vision_proxy import InferenceProxy
from tests.test_account_ledger import (
    ACCOUNT_A,
    ACCOUNT_B,
    ALIASES,
    LOGIN_A,
    LOGIN_B,
    SESSION,
    lines,
)
from tests.test_response_usage import DELTA, FINAL, START, STOP, STREAM

MACHINE_ID = re.compile(
    r"mch_[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)
NON_STREAM = json.dumps(
    {
        "id": "msg_synthetic",
        "model": "claude-sonnet-4-6",
        "usage": {
            "input_tokens": 5,
            "output_tokens": 6,
            "cache_read_input_tokens": 7,
            "cache_creation_input_tokens": 8,
        },
    }
).encode()


@pytest.fixture
def provider():
    """A synthetic provider; the request body names the reply to give."""
    count = iter(range(1, 10_000))
    release = threading.Event()
    release.set()

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def _headers(self, status, content_type, length=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("request-id", f"req_synthetic{next(count):04d}")
            if length is not None:
                self.send_header("Content-Length", str(length))
            self.send_header("Connection", "close")
            self.end_headers()

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            self.close_connection = True
            if self.path.startswith("/v1/messages/count_tokens"):
                reply = b'{"input_tokens": 1234}'
                self._headers(200, "application/json", len(reply))
                self.wfile.write(reply)
                return
            if b"limited" in body and self.headers["Authorization"] == "Bearer token-a":
                self._headers(429, "application/json", 2)
                self.wfile.write(b"{}")
                return
            if b"overloaded" in body:
                reply = b'{"type":"error","error":{"type":"overloaded_error"}}'
                self._headers(529, "application/json", len(reply))
                self.wfile.write(reply)
                return
            if b"plain" in body:
                self._headers(200, "application/json", len(NON_STREAM))
                self.wfile.write(NON_STREAM)
                return
            if b"truncate" in body:
                # Dies mid-stream, after even its final usage: a chunked reply
                # cut off inside a chunk, which the adapter cannot mistake for
                # an end.
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                data = START + DELTA + FINAL
                self.wfile.write(b"%x\r\n" % (len(data) + 100) + data)
                self.wfile.flush()
                self.connection.shutdown(socket.SHUT_RDWR)
                return
            self._headers(200, "text/event-stream")
            if b"slow" in body:
                self.wfile.write(START + DELTA)
                self.wfile.flush()
                release.wait(10)
                self.wfile.write(FINAL + STOP)
                return
            for offset in range(0, len(STREAM), 50):
                self.wfile.write(STREAM[offset : offset + 50])
                self.wfile.flush()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", release
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        worker.join()


def credentials(first=ACCOUNT_A):
    pool = Mock()
    pool.get.return_value = first
    pool.account_alias.side_effect = lambda c: ALIASES.get(c.get("login_id"))
    return pool


def proxy_for(url, tmp_path, pool=None):
    return InferenceProxy(
        pool or credentials(),
        upstream=url,
        ledger=AccountLedger(tmp_path / "ledger"),
        usage_totals=UsageTotals(tmp_path / "totals.json", tmp_path / "machine_id"),
    )


def post(proxy, body, *, path="/v1/messages?beta=true", session=SESSION):
    request = urllib.request.Request(
        proxy.url + path,
        data=body,
        headers={
            "Authorization": "Bearer " + proxy.capability,
            "Content-Type": "application/json",
            "X-Claude-Code-Session-Id": session,
        },
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status, response.read()


def ledger(tmp_path):
    return lines(tmp_path / "ledger" / f"{SESSION}.jsonl")


def totals(tmp_path):
    return json.loads((tmp_path / "totals.json").read_text())


def total_rows(tmp_path):
    """The totals' rows; none when nothing was ever counted."""
    path = tmp_path / "totals.json"
    return json.loads(path.read_text())["rows"] if path.exists() else []


def tokens(line):
    return (
        line["model"],
        line["input_tokens"],
        line["output_tokens"],
        line["cache_read_input_tokens"],
        line["cache_creation_input_tokens"],
    )


def test_a_streamed_reply_passes_through_unchanged_and_its_line_has_usage(
    provider, tmp_path
):
    url, _ = provider
    with proxy_for(url, tmp_path) as proxy:
        assert post(proxy, b'{"model":"claude-opus-4-6","stream":true}') == (
            200,
            STREAM,
        )
    [line] = ledger(tmp_path)
    assert tokens(line) == ("claude-opus-4-6", 12, 57, 4000, 300)
    assert line["reasoning_output_tokens"] is None
    assert line["account_id"] == ACCOUNT_A["account_id"]
    assert line["vision_login_id"] == LOGIN_A
    assert line["status"] == 200


def test_the_line_is_written_when_the_stream_ends_but_dated_when_it_was_sent(
    provider, tmp_path
):
    url, release = provider
    release.clear()
    with proxy_for(url, tmp_path) as proxy:
        before = datetime.now(UTC)
        result = {}
        worker = threading.Thread(
            target=lambda: result.setdefault("reply", post(proxy, b'{"slow":true}'))
        )
        worker.start()
        time.sleep(0.5)
        assert not (tmp_path / "ledger" / f"{SESSION}.jsonl").exists()
        time.sleep(0.6)
        release.set()
        worker.join(10)
    [line] = ledger(tmp_path)
    sent = datetime.strptime(line["ts"], "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
    assert sent - before < timedelta(seconds=0.5)
    assert line["output_tokens"] == 57


def test_a_non_streaming_reply_has_usage(provider, tmp_path):
    url, _ = provider
    with proxy_for(url, tmp_path) as proxy:
        assert post(proxy, b'{"model":"claude-sonnet-4-6","plain":true}') == (
            200,
            NON_STREAM,
        )
    [line] = ledger(tmp_path)
    assert tokens(line) == ("claude-sonnet-4-6", 5, 6, 7, 8)


def test_count_tokens_records_input_only_and_stays_out_of_the_totals(
    provider, tmp_path
):
    url, _ = provider
    with proxy_for(url, tmp_path) as proxy:
        post(
            proxy,
            b'{"model":"claude-opus-4-6"}',
            path="/v1/messages/count_tokens?beta=true",
        )
    [line] = ledger(tmp_path)
    assert tokens(line) == ("claude-opus-4-6", 1234, None, None, None)
    # count_tokens is not inference: it uses no tokens of the account's quota.
    assert total_rows(tmp_path) == []


def test_an_aborted_stream_has_the_model_but_null_usage(provider, tmp_path):
    url, _ = provider
    with proxy_for(url, tmp_path) as proxy:
        with pytest.raises((urllib.error.URLError, OSError, http.client.HTTPException)):
            post(proxy, b'{"model":"claude-opus-4-6","truncate":true}')
    [line] = ledger(tmp_path)
    assert tokens(line) == ("claude-opus-4-6", None, None, None, None)
    assert total_rows(tmp_path) == []


def test_an_error_reply_has_the_requested_model_and_null_usage(provider, tmp_path):
    url, _ = provider
    with proxy_for(url, tmp_path) as proxy:
        with pytest.raises(urllib.error.HTTPError) as error:
            post(proxy, b'{"model":"claude-opus-4-6","overloaded":true}')
        assert error.value.code == 529
    [line] = ledger(tmp_path)
    assert tokens(line) == ("claude-opus-4-6", None, None, None, None)
    assert line["status"] == 529


def test_a_retry_on_another_account_writes_both_lines_in_order(provider, tmp_path):
    url, _ = provider
    pool = credentials()
    pool.rate_limited.return_value = ACCOUNT_B
    with proxy_for(url, tmp_path, pool) as proxy:
        post(proxy, b'{"model":"claude-opus-4-6","limited":true}')
    first, second = ledger(tmp_path)
    assert (first["vision_login_id"], first["status"], first["output_tokens"]) == (
        LOGIN_A,
        429,
        None,
    )
    assert (second["vision_login_id"], second["status"], second["output_tokens"]) == (
        LOGIN_B,
        200,
        57,
    )
    [row] = totals(tmp_path)["rows"]
    assert (row["login_id"], row["requests"]) == (LOGIN_B, 1)


def test_hourly_totals_have_the_report_contract_row_shape(provider, tmp_path):
    url, _ = provider
    with proxy_for(url, tmp_path) as proxy:
        post(proxy, b'{"model":"claude-opus-4-6"}')
        post(proxy, b'{"model":"claude-opus-4-6"}')
        post(proxy, b'{"model":"claude-sonnet-4-6","plain":true}')
    state = totals(tmp_path)
    assert MACHINE_ID.fullmatch(state["machine_id"])
    assert state["v"] == 1
    rows = sorted(state["rows"], key=lambda row: row["model"])
    hour = datetime.now(UTC).strftime("%Y-%m-%dT%H:00:00Z")
    assert rows == [
        {
            "api_key_id": None,
            "bucket_start": hour,
            "account_id": ACCOUNT_A["account_id"],
            "login_id": LOGIN_A,
            "harness": "claude",
            "model": "claude-opus-4-6",
            "requests": 2,
            "input_tokens": 24,
            "output_tokens": 114,
            "cache_read_input_tokens": 8000,
            "cache_creation_input_tokens": 600,
            "reasoning_output_tokens": None,
        },
        {
            "api_key_id": None,
            "bucket_start": hour,
            "account_id": ACCOUNT_A["account_id"],
            "login_id": LOGIN_A,
            "harness": "claude",
            "model": "claude-sonnet-4-6",
            "requests": 1,
            "input_tokens": 5,
            "output_tokens": 6,
            "cache_read_input_tokens": 7,
            "cache_creation_input_tokens": 8,
            "reasoning_output_tokens": None,
        },
    ]


def test_totals_survive_a_restart_and_keep_the_machine_id(provider, tmp_path):
    url, _ = provider
    with proxy_for(url, tmp_path) as proxy:
        post(proxy, b'{"model":"claude-opus-4-6"}')
    machine = totals(tmp_path)["machine_id"]
    with proxy_for(url, tmp_path) as proxy:
        post(proxy, b'{"model":"claude-opus-4-6"}')
    state = totals(tmp_path)
    assert state["machine_id"] == machine
    assert [row["requests"] for row in state["rows"]] == [2]


def test_concurrent_requests_on_two_accounts_are_all_counted(provider, tmp_path):
    url, _ = provider
    pool = Mock()
    picks = iter([ACCOUNT_A, ACCOUNT_B] * 20)
    lock = threading.Lock()

    def pick(**_kwargs):
        with lock:
            return next(picks)

    pool.get.side_effect = pick
    pool.account_alias.side_effect = lambda c: ALIASES.get(c.get("login_id"))
    with proxy_for(url, tmp_path, pool) as proxy:
        threads = [
            threading.Thread(target=post, args=(proxy, b'{"model":"claude-opus-4-6"}'))
            for _ in range(20)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
    assert len(ledger(tmp_path)) == 20
    rows = {row["login_id"]: row for row in totals(tmp_path)["rows"]}
    assert {login: row["requests"] for login, row in rows.items()} == {
        LOGIN_A: 10,
        LOGIN_B: 10,
    }
    assert rows[LOGIN_A]["output_tokens"] == 570


def test_two_processes_add_to_one_totals_file(tmp_path):
    """Each cswap process is its own adapter; their totals must not overwrite."""
    script = f"""
import sys
sys.path.insert(0, {str(Path(__file__).resolve().parents[1] / "src")!r})
from datetime import datetime, UTC
from pathlib import Path
from claude_swap.response_usage import TokenUsage
from claude_swap.usage_totals import UsageRecord, UsageTotals
totals = UsageTotals(Path({str(tmp_path / "totals.json")!r}), Path({str(tmp_path / "machine_id")!r}))
for _ in range(25):
    totals.add([UsageRecord(datetime.now(UTC), None, "aia_x", "ail_x", "claude", "m", TokenUsage(1, 2, 3, 4))])
"""
    import subprocess

    workers = [subprocess.Popen([sys.executable, "-c", script]) for _ in range(4)]
    for worker in workers:
        assert worker.wait(60) == 0
    [row] = totals(tmp_path)["rows"]
    assert (row["requests"], row["output_tokens"]) == (100, 200)


def test_buckets_older_than_the_report_window_are_pruned(tmp_path):
    from claude_swap.response_usage import TokenUsage
    from claude_swap.usage_totals import UsageRecord

    store = UsageTotals(tmp_path / "totals.json", tmp_path / "machine_id")
    now = datetime.now(UTC)
    old = now - timedelta(seconds=RETENTION_SECONDS + 7200)
    usage = TokenUsage(1, 1, 0, 0)
    store.add([UsageRecord(old, None, "aia_x", "ail_x", "claude", "m", usage)])
    store.add([UsageRecord(now, None, "aia_x", "ail_x", "claude", "m", usage)])
    assert [row["bucket_start"] for row in totals(tmp_path)["rows"]] == [
        now.strftime("%Y-%m-%dT%H:00:00Z")
    ]


def test_an_unusable_model_name_is_not_counted(tmp_path):
    from claude_swap.response_usage import TokenUsage
    from claude_swap.usage_totals import UsageRecord

    store = UsageTotals(tmp_path / "totals.json", tmp_path / "machine_id")
    for model in (None, "", "has space", "x" * 129):
        store.add(
            [
                UsageRecord(
                    datetime.now(UTC), None, "aia_x", "ail_x", "claude", model,
                    TokenUsage(1, 1, 0, 0),
                )
            ]
        )
    assert total_rows(tmp_path) == []


def test_a_corrupt_totals_file_is_set_aside_and_requests_carry_on(provider, tmp_path):
    url, _ = provider
    (tmp_path / "totals.json").write_text("{not json")
    with proxy_for(url, tmp_path) as proxy:
        assert post(proxy, b'{"model":"claude-opus-4-6"}')[0] == 200
    assert [row["requests"] for row in totals(tmp_path)["rows"]] == [1]
    assert len(list(tmp_path.glob("totals.json.corrupt-*"))) == 1


def test_unwritable_totals_never_fail_a_request(provider, tmp_path):
    url, _ = provider
    blocker = tmp_path / "blocker"
    blocker.write_text("")
    proxy = InferenceProxy(
        credentials(),
        upstream=url,
        ledger=AccountLedger(tmp_path / "ledger"),
        usage_totals=UsageTotals(blocker / "totals.json", blocker / "machine_id"),
    )
    with proxy:
        for _ in range(2):
            assert post(proxy, b'{"model":"claude-opus-4-6"}') == (200, STREAM)
    assert len(ledger(tmp_path)) == 2


def test_the_machine_id_is_created_once_and_reused(tmp_path):
    first = load_machine_id(tmp_path / "machine_id")
    assert MACHINE_ID.fullmatch(first)
    assert load_machine_id(tmp_path / "machine_id") == first
    if sys.platform != "win32":
        assert (tmp_path / "machine_id").stat().st_mode & 0o777 == 0o600


def test_a_malformed_machine_id_is_replaced(tmp_path):
    (tmp_path / "machine_id").write_text("not-an-id\n")
    replaced = load_machine_id(tmp_path / "machine_id")
    assert MACHINE_ID.fullmatch(replaced)
    assert load_machine_id(tmp_path / "machine_id") == replaced


def test_concurrent_first_use_agrees_on_one_machine_id(tmp_path):
    seen = []
    threads = [
        threading.Thread(target=lambda: seen.append(load_machine_id(tmp_path / "id")))
        for _ in range(16)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(set(seen)) == 1


def test_the_report_body_is_the_contract_request_for_one_key(tmp_path):
    from claude_swap.response_usage import TokenUsage
    from claude_swap.usage_totals import UsageRecord

    store = UsageTotals(tmp_path / "totals.json", tmp_path / "machine_id")
    now = datetime.now(UTC)
    usage = TokenUsage(1, 2, 3, 4)
    store.add(
        [
            UsageRecord(now, "key_one", "aia_x", "ail_x", "claude", "m", usage),
            UsageRecord(now, "key_two", "aia_x", "ail_x", "claude", "m", usage),
        ]
    )
    body = report_body(store.read(), version="9.9", api_key_id="key_one")
    assert body == {
        "reporter": {
            "tool": "claude-swap",
            "version": "9.9",
            "machine_id": store.read()["machine_id"],
        },
        "rows": [
            {
                "bucket_start": now.strftime("%Y-%m-%dT%H:00:00Z"),
                "account_id": "aia_x",
                "login_id": "ail_x",
                "harness": "claude",
                "model": "m",
                "requests": 1,
                "input_tokens": 1,
                "output_tokens": 2,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 4,
                "reasoning_output_tokens": None,
            }
        ],
    }


def test_the_proxy_records_the_vision_key_id_with_each_row(provider, tmp_path):
    url, _ = provider
    proxy = proxy_for(url, tmp_path)
    proxy.key_owner = Mock()
    proxy.key_owner.current.return_value = Mock(
        owner_email="owner-1@infinity.inc", key_id="key_synthetic"
    )
    with proxy:
        post(proxy, b'{"model":"claude-opus-4-6"}')
    [row] = totals(tmp_path)["rows"]
    assert row["api_key_id"] == "key_synthetic"
    [line] = ledger(tmp_path)
    assert line["key_owner_email"] == "owner-1@infinity.inc"


def test_the_default_totals_live_beside_the_ledger(provider, monkeypatch, tmp_path):
    url, _ = provider
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    with InferenceProxy(credentials(), upstream=url) as proxy:
        post(proxy, b'{"model":"claude-opus-4-6"}')
    root = tmp_path / "agent-account-ledger"
    assert (root / "usage-hourly-claude.json").exists()
    assert MACHINE_ID.fullmatch((root / "machine_id").read_text().strip())
    # Not in the per-session directory infy uploads from.
    assert sorted(path.name for path in (root / "claude").iterdir()) == [
        f"{SESSION}.jsonl"
    ]
    assert os.path.isdir(root / "claude")
