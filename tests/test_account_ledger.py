"""The per-request account ledger: which shared account served each request.

The format is a contract shared with infy's uploader and codex-swap (account
ledger format v1); these tests pin it as infy will read it.
"""

import http.server
import json
import logging
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from claude_swap.account_ledger import (
    RETENTION_SECONDS,
    AccountLedger,
    LedgerEntry,
    default_ledger_dir,
)
from claude_swap.vision_proxy import InferenceProxy

SESSION = "6f1c2a4e-0b7d-4c8e-9f3a-2d5e7b9c1a03"
LOGIN_A = "ail_00000000-0000-4000-8000-000000000001"
LOGIN_B = "ail_00000000-0000-4000-8000-000000000002"
FIELDS = [
    "v",
    "ts",
    "harness",
    "session_id",
    "request_id",
    "account_alias",
    "account_email",
    "vision_login_id",
    "status",
    "key_owner_email",
]
TS = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$")


def credential(login, email, token):
    return {
        "accessToken": token,
        "login_id": login,
        "email": email,
        "account_id": "aia_" + login[4:],
    }


ACCOUNT_A = credential(LOGIN_A, "luke-1@infinity.inc", "token-a")
ACCOUNT_B = credential(LOGIN_B, "luke-2@infinity.inc", "token-b")
ALIASES = {LOGIN_A: "luke-1", LOGIN_B: "luke-2"}


@pytest.fixture
def upstream():
    """A synthetic provider that answers with a request-id per request."""
    count = iter(range(1, 1000))

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            request_id = f"req_synthetic{next(count):04d}"
            if b"hangup" in body:
                self.close_connection = True
                return
            if b"limited" in body and self.headers["Authorization"] == "Bearer token-a":
                self.send_response(429)
                self.send_header("Retry-After", "120")
                self.send_header("request-id", request_id)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if b"reject" in body and self.headers["Authorization"] == "Bearer token-a":
                self.send_response(401)
                self.send_header("request-id", request_id)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            payload = b"data: synthetic\n\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("request-id", request_id)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def pool(first=ACCOUNT_A):
    credentials = Mock()
    credentials.get.return_value = first
    credentials.account_alias.side_effect = lambda c: ALIASES.get(c.get("login_id"))
    return credentials


def send(proxy, *, session=SESSION, body=b'{"model":"claude-opus-4-6"}'):
    headers = {
        "Authorization": "Bearer " + proxy.capability,
        "Content-Type": "application/json",
    }
    if session is not None:
        headers["X-Claude-Code-Session-Id"] = session
    request = urllib.request.Request(
        proxy.url + "/v1/messages?beta=true", data=body, headers=headers
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, response.read()


def lines(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_a_request_with_a_session_header_writes_one_line_naming_its_account(
    upstream, tmp_path
):
    with InferenceProxy(
        pool(), upstream=upstream, ledger=AccountLedger(tmp_path)
    ) as proxy:
        assert send(proxy) == (200, b"data: synthetic\n\n")
    [line] = lines(tmp_path / f"{SESSION}.jsonl")
    assert list(line) == FIELDS
    assert TS.match(line.pop("ts"))
    assert line == {
        "v": 1,
        "harness": "claude",
        "session_id": SESSION,
        "request_id": "req_synthetic0001",
        "account_alias": "luke-1",
        "account_email": "luke-1@infinity.inc",
        "vision_login_id": LOGIN_A,
        "status": 200,
        "key_owner_email": None,
    }
    assert sorted(path.name for path in tmp_path.iterdir()) == [f"{SESSION}.jsonl"]


def test_ts_is_when_the_request_was_sent_in_utc(upstream, tmp_path):
    before = datetime.now(UTC).replace(microsecond=0)
    with InferenceProxy(
        pool(), upstream=upstream, ledger=AccountLedger(tmp_path)
    ) as proxy:
        send(proxy)
    after = datetime.now(UTC)
    [line] = lines(tmp_path / f"{SESSION}.jsonl")
    sent = datetime.strptime(line["ts"], "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
    assert before <= sent <= after


def test_no_secret_reaches_the_ledger(upstream, tmp_path):
    with InferenceProxy(
        pool(), upstream=upstream, ledger=AccountLedger(tmp_path)
    ) as proxy:
        send(proxy, body=b'{"model":"m","messages":"synthetic-prompt-text"}')
        capability = proxy.capability
    text = (tmp_path / f"{SESSION}.jsonl").read_text()
    for secret in ("token-a", capability, "synthetic-prompt-text", "aia_"):
        assert secret not in text


@pytest.mark.parametrize("session", [None, "", "../../escape", "_unknown", "a/b"])
def test_a_request_without_a_usable_session_id_goes_to_unknown(
    upstream, tmp_path, session
):
    with InferenceProxy(
        pool(), upstream=upstream, ledger=AccountLedger(tmp_path)
    ) as proxy:
        send(proxy, session=session)
    [line] = lines(tmp_path / "_unknown.jsonl")
    assert line["session_id"] is None
    assert line["status"] == 200
    assert [path.name for path in tmp_path.iterdir()] == ["_unknown.jsonl"]
    assert not (tmp_path.parent / "escape").exists()


def test_a_rate_limit_retry_on_a_second_account_writes_one_line_per_attempt(
    upstream, tmp_path
):
    credentials = pool()
    credentials.rate_limited.return_value = ACCOUNT_B
    with InferenceProxy(
        credentials, upstream=upstream, ledger=AccountLedger(tmp_path)
    ) as proxy:
        assert send(proxy, body=b'{"limited":true}')[0] == 200
    first, second = lines(tmp_path / f"{SESSION}.jsonl")
    assert (first["vision_login_id"], first["status"], first["request_id"]) == (
        LOGIN_A,
        429,
        "req_synthetic0001",
    )
    assert (second["vision_login_id"], second["status"], second["request_id"]) == (
        LOGIN_B,
        200,
        "req_synthetic0002",
    )
    assert second["account_alias"] == "luke-2"
    assert first["ts"] <= second["ts"]


def test_a_401_recovery_replay_writes_one_line_per_attempt(upstream, tmp_path):
    credentials = pool()
    credentials.recover.return_value = ACCOUNT_B
    with InferenceProxy(
        credentials, upstream=upstream, ledger=AccountLedger(tmp_path)
    ) as proxy:
        assert send(proxy, body=b'{"reject":true}')[0] == 200
    assert [
        (line["vision_login_id"], line["status"])
        for line in lines(tmp_path / f"{SESSION}.jsonl")
    ] == [(LOGIN_A, 401), (LOGIN_B, 200)]


def test_a_connection_failure_writes_a_line_with_no_status(upstream, tmp_path):
    with InferenceProxy(
        pool(), upstream=upstream, ledger=AccountLedger(tmp_path)
    ) as proxy:
        with pytest.raises(urllib.error.HTTPError) as error:
            send(proxy, body=b'{"hangup":true}')
        assert error.value.code == 502
    [line] = lines(tmp_path / f"{SESSION}.jsonl")
    assert (line["status"], line["request_id"]) == (None, None)
    assert line["vision_login_id"] == LOGIN_A


def test_a_request_refused_before_any_upstream_attempt_writes_nothing(
    upstream, tmp_path
):
    with InferenceProxy(
        pool(), upstream=upstream, ledger=AccountLedger(tmp_path)
    ) as proxy:
        with pytest.raises(urllib.error.HTTPError):
            send(proxy, body=b"not-json")
    assert list(tmp_path.iterdir()) == []


def test_an_unwritable_ledger_never_fails_a_request_and_warns_once(
    upstream, tmp_path, caplog
):
    # A regular file where the directory should be: unwritable on every OS.
    blocker = tmp_path / "blocker"
    blocker.write_text("")
    ledger = AccountLedger(blocker / "claude")
    with caplog.at_level(logging.WARNING, logger="claude-swap"):
        with InferenceProxy(pool(), upstream=upstream, ledger=ledger) as proxy:
            for _ in range(3):
                assert send(proxy) == (200, b"data: synthetic\n\n")
    warnings = [r for r in caplog.records if "ledger" in r.getMessage().lower()]
    assert len(warnings) == 1
    assert warnings[0].levelno == logging.WARNING


def test_a_non_infinity_email_is_recorded_as_null_with_a_warning(
    upstream, tmp_path, caplog
):
    outside = credential(LOGIN_A, "someone@example.com", "token-a")
    with caplog.at_level(logging.WARNING, logger="claude-swap"):
        with InferenceProxy(
            pool(outside), upstream=upstream, ledger=AccountLedger(tmp_path)
        ) as proxy:
            send(proxy)
            send(proxy)
    rows = lines(tmp_path / f"{SESSION}.jsonl")
    assert [row["account_email"] for row in rows] == [None, None]
    assert [row["vision_login_id"] for row in rows] == [LOGIN_A, LOGIN_A]
    warned = [r for r in caplog.records if "@infinity.inc" in r.getMessage()]
    assert len(warned) == 1


def test_uppercase_infinity_domain_is_accepted(tmp_path):
    ledger = AccountLedger(tmp_path)
    ledger.record(entry(email="Luke-1@Infinity.Inc"))
    [line] = lines(tmp_path / f"{SESSION}.jsonl")
    assert line["account_email"] == "Luke-1@Infinity.Inc"


def test_lookalike_domains_are_rejected(tmp_path):
    ledger = AccountLedger(tmp_path)
    for email in ("a@infinity.inc.evil", "a@notinfinity.inc", "infinity.inc", 7):
        ledger.record(entry(email=email))
    assert [row["account_email"] for row in lines(tmp_path / f"{SESSION}.jsonl")] == [
        None
    ] * 4


def entry(**overrides):
    values = {
        "sent_at": datetime(2026, 10, 31, 17, 0, 0, 123456, tzinfo=UTC),
        "session_id": SESSION,
        "request_id": "req_x",
        "account_alias": "luke-1",
        "email": "luke-1@infinity.inc",
        "vision_login_id": LOGIN_A,
        "status": 200,
    }
    values.update(overrides)
    return LedgerEntry(
        sent_at=values["sent_at"],
        session_id=values["session_id"],
        request_id=values["request_id"],
        account_alias=values["account_alias"],
        account_email=values["email"],
        vision_login_id=values["vision_login_id"],
        status=values["status"],
    )


def test_ts_formats_milliseconds_with_z(tmp_path):
    AccountLedger(tmp_path).record(entry())
    [line] = lines(tmp_path / f"{SESSION}.jsonl")
    assert line["ts"] == "2026-10-31T17:00:00.123Z"


def test_malformed_values_become_null_rather_than_reaching_the_file(tmp_path):
    AccountLedger(tmp_path).record(
        entry(
            request_id="x" * 1000,
            account_alias=Mock(),
            vision_login_id="aia_wrong-prefix",
            status=True,
        )
    )
    [line] = lines(tmp_path / f"{SESSION}.jsonl")
    assert line["request_id"] is None
    assert line["account_alias"] is None
    assert line["vision_login_id"] is None
    assert line["status"] is None


def test_concurrent_writers_never_interleave_lines(tmp_path):
    ledger = AccountLedger(tmp_path)

    def write(worker):
        for index in range(50):
            ledger.record(entry(request_id=f"req_{worker}_{index}"))

    threads = [threading.Thread(target=write, args=(n,)) for n in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    rows = lines(tmp_path / f"{SESSION}.jsonl")
    assert len(rows) == 16 * 50
    assert len({row["request_id"] for row in rows}) == 16 * 50


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_files_are_private_to_the_user(tmp_path):
    directory = tmp_path / "agent-account-ledger" / "claude"
    AccountLedger(directory).record(entry())
    assert (directory / f"{SESSION}.jsonl").stat().st_mode & 0o777 == 0o600
    assert directory.stat().st_mode & 0o777 == 0o700
    assert directory.parent.stat().st_mode & 0o777 == 0o700


def test_appends_to_an_existing_session_file(tmp_path):
    AccountLedger(tmp_path).record(entry(request_id="req_one"))
    AccountLedger(tmp_path).record(entry(request_id="req_two"))
    assert [row["request_id"] for row in lines(tmp_path / f"{SESSION}.jsonl")] == [
        "req_one",
        "req_two",
    ]


def age(path: Path, seconds: float):
    stamp = time.time() - seconds
    os.utime(path, (stamp, stamp))


def test_pruning_removes_ledger_files_older_than_30_days_once_per_process(tmp_path):
    old = tmp_path / "old-session.jsonl"
    fresh = tmp_path / "fresh-session.jsonl"
    other = tmp_path / "notes.txt"
    for path in (old, fresh, other):
        path.write_text("{}\n")
    age(old, RETENTION_SECONDS + 3600)
    age(fresh, RETENTION_SECONDS - 3600)
    age(other, RETENTION_SECONDS + 3600)
    AccountLedger(tmp_path).prune_once()
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "fresh-session.jsonl",
        "notes.txt",
    ]
    # A second ledger in the same process does not prune again.
    later = tmp_path / "later-session.jsonl"
    later.write_text("{}\n")
    age(later, RETENTION_SECONDS + 3600)
    AccountLedger(tmp_path).prune_once()
    assert later.exists()


def test_the_proxy_prunes_at_start(upstream, tmp_path):
    old = tmp_path / "old-session.jsonl"
    old.write_text("{}\n")
    age(old, RETENTION_SECONDS + 3600)
    with InferenceProxy(pool(), upstream=upstream, ledger=AccountLedger(tmp_path)):
        pass
    assert not old.exists()


def test_pruning_a_missing_or_unreadable_directory_never_raises(tmp_path):
    AccountLedger(tmp_path / "missing").prune_once()
    blocker = tmp_path / "blocker"
    blocker.write_text("")
    AccountLedger(blocker / "claude").prune_once()


def test_default_directory_follows_xdg_state_home(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    assert default_ledger_dir() == tmp_path / "state" / "agent-account-ledger" / "claude"


def test_default_directory_falls_back_to_local_state(monkeypatch):
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    assert (
        default_ledger_dir()
        == Path.home() / ".local" / "state" / "agent-account-ledger" / "claude"
    )


def test_a_relative_xdg_state_home_is_ignored(monkeypatch):
    # The XDG spec: relative paths are invalid and must be ignored.
    monkeypatch.setenv("XDG_STATE_HOME", "relative/state")
    assert (
        default_ledger_dir()
        == Path.home() / ".local" / "state" / "agent-account-ledger" / "claude"
    )


def test_the_proxy_writes_to_the_default_directory_unless_told_otherwise(
    upstream, monkeypatch, tmp_path
):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    with InferenceProxy(pool(), upstream=upstream) as proxy:
        send(proxy)
    assert (tmp_path / "agent-account-ledger" / "claude" / f"{SESSION}.jsonl").exists()


# -- the account alias comes from the roster row that issued the credential --


def test_central_credential_names_its_own_rosters_alias():
    from claude_swap.vision_proxy import CentralCredential

    single = CentralCredential(
        Mock(), {"visionLoginId": LOGIN_A, "alias": "luke-1", "email": "x"}
    )
    assert single.account_alias({"login_id": LOGIN_A}) == "luke-1"
    assert single.account_alias({"login_id": LOGIN_B}) is None


def test_pool_names_the_alias_of_the_row_it_selected(temp_home, monkeypatch):
    from tests.test_vision_pool import NOW, item, observation

    from claude_swap.switcher import ClaudeAccountSwitcher
    from claude_swap.vision import VisionClient
    from claude_swap.vision_pool import CentralPoolCredential

    monkeypatch.delenv("VISION_API_KEY", raising=False)
    switcher = ClaudeAccountSwitcher()
    switcher.backup_dir.mkdir(parents=True, exist_ok=True)
    client = VisionClient("https://vision.example.invalid", "synthetic-key")
    client.discover = Mock(return_value=[item(1), item(2)])
    client.credential = Mock(
        side_effect=lambda account, login: {
            "account_id": account,
            "login_id": login,
            "email": item(int(login[-12:]))["email"],
            "organization_id": "organization",
            "generation": 1,
            "kind": "login_oauth",
            "accessToken": "synthetic",
        }
    )
    observations = {
        item(1)["login_id"]: observation(1, 95),
        item(2)["login_id"]: observation(2, 20),
    }
    monkeypatch.setattr(
        "claude_swap.vision_pool.read_usage", lambda *_a, **_k: observations
    )
    credentials = CentralPoolCredential(
        switcher, client, {"visionLoginId": item(1)["login_id"]}, clock=lambda: NOW
    )
    issued = credentials.get()
    assert issued["login_id"] == item(2)["login_id"]
    roster = switcher._get_sequence_data()["accounts"]
    expected = next(
        row["alias"]
        for row in roster.values()
        if row.get("visionLoginId") == item(2)["login_id"]
    )
    assert expected
    assert credentials.account_alias(issued) == expected
    assert credentials.account_alias({"login_id": "ail_unknown"}) is None
