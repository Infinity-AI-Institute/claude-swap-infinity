"""The ledger names the Vision person whose API key the adapter is using.

A synthetic Vision answers ``GET /api/api-keys/self`` the way Vision does
(``{"data": {"via": "api_key", "key": {...}, "owner": {"email": ...}}}``), so
the real VisionClient, including its re-read of a rejected key, is exercised.
"""

import http.server
import json
import logging
import threading
import time
import urllib.request
from unittest.mock import Mock

import pytest

from claude_swap.account_ledger import AccountLedger
from claude_swap.vision import VisionClient
from claude_swap.vision_key_owner import KeyOwnerLookup
from claude_swap.vision_proxy import InferenceProxy
from tests.test_account_ledger import ACCOUNT_A, SESSION, lines, pool, send, upstream

__all__ = ["upstream"]  # the fixture, imported for use here

KEY_ONE = "vsk_" + "1" * 40
KEY_TWO = "vsk_" + "2" * 40
OWNERS = {KEY_ONE: "owner-1@infinity.inc", KEY_TWO: "owner-2@infinity.inc"}


class SyntheticVision:
    def __init__(self):
        self.revoked: set[str] = set()
        self.status_override: int | None = None
        self.hold = threading.Event()
        self.hold.set()
        self.self_calls = 0
        vision = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                key = self.headers.get("X-Api-Key")
                if self.path == "/api/api-keys/self":
                    vision.self_calls += 1
                    vision.hold.wait(10)
                if key not in OWNERS or key in vision.revoked:
                    return self._send(401, {"error": {"code": "unauthorized"}})
                if vision.status_override is not None:
                    return self._send(
                        vision.status_override, {"error": {"code": "internal"}}
                    )
                if self.path != "/api/api-keys/self":
                    return self._send(200, {"data": {}})
                self._send(
                    200,
                    {
                        "data": {
                            "via": "api_key",
                            "key": {"id": "key_synthetic", "name": "synthetic"},
                            "owner": {
                                "id": "usr_synthetic",
                                "email": OWNERS[key],
                                "role": "member",
                                "rights": [],
                            },
                        }
                    },
                )

            def _send(self, status, payload):
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()


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


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def pool_with(client):
    credentials = pool()
    credentials.client = client
    return credentials


def test_each_line_names_the_owner_of_the_key_in_use(upstream, vision, tmp_path):
    client = VisionClient(vision.url, KEY_ONE)
    with InferenceProxy(
        pool_with(client), upstream=upstream, ledger=AccountLedger(tmp_path)
    ) as proxy:
        assert wait_for(lambda: proxy.key_owner.current() is not None)
        send(proxy)
        send(proxy)
    rows = lines(tmp_path / f"{SESSION}.jsonl")
    assert [row["key_owner_email"] for row in rows] == ["owner-1@infinity.inc"] * 2
    assert list(rows[0])[-1] == "key_owner_email"
    assert rows[0]["account_email"] == ACCOUNT_A["email"]
    # Resolved once per process, not per request.
    assert vision.self_calls == 1


def test_the_lookup_never_delays_a_request(upstream, vision, tmp_path):
    vision.hold.clear()  # Vision does not answer the lookup yet.
    client = VisionClient(vision.url, KEY_ONE)
    with InferenceProxy(
        pool_with(client), upstream=upstream, ledger=AccountLedger(tmp_path)
    ) as proxy:
        started = time.monotonic()
        assert send(proxy)[0] == 200
        assert time.monotonic() - started < 2
        vision.hold.set()
        assert wait_for(lambda: proxy.key_owner.current() is not None)
        send(proxy)
    assert [
        row["key_owner_email"] for row in lines(tmp_path / f"{SESSION}.jsonl")
    ] == [None, "owner-1@infinity.inc"]


def test_a_failed_lookup_records_null_and_never_logs_an_address(
    upstream, vision, tmp_path, caplog
):
    vision.status_override = 500
    client = VisionClient(vision.url, KEY_ONE)
    with caplog.at_level(logging.DEBUG, logger="claude-swap"):
        with InferenceProxy(
            pool_with(client), upstream=upstream, ledger=AccountLedger(tmp_path)
        ) as proxy:
            assert wait_for(lambda: vision.self_calls >= 1)
            assert wait_for(lambda: not proxy.key_owner.resolving())
            assert send(proxy)[0] == 200
    [row] = lines(tmp_path / f"{SESSION}.jsonl")
    assert row["key_owner_email"] is None
    assert all("@" not in record.getMessage() for record in caplog.records)


def test_a_key_re_read_after_a_rejection_is_looked_up_again(vision):
    replacement = VisionClient(vision.url, KEY_TWO)
    client = VisionClient(vision.url, KEY_ONE, reload=lambda: replacement)
    lookup = KeyOwnerLookup(client)
    lookup.start()
    assert wait_for(lambda: lookup.current() == "owner-1@infinity.inc")
    # The key is revoked and rotated; the next Vision request re-reads it (P3).
    vision.revoked.add(KEY_ONE)
    client.request("GET", "/api/me")
    # Never the previous key's owner for the new key.
    assert lookup.current() in (None, "owner-2@infinity.inc")
    assert wait_for(lambda: lookup.current() == "owner-2@infinity.inc")
    assert vision.self_calls == 2


def test_a_failed_lookup_is_retried_later_not_on_every_request(vision):
    vision.status_override = 503
    clock = Mock(return_value=1000.0)
    lookup = KeyOwnerLookup(VisionClient(vision.url, KEY_ONE), clock=clock)
    lookup.start()
    assert wait_for(lambda: vision.self_calls == 1 and not lookup.resolving())
    for _ in range(5):
        assert lookup.current() is None
    assert not lookup.resolving()
    assert vision.self_calls == 1
    vision.status_override = None
    clock.return_value += 61
    lookup.current()
    assert wait_for(lambda: lookup.current() == "owner-1@infinity.inc")
    assert vision.self_calls == 2


@pytest.mark.parametrize(
    "payload",
    [
        {"via": "jwt", "owner": {"email": "owner-1@infinity.inc"}},
        {"via": "api_key", "owner": {"email": 7}},
        {"via": "api_key", "owner": {"email": "no-at-sign"}},
        {"via": "api_key", "owner": {"email": "a@b\ninjected"}},
        {"via": "api_key", "owner": None},
        [],
    ],
)
def test_an_unexpected_self_answer_is_unknown(payload):
    client = Mock(spec=VisionClient)
    client.key_generation = 0
    client.request.return_value = payload
    lookup = KeyOwnerLookup(client)
    lookup.start()
    assert wait_for(lambda: not lookup.resolving())
    assert lookup.current() is None


def test_the_owner_is_recorded_exactly_as_vision_returns_it():
    # A person's address, unlike an account's, is not checked against the
    # @infinity.inc domain or normalized.
    client = Mock(spec=VisionClient)
    client.key_generation = 0
    client.request.return_value = {
        "via": "api_key",
        "owner": {"email": "Owner-1@Infinity.Inc"},
    }
    lookup = KeyOwnerLookup(client)
    lookup.start()
    assert wait_for(lambda: lookup.current() is not None)
    assert lookup.current() == "Owner-1@Infinity.Inc"


def test_credentials_without_a_vision_client_record_null(upstream, tmp_path):
    with InferenceProxy(
        pool(), upstream=upstream, ledger=AccountLedger(tmp_path)
    ) as proxy:
        send(proxy)
    [row] = lines(tmp_path / f"{SESSION}.jsonl")
    assert row["key_owner_email"] is None


def test_the_client_counts_each_adopted_key(vision):
    replacement = VisionClient(vision.url, KEY_TWO)
    client = VisionClient(vision.url, KEY_ONE, reload=lambda: replacement)
    assert client.key_generation == 0
    client.request("GET", "/api/me")
    assert client.key_generation == 0
    vision.revoked.add(KEY_ONE)
    client.request("GET", "/api/me")
    assert client.key_generation == 1
