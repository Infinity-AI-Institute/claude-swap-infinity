"""The adapter reuses an issued Vision credential instead of fetching one per request.

Incident, 2026-10-26 to 11-02: one Vision key fetched a credential for one
Claude account about 1.24 million times in a week, about 8 fetches for every
labeling job a fleet of headless agents ran. Nothing was retrying: the
adapter asked Vision for a fresh access token before every native request,
count_tokens and native's own retries included, although the token it had
fetched a second earlier was still valid for hours. Every fetch is a row in
Vision's issuance log and a decrypt on Vision's side.
"""

import http.server
import json
import threading
import urllib.error
import urllib.request
from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from claude_swap.account_ledger import AccountLedger
from claude_swap.switcher import ClaudeAccountSwitcher
from claude_swap.usage_store import UsageEntry
from claude_swap.usage_totals import UsageTotals
from claude_swap.vision import VisionClient, VisionError
from claude_swap.vision_pool import CentralPoolCredential
from claude_swap.vision_proxy import CentralCredential, InferenceProxy
from claude_swap.vision_session import (
    CREDENTIAL_EXPIRY_MARGIN_SECONDS,
    CREDENTIAL_REUSE_SECONDS,
)
from tests.test_vision_registry import item

NOW = 1_800_000_000.0
TOKEN_LIFETIME_SECONDS = 8 * 3600


def iso(timestamp):
    return datetime.fromtimestamp(timestamp, UTC).isoformat()


def healthy():
    return {
        "five_hour": {"pct": 10.0, "resets_at": iso(NOW + 3600)},
        "seven_day": {"pct": 10.0, "resets_at": iso(NOW + 3 * 86400)},
    }


class FakeVision:
    """Vision's registry as the adapter sees it, counting every credential issued."""

    def __init__(self, clock):
        self.clock = clock
        self.generations = {1: 1, 2: 1}
        self.expires_in = {1: TOKEN_LIFETIME_SECONDS, 2: TOKEN_LIFETIME_SECONDS}
        self.client = VisionClient("https://vision.example.invalid", "synthetic-key")
        # Anything else the adapter asks Vision (the key's owner, usage
        # reports) fails the way an unreachable Vision does.
        self.client.request = Mock(side_effect=VisionError("service_unavailable"))
        self.client.discover = Mock(side_effect=self._discover)
        self.client.credential = Mock(side_effect=self._credential)
        self.client.refresh = Mock(return_value={"state": "current", "generation": 1})

    def _discover(self):
        return [
            {**item(index), "login_generation": generation}
            for index, generation in self.generations.items()
        ]

    def _credential(self, account, login):
        index = int(login[-12:])
        row = item(index)
        generation = self.generations[index]
        return {
            "account_id": account,
            "login_id": login,
            "email": row["email"],
            "organization_id": row["organization_id"],
            "generation": generation,
            "expires_at": iso(self.clock() + self.expires_in[index]),
            "kind": "login_oauth",
            "accessToken": f"synthetic-token-{index}-g{generation}",
        }

    def issued(self, index=None):
        """Credentials issued so far, for one login or for all."""
        if index is None:
            return self.client.credential.call_count
        login = item(index)["login_id"]
        return sum(
            1 for call in self.client.credential.call_args_list if call.args[1] == login
        )


@pytest.fixture
def clock():
    return Mock(return_value=NOW)


@pytest.fixture
def vision(temp_home, monkeypatch, clock):
    monkeypatch.delenv("VISION_API_KEY", raising=False)
    fake = FakeVision(clock)
    usage = {
        item(index)["login_id"]: (
            item(index)["account_id"],
            UsageEntry(last_good=healthy(), age_s=10),
        )
        for index in (1, 2)
    }
    monkeypatch.setattr(
        "claude_swap.vision_pool.read_usage", lambda *_args, **_kwargs: usage
    )
    return fake


@pytest.fixture
def pool(vision, clock):
    switcher = ClaudeAccountSwitcher()
    switcher.backup_dir.mkdir(parents=True, exist_ok=True)
    return CentralPoolCredential(
        switcher, vision.client, {"visionLoginId": item(1)["login_id"]}, clock=clock
    )


@pytest.fixture
def upstream():
    """Anthropic's message endpoint: answers 200, or as a test tells it to."""
    seen = []
    answers = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            token = self.headers["Authorization"].removeprefix("Bearer ")
            seen.append(token)
            status, headers = answers.get(token, (200, {}))
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            body = b"data: synthetic\n\n" if status == 200 else b"{}"
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", seen, answers
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def adapter(credentials, upstream_url, tmp_path):
    return InferenceProxy(
        credentials,
        upstream=upstream_url,
        ledger=AccountLedger(tmp_path / "ledger"),
        usage_totals=UsageTotals(tmp_path / "totals.json", tmp_path / "machine-id"),
    )


def send(proxy, path="/v1/messages?beta=true"):
    request = urllib.request.Request(
        proxy.url + path,
        data=json.dumps({"model": "claude-haiku-4-5"}).encode(),
        headers={
            "Authorization": "Bearer " + proxy.capability,
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


# -- the adapter, as native drives it -------------------------------------------


def test_a_session_of_native_requests_fetches_one_credential(
    vision, pool, upstream, tmp_path
):
    url, seen, _ = upstream
    with adapter(pool, url, tmp_path) as proxy:
        for _ in range(20):
            assert send(proxy) == 200
            assert send(proxy, "/v1/messages/count_tokens") == 200
    assert len(seen) == 40
    assert set(seen) == {"synthetic-token-1-g1"}
    assert vision.issued() == 1


def test_a_single_login_session_fetches_one_credential(vision, upstream, tmp_path):
    url, seen, _ = upstream
    record = {
        "visionAccountId": item(1)["account_id"],
        "visionLoginId": item(1)["login_id"],
        "visionGeneration": 1,
        "email": item(1)["email"],
        "organizationUuid": item(1)["organization_id"],
    }
    credentials = CentralCredential(vision.client, record, clock=vision.clock)
    with adapter(credentials, url, tmp_path) as proxy:
        for _ in range(10):
            assert send(proxy) == 200
    assert len(seen) == 10
    assert vision.issued() == 1


def test_native_retrying_a_throttle_fetches_once_per_login_not_per_retry(
    vision, pool, upstream, tmp_path
):
    url, seen, answers = upstream
    answers["synthetic-token-1-g1"] = (429, {"Retry-After": "120"})
    with adapter(pool, url, tmp_path) as proxy:
        # The first request is throttled on login 1 and replayed on login 2;
        # the rest go straight to login 2 while login 1 waits out Retry-After.
        for _ in range(10):
            assert send(proxy) == 200
    assert seen.count("synthetic-token-1-g1") == 1
    assert vision.issued(1) == 1
    assert vision.issued(2) == 1


def test_a_rejected_token_is_never_reused_and_its_successor_is(
    vision, pool, upstream, tmp_path
):
    url, seen, answers = upstream
    with adapter(pool, url, tmp_path) as proxy:
        assert send(proxy) == 200
        # The provider stops accepting generation 1; Vision has rotated it.
        answers["synthetic-token-1-g1"] = (401, {})
        vision.generations[1] = 2
        for _ in range(5):
            assert send(proxy) == 200
    assert seen == ["synthetic-token-1-g1", "synthetic-token-1-g1"] + [
        "synthetic-token-1-g2"
    ] * 5
    # The first fetch, then one for recovery; the successor is reused.
    assert vision.issued(1) == 2


# -- how long an issued credential is reused --------------------------------------


def test_reuse_ends_after_the_reuse_window_so_vision_rechecks_access(
    vision, pool, clock
):
    pool.get()
    clock.return_value = NOW + CREDENTIAL_REUSE_SECONDS - 1
    pool.get()
    assert vision.issued() == 1
    clock.return_value = NOW + CREDENTIAL_REUSE_SECONDS
    pool.get()
    assert vision.issued() == 2


def test_a_credential_near_its_expiry_is_not_reused(vision, pool, clock):
    vision.expires_in[1] = CREDENTIAL_EXPIRY_MARGIN_SECONDS + 10
    pool.get()
    pool.get()
    assert vision.issued() == 1
    clock.return_value = NOW + 10
    pool.get()
    assert vision.issued() == 2


def test_a_newer_generation_in_the_roster_replaces_the_reused_credential(
    vision, pool
):
    assert pool.get()["accessToken"] == "synthetic-token-1-g1"
    vision.generations[1] = 2
    pool.pool.sync(force=True)
    assert pool.get()["accessToken"] == "synthetic-token-1-g2"
    assert pool.get()["accessToken"] == "synthetic-token-1-g2"
    assert vision.issued() == 2


def test_a_revoked_key_still_stops_requests_at_the_next_discovery(vision, pool, clock):
    pool.get()
    vision.client.discover.side_effect = VisionError("unauthorized", status=401)
    clock.return_value = NOW + 31  # past the 30-second discovery cache
    with pytest.raises(VisionError) as refused:
        pool.get()
    assert refused.value.code == "unauthorized"
    assert vision.issued() == 1


def test_a_login_vision_no_longer_grants_is_not_served_from_reuse(vision, pool, clock):
    assert pool.get()["accessToken"] == "synthetic-token-1-g1"
    del vision.generations[1]
    pool.pool.sync(force=True)
    assert pool.get()["accessToken"] == "synthetic-token-2-g1"
