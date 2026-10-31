"""The adapter reports its usage to Vision without touching a request's path."""

import json
import time
import urllib.error
from unittest.mock import Mock

import pytest

from claude_swap.account_ledger import AccountLedger
from claude_swap.usage_totals import UsageTotals
from claude_swap.vision import VisionClient
from claude_swap.vision_proxy import InferenceProxy
from tests.test_account_ledger import ACCOUNT_A, ACCOUNT_B, ALIASES, LOGIN_A
from tests.test_usage_ledger import post, provider
from tests.test_usage_report import KEY, SyntheticVision

__all__ = ["provider"]  # the fixture, imported for use here

KEY_ID = "vkey_00000000-0000-4000-8000-000000000001"


class VisionWithSelf(SyntheticVision):
    """The report intake plus api-keys/self, which names the key."""

    def __init__(self):
        super().__init__()
        self.self_answer = {
            "via": "api_key",
            "key": {"id": KEY_ID},
            "owner": {"email": "owner-1@infinity.inc"},
        }
        handler = self.server.RequestHandlerClass
        original = handler.do_GET
        vision = self

        def do_GET(request):
            if request.path == "/api/api-keys/self":
                if vision.self_answer is None:
                    return request._send(503, {"error": {"code": "unavailable"}})
                return request._send(200, {"data": vision.self_answer})
            return original(request)

        handler.do_GET = do_GET


@pytest.fixture
def vision():
    instance = VisionWithSelf()
    try:
        yield instance
    finally:
        instance.hold.set()
        instance.server.shutdown()
        instance.server.server_close()
        instance.thread.join()


def proxy_for(provider_url, vision, tmp_path, *, first=ACCOUNT_A):
    pool = Mock()
    pool.get.return_value = first
    pool.account_alias.side_effect = lambda c: ALIASES.get(c.get("login_id"))
    pool.client = VisionClient(vision.url, KEY)
    return InferenceProxy(
        pool,
        upstream=provider_url,
        ledger=AccountLedger(tmp_path / "ledger"),
        usage_totals=UsageTotals(tmp_path / "totals.json", tmp_path / "machine_id"),
    )


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_usage_is_reported_when_the_adapter_exits(provider, vision, tmp_path, capsys):
    url, _ = provider
    proxy = proxy_for(url, vision, tmp_path)
    with proxy:
        assert wait_for(lambda: proxy.key_owner.current() is not None)
        for _ in range(3):
            post(proxy, b'{"model":"claude-opus-4-6"}')
    rows = vision.sent_rows()
    assert rows[-1]["login_id"] == LOGIN_A
    assert rows[-1]["account_id"] == ACCOUNT_A["account_id"]
    assert (rows[-1]["requests"], rows[-1]["output_tokens"]) == (3, 171)
    stored = json.loads((tmp_path / "totals.json").read_text())["rows"]
    assert stored[0]["sent_version"] == stored[0]["version"]
    assert capsys.readouterr().err == ""


def test_a_burst_of_requests_reports_before_the_interval(provider, vision, tmp_path):
    url, _ = provider
    proxy = proxy_for(url, vision, tmp_path)
    proxy.reporter.burst = 3
    proxy.reporter.interval = 3600
    with proxy:
        assert wait_for(lambda: proxy.key_owner.current() is not None)
        # The first pass (at start) may have found nothing to send.
        assert wait_for(lambda: proxy.reporter._last_report is not None)
        proxy.reporter._last_report -= 60  # past the minimum gap
        for _ in range(3):
            post(proxy, b'{"model":"claude-opus-4-6"}')
        assert wait_for(lambda: sum(r["requests"] for r in vision.sent_rows()) >= 3)


def test_requests_finished_before_the_key_is_known_are_reported_later(
    provider, vision, tmp_path
):
    url, _ = provider
    vision.self_answer = None  # Vision cannot name the key yet
    proxy = proxy_for(url, vision, tmp_path)
    with proxy:
        post(proxy, b'{"model":"claude-opus-4-6"}')
        assert not (tmp_path / "totals.json").exists()
        vision.self_answer = {"via": "api_key", "key": {"id": KEY_ID}, "owner": {}}
        proxy.key_owner._retry_at = 0  # skip the lookup's one-minute retry wait
        assert wait_for(lambda: proxy.key_owner.current() is not None)
    [row] = vision.sent_rows()
    assert row["requests"] == 1


def test_a_refused_key_is_reported_to_the_person_once_after_claude_exits(
    provider, vision, tmp_path, capsys
):
    url, _ = provider
    vision.answers.append((403, {"error": {"code": "capability_required"}}, {}))
    proxy = proxy_for(url, vision, tmp_path)
    with proxy:
        assert wait_for(lambda: proxy.key_owner.current() is not None)
        post(proxy, b'{"model":"claude-opus-4-6"}')
        proxy.reporter.report_now()
        assert capsys.readouterr().err == ""  # never while Claude draws
    err = capsys.readouterr().err
    assert err.count("cswap: Vision refused this session's usage reports") == 1
    assert "agent_accounts:use" in err


def test_exit_waits_at_most_about_two_seconds_for_a_hung_vision(
    provider, vision, tmp_path
):
    url, _ = provider
    proxy = proxy_for(url, vision, tmp_path)
    with proxy:
        assert wait_for(lambda: proxy.key_owner.current() is not None)
        post(proxy, b'{"model":"claude-opus-4-6"}')
        vision.hold.clear()
        started = time.monotonic()
    assert time.monotonic() - started < 2.6


def test_reporting_failures_never_affect_a_request(provider, tmp_path):
    url, _ = provider
    pool = Mock()
    pool.get.return_value = ACCOUNT_B
    pool.account_alias.return_value = None
    pool.client = VisionClient("http://127.0.0.1:9", KEY)  # Vision unreachable
    proxy = InferenceProxy(
        pool,
        upstream=url,
        ledger=AccountLedger(tmp_path / "ledger"),
        usage_totals=UsageTotals(tmp_path / "totals.json", tmp_path / "machine_id"),
    )
    with proxy:
        assert post(proxy, b'{"model":"claude-opus-4-6"}')[0] == 200


def test_the_local_limit_gate_is_asked_before_any_credential(provider, vision, tmp_path):
    url, _ = provider
    proxy = proxy_for(url, vision, tmp_path)
    proxy.limit_gate.refusal = lambda: "cswap: synthetic limit reached."
    with proxy:
        with pytest.raises(urllib.error.HTTPError) as refused:
            post(proxy, b'{"model":"claude-opus-4-6"}')
    assert refused.value.code == 400
    assert "synthetic limit reached" in json.loads(refused.value.read())["error"]["message"]
    proxy.credentials.get.assert_not_called()


def test_without_a_vision_client_nothing_is_reported(provider, tmp_path):
    url, _ = provider
    pool = Mock()
    pool.get.return_value = ACCOUNT_A
    pool.account_alias.return_value = None
    proxy = InferenceProxy(
        pool,
        upstream=url,
        ledger=AccountLedger(tmp_path / "ledger"),
        usage_totals=UsageTotals(tmp_path / "totals.json", tmp_path / "machine_id"),
    )
    assert proxy.reporter is None
    with proxy:
        post(proxy, b'{"model":"claude-opus-4-6"}')
    [row] = json.loads((tmp_path / "totals.json").read_text())["rows"]
    assert row["api_key_id"] is None
