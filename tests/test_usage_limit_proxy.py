"""The adapter refuses model requests while the key is over a Vision usage limit.

Vision (PR #421) states the key's limits in ``GET .../usage/limits/self`` and
every report answer, and refuses a credential fetch at a limit with 429
``usage_limit_reached``. The adapter answers native with the "you must act"
shape (400, ``x-should-retry: false``), which native shows verbatim and never
retries, and keeps reporting usage meanwhile.
"""

import http.server
import json
import threading
import urllib.error
from datetime import UTC, datetime, timedelta

import pytest

from claude_swap.usage_limits import pacific_time
from claude_swap.vision import VisionClient, VisionError
from tests.test_account_ledger import LOGIN_A
from tests.test_usage_ledger import post, provider
from tests.test_usage_report_proxy import VisionWithSelf, proxy_for, wait_for

__all__ = ["provider"]  # the fixture, imported for use here

ACCOUNT = "aia_00000000-0000-4000-8000-000000000001"


def resets_in(hours):
    return (datetime.now(UTC) + timedelta(hours=hours)).replace(microsecond=0)


def window(name="day", *, state, used, limit, resets_at):
    return {
        "window": name,
        "resets_at": resets_at.isoformat().replace("+00:00", "Z"),
        "limit": {"unit": "usd_estimated", "value": limit},
        "used": {"value": used, "as_of": datetime.now(UTC).isoformat()},
        "state": state,
    }


def refusal_body(resets_at):
    """Vision's 429 at a limit, exactly as PR #421's credential route builds it."""
    stamp = resets_at.isoformat().replace("+00:00", "Z")
    return {
        "error": {
            "message": (
                "This API key reached its week usage limit (31.40 of 30.00 USD at "
                "API list price, not what Infinity pays). Credentials are refused "
                f"until {stamp}, when the window resets; ask its owner or an admin "
                "to raise the limit."
            ),
            "code": "usage_limit_reached",
            "window": "week",
            "resets_at": stamp,
            "limit": {"unit": "usd_estimated", "value": 30},
            "used": {"value": 31.4, "as_of": datetime.now(UTC).isoformat()},
        }
    }


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


@pytest.fixture
def refusing_vision():
    """A credential route that answers every fetch with the limit's 429."""
    resets_at = resets_in(30)
    calls = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            calls.append(self.path)
            body = json.dumps(refusal_body(resets_at)).encode()
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", resets_at, calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def refused(proxy):
    with pytest.raises(urllib.error.HTTPError) as answer:
        post(proxy, b'{"model":"claude-opus-4-6"}')
    return answer.value


# -- the transport ------------------------------------------------------------


def test_visions_limit_refusal_keeps_its_code_fields_and_message(refusing_vision):
    url, resets_at, calls = refusing_vision
    client = VisionClient(url, "vsk_" + "1" * 40)
    with pytest.raises(VisionError) as raised:
        client.credential(ACCOUNT, LOGIN_A)
    error = raised.value
    assert (error.code, error.status) == ("usage_limit_reached", 429)
    assert error.details["window"] == "week"
    assert error.details["resets_at"] == refusal_body(resets_at)["error"]["resets_at"]
    # People read Vision's own words, plus the reset in Pacific time.
    assert str(error) == (
        refusal_body(resets_at)["error"]["message"]
        + f" (Resets {pacific_time(resets_at)}.)"
    )
    assert len(calls) == 1  # no retry


# -- the adapter --------------------------------------------------------------


def test_a_credential_refused_at_a_limit_is_shown_as_is_and_refused_locally_after(
    provider, vision, tmp_path
):
    url, _ = provider
    proxy = proxy_for(url, vision, tmp_path)
    resets_at = resets_in(30)
    body = refusal_body(resets_at)["error"]
    proxy.credentials.get.side_effect = VisionError(
        "usage_limit_reached", 429, details=body
    )
    with proxy:
        first = refused(proxy)
        assert first.code == 400
        assert first.headers["x-should-retry"] == "false"
        assert json.loads(first.read())["error"]["message"] == (
            f"cswap: {body['message']} (Resets {pacific_time(resets_at)}.)"
        )
        second = refused(proxy)
        assert second.code == 400
        assert json.loads(second.read())["error"]["message"] == (
            "cswap: this Vision API key reached its weekly usage limit (31.40 of 30.00 "
            "USD at API list price, not what Infinity pays). cswap refuses new model "
            f"requests until {pacific_time(resets_at)}, when the week resets; ask the "
            "key's owner or a Vision admin to raise the limit."
        )
    # Vision is not asked again before the reset.
    assert proxy.credentials.get.call_count == 1


def test_an_exceeded_limit_read_refuses_before_any_credential_and_reporting_goes_on(
    provider, vision, tmp_path
):
    url, _ = provider
    proxy = proxy_for(url, vision, tmp_path)
    with proxy:
        assert wait_for(lambda: proxy.reporter.limits_read_once.is_set())
        assert wait_for(lambda: proxy.key_owner.current() is not None)
        assert post(proxy, b'{"model":"claude-opus-4-6"}')[0] == 200
        vision.limits = [
            window(state="exceeded", used=52.5, limit=50, resets_at=resets_in(3))
        ]
        proxy.reporter._limits_due_at = 0  # the next pass reads the limits
        proxy.reporter._wake.set()
        assert wait_for(lambda: proxy.limit_gate.refusal() is not None)
        answer = refused(proxy)
        assert answer.code == 400
        message = json.loads(answer.read())["error"]["message"]
        assert "daily usage limit (52.50 of 50.00 USD at API list price, not what " in message
    assert proxy.credentials.get.call_count == 1  # only the request served
    # The served request's usage was still reported.
    assert sum(row["requests"] for row in vision.sent_rows()) == 1


def test_a_limit_raised_by_an_admin_lifts_the_refusal_at_the_next_read(
    provider, vision, tmp_path
):
    url, _ = provider
    vision.limits = [window(state="exceeded", used=52.5, limit=50, resets_at=resets_in(3))]
    proxy = proxy_for(url, vision, tmp_path)
    with proxy:
        assert wait_for(lambda: proxy.limit_gate.refusal() is not None)
        assert refused(proxy).code == 400
        vision.limits = [window(state="ok", used=52.5, limit=100, resets_at=resets_in(3))]
        proxy.reporter._limits_due_at = 0
        proxy.reporter._wake.set()
        assert wait_for(lambda: proxy.limit_gate.refusal() is None)
        assert post(proxy, b'{"model":"claude-opus-4-6"}')[0] == 200


def test_no_limit_never_refuses(provider, vision, tmp_path):
    url, _ = provider
    proxy = proxy_for(url, vision, tmp_path)
    with proxy:
        assert wait_for(lambda: proxy.reporter.limits_read_once.is_set())
        assert post(proxy, b'{"model":"claude-opus-4-6"}')[0] == 200


def test_a_warning_known_at_launch_is_printed_before_claude_starts_and_only_then(
    provider, vision, tmp_path, capsys
):
    url, _ = provider
    resets_at = resets_in(5)
    vision.limits = [window(state="warning", used=41, limit=50, resets_at=resets_at)]
    proxy = proxy_for(url, vision, tmp_path)
    with proxy:
        assert capsys.readouterr().err == (
            "cswap: this Vision API key has used 82% of its daily usage limit (41.00 "
            "of 50.00 USD at API list price, not what Infinity pays). At the limit, "
            "cswap refuses new model requests until the day resets "
            f"({pacific_time(resets_at)}).\n"
        )
        assert post(proxy, b'{"model":"claude-opus-4-6"}')[0] == 200
    assert capsys.readouterr().err == ""


def test_a_warning_learned_while_claude_runs_is_printed_after_it_exits(
    provider, vision, tmp_path, capsys
):
    url, _ = provider
    proxy = proxy_for(url, vision, tmp_path)
    with proxy:
        assert capsys.readouterr().err == ""
        vision.limits = [window(state="warning", used=45, limit=50, resets_at=resets_in(5))]
        proxy.reporter._limits_due_at = 0
        proxy.reporter._wake.set()
        assert wait_for(lambda: proxy.limit_gate.windows is not None)
        assert capsys.readouterr().err == ""  # never over Claude's screen
    err = capsys.readouterr().err
    assert err.count("cswap: this Vision API key has used 90% of its daily usage limit") == 1


def test_a_vision_outage_on_a_credential_stays_a_retryable_503(provider, vision, tmp_path):
    url, _ = provider
    proxy = proxy_for(url, vision, tmp_path)
    proxy.credentials.get.side_effect = VisionError("service_unavailable", 503)
    with proxy:
        answer = refused(proxy)
    assert answer.code == 503
    assert answer.headers.get("x-should-retry") is None
    assert proxy.limit_gate.refusal() is None
