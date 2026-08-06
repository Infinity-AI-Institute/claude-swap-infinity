"""Tests for the Codex / Kimi usage fetchers (all HTTP mocked)."""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from claude_swap import tool_usage
from claude_swap.tool_usage import (
    ToolUsageError,
    fetch_codex_usage,
    fetch_kimi_usage,
    fetch_usage,
)


class _FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode()

    def read(self, _limit: int = -1) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _mock_urlopen(monkeypatch, payload=None, error=None):
    """Patch urllib.request.urlopen inside tool_usage; returns captured requests."""
    captured = []

    def fake_urlopen(req, timeout=None):
        captured.append(req)
        if error is not None:
            raise error
        return _FakeResponse(payload)

    monkeypatch.setattr(tool_usage.urllib.request, "urlopen", fake_urlopen)
    return captured


CODEX_CRED = {
    "tokens": {
        "access_token": "at-1",
        "id_token": "x.y.z",
        "account_id": "acct-1",
    }
}

KIMI_CRED = {"access_token": "kt-1"}


class TestCodexUsage:
    PAYLOAD = {
        "plan_type": "plus",
        "rate_limit": {
            "allowed": True,
            "limit_reached": False,
            "primary_window": {
                "used_percent": 37,
                "reset_at": 2000000000,
                "limit_window_seconds": 18000,
            },
            "secondary_window": {
                "used_percent": 12,
                "reset_at": 2000000001,
                "limit_window_seconds": 604800,
            },
        },
    }

    def test_parses_windows_and_plan(self, monkeypatch):
        captured = _mock_urlopen(monkeypatch, payload=self.PAYLOAD)
        usage = fetch_codex_usage(CODEX_CRED)
        assert usage.plan == "plus"
        assert [(w.label, w.pct) for w in usage.windows] == [("5h", 37.0), ("7d", 12.0)]
        assert usage.windows[0].resets_at == 2000000000.0
        req = captured[0]
        assert req.full_url == tool_usage.CODEX_USAGE_URL
        assert req.headers["Authorization"] == "Bearer at-1"
        assert req.headers["Chatgpt-account-id"] == "acct-1"

    def test_missing_access_token(self):
        with pytest.raises(ToolUsageError, match="no-access-token"):
            fetch_codex_usage({"tokens": {}})

    def test_http_error_becomes_short_reason(self, monkeypatch):
        err = urllib.error.HTTPError(
            "u", 401, "unauthorized", {}, io.BytesIO(b"nope")
        )
        _mock_urlopen(monkeypatch, error=err)
        with pytest.raises(ToolUsageError, match="http-401"):
            fetch_codex_usage(CODEX_CRED)

    def test_timeout(self, monkeypatch):
        _mock_urlopen(monkeypatch, error=TimeoutError())
        with pytest.raises(ToolUsageError, match="timeout"):
            fetch_codex_usage(CODEX_CRED)

    def test_network_error(self, monkeypatch):
        _mock_urlopen(monkeypatch, error=urllib.error.URLError("down"))
        with pytest.raises(ToolUsageError, match="network"):
            fetch_codex_usage(CODEX_CRED)

    def test_payload_without_windows(self, monkeypatch):
        _mock_urlopen(monkeypatch, payload={"plan_type": "pro"})
        with pytest.raises(ToolUsageError, match="no-rate-limits"):
            fetch_codex_usage(CODEX_CRED)

    def test_account_id_falls_back_to_id_token_claims(self, monkeypatch):
        import base64

        payload = base64.urlsafe_b64encode(
            json.dumps(
                {"https://api.openai.com/auth": {"chatgpt_account_id": "acct-jwt"}}
            ).encode()
        ).rstrip(b"=").decode()
        cred = {"tokens": {"access_token": "at-1", "id_token": f"h.{payload}."}}
        captured = _mock_urlopen(monkeypatch, payload=self.PAYLOAD)
        fetch_codex_usage(cred)
        assert captured[0].headers["Chatgpt-account-id"] == "acct-jwt"


class TestKimiUsage:
    def test_parses_usage_and_limits(self, monkeypatch):
        payload = {
            "usage": {"used": 40, "limit": 100, "reset_in": 3600},
            "limits": [
                {
                    "window": {"duration": 5, "timeUnit": "HOUR"},
                    "detail": {"used": 10, "limit": 50},
                },
                {"name": "Weekly", "detail": {"remaining": 25, "limit": 100}},
            ],
        }
        captured = _mock_urlopen(monkeypatch, payload=payload)
        usage = fetch_kimi_usage(KIMI_CRED)
        rows = {w.label: w for w in usage.windows}
        assert rows["Weekly limit"].pct == 40.0
        assert rows["Weekly limit"].resets_at is not None
        assert rows["5h limit"].pct == 20.0
        # remaining-derived: used = 100 - 25 -> 75%
        assert rows["Weekly"].pct == 75.0
        req = captured[0]
        assert req.full_url == tool_usage.KIMI_USAGE_URL
        assert req.headers["Authorization"] == "Bearer kt-1"
        assert req.headers["User-agent"] == tool_usage.KIMI_USER_AGENT
        assert req.headers["X-msh-platform"] == "kimi_cli"
        assert req.headers["X-msh-device-id"]

    def test_real_payload_shape(self, monkeypatch):
        """Regression: the live /usages response uses string numbers,
        ``resetTime`` (camelCase), and user.membership.level for the plan."""
        payload = {
            "user": {"userId": "u1", "membership": {"level": "LEVEL_STANDARD"}},
            "usage": {
                "limit": "100",
                "used": "83",
                "remaining": "17",
                "resetTime": "2099-08-07T19:27:44.418564Z",
            },
            "limits": [
                {
                    "window": {"duration": 300, "timeUnit": "TIME_UNIT_MINUTE"},
                    "detail": {
                        "limit": "100",
                        "used": "9",
                        "remaining": "91",
                        "resetTime": "2099-08-05T18:27:44.418564Z",
                    },
                }
            ],
        }
        _mock_urlopen(monkeypatch, payload=payload)
        usage = fetch_kimi_usage(KIMI_CRED)
        assert usage.plan == "standard"
        rows = {w.label: w for w in usage.windows}
        assert rows["Weekly limit"].pct == 83.0
        assert rows["Weekly limit"].resets_at is not None
        assert rows["5h limit"].pct == 9.0

    def test_missing_access_token(self):
        with pytest.raises(ToolUsageError, match="no-access-token"):
            fetch_kimi_usage({})

    def test_http_401(self, monkeypatch):
        err = urllib.error.HTTPError("u", 401, "unauthorized", {}, io.BytesIO(b"x"))
        _mock_urlopen(monkeypatch, error=err)
        with pytest.raises(ToolUsageError, match="http-401"):
            fetch_kimi_usage(KIMI_CRED)

    def test_empty_payload(self, monkeypatch):
        _mock_urlopen(monkeypatch, payload={})
        with pytest.raises(ToolUsageError, match="no-usage-rows"):
            fetch_kimi_usage(KIMI_CRED)

    def test_non_json_response(self, monkeypatch):
        class _Bad:
            def read(self, _limit=-1):
                return b"<html>oops</html>"

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr(
            tool_usage.urllib.request, "urlopen", lambda req, timeout=None: _Bad()
        )
        with pytest.raises(ToolUsageError, match="bad-response"):
            fetch_kimi_usage(KIMI_CRED)


class TestDispatch:
    def test_fetch_usage_by_tool(self, monkeypatch):
        _mock_urlopen(monkeypatch, payload=TestCodexUsage.PAYLOAD)
        usage = fetch_usage("codex", json.dumps(CODEX_CRED))
        assert usage.plan == "plus"

    def test_bad_credential_json(self):
        with pytest.raises(ToolUsageError, match="bad-credential"):
            fetch_usage("codex", "not json")

    def test_unknown_tool(self):
        with pytest.raises(ToolUsageError, match="unknown tool"):
            fetch_usage("nope", "{}")
