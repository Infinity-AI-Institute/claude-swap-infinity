"""Tests for the Codex usage fetcher and its wall-machinery bridge
(codex_usage.py) — all HTTP mocked."""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from claude_swap import codex_usage
from claude_swap.codex_handoff import codex_switch_margin
from claude_swap.codex_usage import (
    CodexUsage,
    CodexUsageError,
    UsageWindow,
    as_usage_dict,
    fetch_codex_usage,
    fetch_usage_from_text,
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
    """Patch urllib.request.urlopen inside codex_usage; returns captured
    requests."""
    captured = []

    def fake_urlopen(req, timeout=None):
        captured.append(req)
        if error is not None:
            raise error
        return _FakeResponse(payload)

    monkeypatch.setattr(codex_usage.urllib.request, "urlopen", fake_urlopen)
    return captured


CODEX_CRED = {
    "tokens": {
        "access_token": "at-1",
        "id_token": "x.y.z",
        "account_id": "acct-1",
    }
}

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


class TestFetchCodexUsage:
    def test_parses_windows_and_plan(self, monkeypatch):
        captured = _mock_urlopen(monkeypatch, payload=PAYLOAD)
        usage = fetch_codex_usage(CODEX_CRED)
        assert usage.plan == "plus"
        assert [(w.label, w.pct) for w in usage.windows] == [("5h", 37.0), ("7d", 12.0)]
        assert usage.windows[0].resets_at == 2000000000.0
        req = captured[0]
        assert req.full_url == codex_usage.CODEX_USAGE_URL
        assert req.headers["Authorization"] == "Bearer at-1"
        assert req.headers["Chatgpt-account-id"] == "acct-1"

    def test_missing_access_token(self):
        with pytest.raises(CodexUsageError, match="no-access-token"):
            fetch_codex_usage({"tokens": {}})

    def test_http_error_becomes_short_reason(self, monkeypatch):
        err = urllib.error.HTTPError("u", 401, "unauthorized", {}, io.BytesIO(b"nope"))
        _mock_urlopen(monkeypatch, error=err)
        with pytest.raises(CodexUsageError, match="http-401"):
            fetch_codex_usage(CODEX_CRED)

    def test_timeout(self, monkeypatch):
        _mock_urlopen(monkeypatch, error=TimeoutError())
        with pytest.raises(CodexUsageError, match="timeout"):
            fetch_codex_usage(CODEX_CRED)

    def test_network_error(self, monkeypatch):
        _mock_urlopen(monkeypatch, error=urllib.error.URLError("down"))
        with pytest.raises(CodexUsageError, match="network"):
            fetch_codex_usage(CODEX_CRED)

    def test_payload_without_windows(self, monkeypatch):
        _mock_urlopen(monkeypatch, payload={"plan_type": "pro"})
        with pytest.raises(CodexUsageError, match="no-rate-limits"):
            fetch_codex_usage(CODEX_CRED)

    def test_non_json_response(self, monkeypatch):
        class _Bad:
            def read(self, _limit=-1):
                return b"<html>oops</html>"

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr(
            codex_usage.urllib.request, "urlopen", lambda req, timeout=None: _Bad()
        )
        with pytest.raises(CodexUsageError, match="bad-response"):
            fetch_codex_usage(CODEX_CRED)

    def test_account_id_falls_back_to_id_token_claims(self, monkeypatch):
        import base64

        payload = (
            base64.urlsafe_b64encode(
                json.dumps(
                    {"https://api.openai.com/auth": {"chatgpt_account_id": "acct-jwt"}}
                ).encode()
            )
            .rstrip(b"=")
            .decode()
        )
        cred = {"tokens": {"access_token": "at-1", "id_token": f"h.{payload}."}}
        captured = _mock_urlopen(monkeypatch, payload=PAYLOAD)
        fetch_codex_usage(cred)
        assert captured[0].headers["Chatgpt-account-id"] == "acct-jwt"

    def test_fetch_from_text(self, monkeypatch):
        _mock_urlopen(monkeypatch, payload=PAYLOAD)
        usage = fetch_usage_from_text(json.dumps(CODEX_CRED))
        assert usage.plan == "plus"

    def test_fetch_from_bad_text(self):
        with pytest.raises(CodexUsageError, match="bad-credential"):
            fetch_usage_from_text("not json")
        with pytest.raises(CodexUsageError, match="bad-credential"):
            fetch_usage_from_text("[1, 2]")


class TestAsUsageDict:
    def test_maps_5h_and_7d_to_oauth_shape(self):
        usage = CodexUsage(
            plan="plus",
            windows=(
                UsageWindow("5h", 37.0, 2000000000.0),
                UsageWindow("7d", 12.0, None),
            ),
        )
        d = as_usage_dict(usage)
        assert d["five_hour"]["pct"] == 37.0
        assert d["five_hour"]["resets_at"] == "2033-05-18T03:33:20Z"
        assert d["seven_day"] == {"pct": 12.0}

    def test_unrecognized_window_becomes_scoped_not_dropped(self):
        usage = CodexUsage(windows=(UsageWindow("3h", 99.0, None),))
        d = as_usage_dict(usage)
        assert "five_hour" not in d
        assert d["scoped"] == [{"name": "3h", "pct": 99.0}]

    def test_empty_windows_give_empty_dict(self):
        assert as_usage_dict(CodexUsage()) == {}


class TestCodexSwitchMargin:
    """Wall evaluation reuses oauth.switch_margin over the bridged shape —
    the same machinery `cswap auto` uses for Claude accounts."""

    def _usage(self, five_hour: float, seven_day: float) -> CodexUsage:
        return CodexUsage(
            windows=(
                UsageWindow("5h", five_hour, None),
                UsageWindow("7d", seven_day, None),
            )
        )

    def test_margin_is_distance_to_global_threshold(self):
        assert codex_switch_margin(self._usage(80.0, 40.0), 90.0, None) == 10.0

    def test_binding_window_wins(self):
        assert codex_switch_margin(self._usage(40.0, 85.0), 90.0, None) == 5.0

    def test_past_the_wall_is_negative(self):
        assert codex_switch_margin(self._usage(95.0, 10.0), 90.0, None) == -5.0

    def test_per_window_walls_override_the_global_threshold(self):
        # 7d held to a tighter wall than the global threshold: with 7d at
        # 97% and a 7d=96 wall, the weekly axis binds even though the 5h
        # window is nowhere near the global 99 threshold.
        margin = codex_switch_margin(
            self._usage(50.0, 97.0), 99.0, {"7d": 96.0}
        )
        assert margin == -1.0

    def test_unwalled_window_keeps_the_global_threshold(self):
        margin = codex_switch_margin(
            self._usage(94.0, 10.0), 95.0, {"7d": 98.0}
        )
        assert margin == 1.0

    def test_unrecognized_window_gates_via_scoped(self):
        usage = CodexUsage(
            windows=(UsageWindow("5h", 10.0, None), UsageWindow("3h", 99.5, None))
        )
        assert codex_switch_margin(usage, 90.0, None) == pytest.approx(-9.5)

    def test_no_windows_is_none(self):
        assert codex_switch_margin(CodexUsage(), 90.0, None) is None
