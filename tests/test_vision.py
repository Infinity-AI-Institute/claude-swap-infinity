"""Claude-specific registry contract checks; all credentials are synthetic."""

import io
import json
import time
import urllib.error
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from claude_swap.vision import MAX_RESPONSE_BYTES, VisionClient, VisionError


def item(index=1):
    return {
        "account_id": f"aia_00000000-0000-4000-8000-{index:012d}",
        "login_id": f"ail_00000000-0000-4000-8000-{index:012d}",
        "provider": "claude",
        "email": f"user{index}@example.invalid",
        "organization_id": "organization",
        "title": None,
        "subscription": {
            "plan": "max",
            "rawTier": "default_claude_max_20x",
            "multiplier": 20,
        },
        "login_generation": 1,
    }


def credential(kind="login_oauth"):
    metadata = item()
    return {
        "version": 1,
        **{
            key: metadata[key]
            for key in (
                "account_id",
                "login_id",
                "provider",
                "email",
                "organization_id",
                "subscription",
            )
        },
        "kind": kind,
        "generation": 1,
        "accessToken": "synthetic-access-token",
        "capabilities": ["identity", "inference"],
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": datetime.fromtimestamp(
            time.time() + 3600, timezone.utc
        ).isoformat(),
    }


def client():
    return VisionClient("https://vision.example.invalid", "synthetic-private-key")


def test_complete_paginated_claude_pool_preserves_plan_and_discards_secret_shaped_metadata():
    registry = client()
    first = [item(i) for i in range(1, 101)]
    first[0]["subscription"]["accessToken"] = "must-not-be-returned"
    registry.request = Mock(
        side_effect=[
            {"version": 1, "items": first, "next_cursor": first[-1]["login_id"]},
            {"version": 1, "items": [item(101)], "next_cursor": None},
        ]
    )
    pool = registry.discover()
    assert len(pool) == 101
    assert pool[0]["subscription"]["multiplier"] == 20
    assert "must-not-be-returned" not in json.dumps(pool)
    assert all(
        "provider=claude" in call.args[1] for call in registry.request.call_args_list
    )
    assert "synthetic-private-key" not in repr(registry)


@pytest.mark.parametrize(
    "second",
    [
        {"version": 1, "items": [item(1)], "next_cursor": None},
        {"version": 1, "items": [], "next_cursor": item(1)["login_id"]},
        {
            "version": 1,
            "items": [{**item(2), "provider": "codex"}],
            "next_cursor": None,
        },
    ],
)
def test_bad_second_page_cannot_return_a_partial_pool(second):
    registry = client()
    registry.request = Mock(
        side_effect=[
            {"version": 1, "items": [item(1)], "next_cursor": item(1)["login_id"]},
            second,
        ]
    )
    with pytest.raises(VisionError):
        registry.discover()


@pytest.mark.parametrize("kind", ["login_oauth", "subscription_oauth_token"])
@pytest.mark.parametrize("unknown_expiry", [False, True])
def test_both_subscription_kinds_accept_verified_access_credentials(
    kind, unknown_expiry
):
    registry = client()
    value = credential(kind)
    if unknown_expiry:
        value["expires_at"] = None
    registry.request = Mock(return_value=value)
    result = registry.credential(value["account_id"], value["login_id"])
    assert result["kind"] == kind
    assert result["expires_at"] == value["expires_at"]
    assert "refreshToken" not in result


@pytest.mark.parametrize(
    "change",
    [
        {"provider": "codex"},
        {"refreshToken": "forbidden-private-refresh"},
        {"generation": True},
        {"expires_at": "2000-01-01T00:00:00Z"},
        {"verified_at": None},
        {"accessToken": "bad\r\nheader"},
        {"login_id": item(2)["login_id"]},
    ],
)
def test_invalid_delivery_is_refused_without_retaining_secret_errors(
    change,
):
    registry = client()
    value = credential()
    registry.request = Mock(return_value={**value, **change})
    with pytest.raises(VisionError) as result:
        registry.credential(value["account_id"], value["login_id"])
    assert "private-refresh" not in str(result.value)


@pytest.mark.parametrize(
    "url",
    [
        "http://remote.example.invalid",
        "https://user:secret@vision.example.invalid",
        "https://vision.example.invalid?key=secret",
        "file:///private/tmp/credential",
    ],
)
def test_invalid_origin_cannot_send_a_key(url):
    with pytest.raises(VisionError) as result:
        VisionClient(url, "private-key")
    assert "private-key" not in str(result.value) and "secret" not in str(result.value)


def test_transport_rejects_oversized_delivery(monkeypatch):
    class Opener:
        def open(self, req, timeout):
            return io.BytesIO(b" " * (MAX_RESPONSE_BYTES + 1))

    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Opener())
    with pytest.raises(VisionError):
        client().request("GET", "/api/ai-accounts/registry")


@pytest.mark.parametrize(
    ("status", "code", "advice"),
    [(401, "unauthorized", "API key"), (403, "not_permitted", "Vision admin")],
)
def test_refused_key_error_names_the_vision_url_and_the_fix(
    monkeypatch, status, code, advice
):
    """A key-only user's next step comes from Vision's refusal, not a Claude login."""

    class Opener:
        def open(self, req, timeout):
            body = io.BytesIO(json.dumps({"error": {"code": code}}).encode())
            raise urllib.error.HTTPError(req.full_url, status, "refused", {}, body)

    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Opener())
    with pytest.raises(VisionError) as refused:
        client().discover()
    message = str(refused.value)
    assert refused.value.code == code
    assert "https://vision.example.invalid" in message
    assert advice in message
    assert "synthetic-private-key" not in message


def test_service_errors_keep_their_short_message():
    assert str(VisionError("service_unavailable")) == (
        "Vision account registry: service_unavailable."
    )


def test_refresh_request_is_generation_fenced_and_never_calls_a_provider():
    registry = client()
    value = item()
    registry.request = Mock(
        return_value={"version": 1, "generation": 2, "state": "queued"}
    )
    assert (
        registry.refresh(value["account_id"], value["login_id"], 1)["state"] == "queued"
    )
    assert registry.request.call_args.args[2] == {"version": 1, "generation": 1}
    assert registry.request.call_args.args[1].endswith("/refresh")


@pytest.mark.parametrize("kind", ["login_oauth", "subscription_oauth_token"])
@pytest.mark.parametrize("capabilities", [[], ["identity"], ["profile"]])
def test_provider_verified_access_does_not_require_inference_qualification(kind, capabilities):
    registry = client()
    value = {**credential(kind), "capabilities": capabilities}
    registry.request = Mock(return_value=value)
    assert registry.credential(value["account_id"], value["login_id"]) == value
