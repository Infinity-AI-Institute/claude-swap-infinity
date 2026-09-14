"""Registration keeps server-verified identity and the original recovery proof."""

import io
import json
from unittest.mock import Mock

import pytest

from claude_swap.vision import VisionClient, VisionError
from claude_swap.vision_registration import RegistrationClient

REQUEST = "00000000-0000-4000-8000-000000000001"
PROOF = "a" * 43


def receipt(**changes):
    return {
        "version": 1,
        "request_id": REQUEST,
        "provider": "claude",
        "kind": "login_oauth",
        "state": "pending_handoff",
        "account_id": "aia_00000000-0000-4000-8000-000000000002",
        "login_id": "ail_00000000-0000-4000-8000-000000000003",
        "email": "verified@example.invalid",
        "organization_id": "organization",
        "expected_generation": 2,
        "generation": None,
        "expires_at": "2030-01-01T00:00:00Z",
        "capabilities": ["identity", "inference"],
        **changes,
    }


@pytest.fixture
def client():
    transport = VisionClient("https://vision.example.invalid", "synthetic-key")
    transport.request = Mock(return_value=receipt())
    return RegistrationClient(transport)


def test_upload_carries_actual_credential_without_alias_or_cached_identity(client):
    credential = {
        "accessToken": "synthetic-access",
        "refreshToken": "synthetic-refresh",
    }
    result = client.prepare_registration(REQUEST, PROOF, credential)
    body = client.client.request.call_args.args[2]
    assert body == {
        "version": 1,
        "request_id": REQUEST,
        "handoff_secret": PROOF,
        "provider": "claude",
        "kind": "login_oauth",
        "credential": credential,
    }
    assert result["email"] == "verified@example.invalid"


def test_subscription_token_cannot_smuggle_refresh_authority(client):
    client.client.request.return_value = receipt(kind="subscription_oauth_token")
    with pytest.raises(VisionError):
        client.prepare_registration(
            REQUEST,
            PROOF,
            {
                "accessToken": "synthetic",
                "refreshToken": "synthetic-refresh",
            },
            kind="subscription_oauth_token",
        )
    client.client.request.assert_not_called()
    assert (
        client.prepare_registration(
            REQUEST,
            PROOF,
            {"accessToken": "synthetic"},
            kind="subscription_oauth_token",
        )["kind"]
        == "subscription_oauth_token"
    )


@pytest.mark.parametrize("value", [False, None, 1, "true"])
def test_handoff_confirmation_requires_explicit_true(client, value):
    with pytest.raises(VisionError):
        client.confirm_registration(REQUEST, PROOF, local_refreshers_stopped=value)
    client.client.request.assert_not_called()


def test_status_recovers_by_original_request_and_header_proof(client):
    assert client.registration_status(REQUEST, PROOF) == receipt()
    client.client.request.assert_called_once_with(
        "GET",
        f"/api/ai-accounts/registrations/{REQUEST}",
        handoff_secret=PROOF,
    )


def test_confirm_and_cancel_never_resend_provider_tokens(client):
    client.client.request.return_value = receipt(state="committed", generation=3)
    client.confirm_registration(REQUEST, PROOF, local_refreshers_stopped=True)
    assert client.client.request.call_args.args[2] == {
        "version": 1,
        "handoff_secret": PROOF,
        "local_refreshers_stopped": True,
    }
    client.client.request.return_value = receipt(state="cancelled")
    client.cancel_registration(REQUEST, PROOF)
    assert client.client.request.call_args.args[2] == {
        "version": 1,
        "handoff_secret": PROOF,
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"request_id": "00000000-0000-4000-8000-000000000004"},
        {"provider": "codex"},
        {"accessToken": "must-not-appear"},
        {"state": "committed", "generation": None},
        {"generation": 7},
        {"expected_generation": True},
        {"expires_at": "not-a-time"},
    ],
)
def test_bad_or_secret_bearing_receipt_cannot_advance_handoff(client, changes):
    client.client.request.return_value = receipt(**changes)
    with pytest.raises(VisionError):
        client.registration_status(REQUEST, PROOF)


@pytest.mark.parametrize(
    "credential",
    [
        {"accessToken": "synthetic"},
        {"accessToken": "a\nheader", "refreshToken": "synthetic"},
        {"accessToken": "a\0value", "refreshToken": "synthetic"},
        {
            "accessToken": "synthetic",
            "refreshToken": "synthetic",
            "email": "claimed@example.invalid",
        },
    ],
)
def test_invalid_login_payload_is_rejected_before_network(client, credential):
    with pytest.raises(VisionError):
        client.prepare_registration(REQUEST, PROOF, credential)
    client.client.request.assert_not_called()


def test_status_proof_is_only_in_the_http_header(monkeypatch):
    sent = []

    class Opener:
        def open(self, request, timeout):
            sent.append(request)
            return io.BytesIO(json.dumps({"data": receipt()}).encode())

    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Opener())
    transport = VisionClient("https://vision.example.invalid", "synthetic-key")
    RegistrationClient(transport).registration_status(REQUEST, PROOF)
    assert sent[0].get_header("X-vision-handoff-secret") == PROOF
    assert PROOF not in sent[0].full_url
    assert sent[0].data is None
    assert sent[0].get_header("X-api-key") == "synthetic-key"
