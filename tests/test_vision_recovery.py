"""A provider rejection must not replay its rejected central access token."""

from unittest.mock import Mock

import pytest

from claude_swap.vision import VisionError
from claude_swap.vision_session import recover_rejected_credential


@pytest.fixture
def rejected():
    return {
        "account_id": "aia_00000000-0000-4000-8000-000000000001",
        "login_id": "ail_00000000-0000-4000-8000-000000000001",
        "generation": 4,
        "kind": "login_oauth",
        "email": "synthetic@example.invalid",
        "organization_id": "organization",
        "accessToken": "synthetic-old",
    }


def test_already_rotated_credential_needs_no_refresh(rejected):
    client = Mock()
    newer = {**rejected, "generation": 5, "accessToken": "synthetic-new"}
    client.credential.return_value = newer
    assert recover_rejected_credential(client, rejected) == newer
    client.refresh.assert_not_called()


def test_pending_refresh_waits_for_different_token_and_generation(
    rejected, monkeypatch
):
    client = Mock()
    newer = {**rejected, "generation": 5, "accessToken": "synthetic-new"}
    client.credential.side_effect = [rejected, rejected, newer]
    client.refresh.return_value = {"state": "running"}
    sleep = Mock()
    monkeypatch.setattr("claude_swap.vision_session.time.sleep", sleep)
    assert recover_rejected_credential(client, rejected) == newer
    client.refresh.assert_called_once_with(
        rejected["account_id"], rejected["login_id"], 4
    )
    sleep.assert_called_once()


@pytest.mark.parametrize("state", ["current", "reauth_required", "obsolete"])
def test_terminal_or_not_due_refresh_does_not_replay_rejected_token(rejected, state):
    client = Mock()
    client.credential.return_value = rejected
    client.refresh.return_value = {"state": state}
    assert recover_rejected_credential(client, rejected) is None
    assert client.credential.call_count == 1


@pytest.mark.parametrize(
    "change", [{"generation": 5}, {"accessToken": "synthetic-new"}]
)
def test_both_new_generation_and_new_token_are_required(rejected, change):
    client = Mock()
    client.credential.return_value = {**rejected, **change}
    client.refresh.return_value = {"state": "running"}
    assert recover_rejected_credential(client, rejected, wait_seconds=0) is None


def test_registry_revocation_stops_recovery(rejected):
    client = Mock()
    client.credential.side_effect = VisionError("not_permitted", status=403)
    with pytest.raises(VisionError):
        recover_rejected_credential(client, rejected)
    client.refresh.assert_not_called()


def test_subscription_token_never_requests_oauth_refresh(rejected):
    rejected["kind"] = "subscription_oauth_token"
    client = Mock()
    client.credential.return_value = rejected
    assert recover_rejected_credential(client, rejected) is None
    client.refresh.assert_not_called()


def test_identity_change_is_rejected(rejected):
    client = Mock()
    client.credential.return_value = {
        **rejected,
        "generation": 5,
        "accessToken": "new",
        "email": "wrong@example.invalid",
    }
    with pytest.raises(VisionError):
        recover_rejected_credential(client, rejected)
    client.refresh.assert_not_called()
