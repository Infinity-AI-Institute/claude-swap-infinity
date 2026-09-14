"""Batch previews bind selection and credential versions without uploading."""

import json
from unittest.mock import Mock

import pytest

from claude_swap.exceptions import SessionError
from claude_swap.models import Platform
from claude_swap.switcher import ClaudeAccountSwitcher
from claude_swap.vision import VisionClient, VisionError
from claude_swap.vision_batch import BatchUpload
from claude_swap.vision_cli import ManagedProfiles
from claude_swap.vision_handoff import ManagedLoginHandoff, _write_private
from claude_swap.vision_registration import RegistrationClient


def material(name):
    return json.dumps(
        {
            "claudeAiOauth": {
                "accessToken": "synthetic-access-" + name,
                "refreshToken": "synthetic-refresh-" + name,
            }
        }
    )


@pytest.fixture
def setup(temp_home, monkeypatch):
    monkeypatch.setattr(Platform, "detect", lambda: Platform.LINUX)
    monkeypatch.setattr(
        "claude_swap.vision_handoff.profile_is_quiescent", lambda _: True
    )
    switcher = ClaudeAccountSwitcher()
    client = VisionClient("https://vision.example.invalid", "synthetic-registry-key")
    client.discover = Mock(return_value=[])
    paths = {}
    for name in ("one", "two", "three"):
        profile = ManagedProfiles(switcher.backup_dir).profile(name, create=True)
        with ManagedLoginHandoff(
            switcher.backup_dir, profile, RegistrationClient(client)
        ) as lease:
            _write_private(lease.auth_path, material(name))
            paths[name] = lease.auth_path

    def request(method, path, body=None, **kwargs):
        request_id = (
            body["request_id"]
            if path.endswith("registrations")
            else path.split("/")[-2]
        )
        committed = path.endswith("confirm-handoff")
        return {
            "version": 1,
            "request_id": request_id,
            "provider": "claude",
            "kind": "login_oauth",
            "state": "committed" if committed else "pending_handoff",
            "account_id": "aia_00000000-0000-4000-8000-000000000002",
            "login_id": "ail_00000000-0000-4000-8000-000000000003",
            "email": "synthetic@example.invalid",
            "organization_id": "organization",
            "expected_generation": 0,
            "generation": 1 if committed else None,
            "expires_at": "2030-01-01T00:00:00Z",
            "capabilities": ["identity", "inference"],
        }

    client.request = Mock(side_effect=request)
    return BatchUpload(switcher, client), client, paths


def test_preview_has_no_network_or_secret_material(setup):
    batch, client, paths = setup
    before = {name: path.read_bytes() for name, path in paths.items()}
    result = batch.preview(["one", "three"])
    assert [item["profile"] for item in result["items"]] == ["one", "three"]
    assert len(result["confirmation"]) == 64
    assert "synthetic-access" not in json.dumps(result)
    assert "synthetic-refresh" not in json.dumps(result)
    assert client.api_key not in json.dumps(result)
    client.request.assert_not_called()
    assert {name: path.read_bytes() for name, path in paths.items()} == before


def test_only_selected_profiles_are_uploaded(setup):
    batch, client, paths = setup
    preview = batch.preview(["one", "three"])
    result = batch.apply(["one", "three"], preview["confirmation"])
    assert [item["state"] for item in result["items"]] == ["committed", "committed"]
    assert paths["two"].read_text() == material("two")
    assert not paths["one"].exists() and not paths["three"].exists()
    uploaded = [
        call.args[2]["credential"]["accessToken"]
        for call in client.request.call_args_list
        if call.args[1].endswith("registrations")
    ]
    assert uploaded == ["synthetic-access-one", "synthetic-access-three"]


@pytest.mark.parametrize("change", ["credential", "key", "selection"])
def test_changed_preview_cannot_send_any_credentials(setup, change):
    batch, client, paths = setup
    names = ["one", "two"]
    preview = batch.preview(names)
    if change == "credential":
        _write_private(paths["two"], material("new"))
    elif change == "key":
        client.api_key = "another-synthetic-key"
    else:
        names = ["one", "three"]
    with pytest.raises(SessionError, match="review a new"):
        batch.apply(names, preview["confirmation"])
    client.request.assert_not_called()


def test_one_failed_item_does_not_abandon_other_selected_items(setup):
    batch, client, paths = setup
    original = client.request.side_effect

    def request(method, path, body=None, **kwargs):
        if path.endswith("registrations") and body["credential"][
            "accessToken"
        ].endswith("one"):
            raise VisionError("verification_unavailable")
        return original(method, path, body, **kwargs)

    client.request.side_effect = request
    preview = batch.preview(["one", "two"])
    result = batch.apply(["one", "two"], preview["confirmation"])
    assert [item["state"] for item in result["items"]] == ["unavailable", "committed"]
    assert paths["one"].exists() and not paths["two"].exists()


def test_later_item_changed_during_apply_is_rechecked_under_its_lease(setup):
    batch, client, paths = setup
    original = client.request.side_effect

    def request(method, path, body=None, **kwargs):
        if path.endswith("confirm-handoff"):
            _write_private(paths["two"], material("new"))
        return original(method, path, body, **kwargs)

    client.request.side_effect = request
    preview = batch.preview(["one", "two"])
    result = batch.apply(["one", "two"], preview["confirmation"])
    assert [item["state"] for item in result["items"]] == ["committed", "unavailable"]
    assert paths["two"].read_text() == material("new")


@pytest.mark.parametrize("names", [[], ["one", "one"], ["one"] * 101])
def test_invalid_selection_is_rejected_before_upload(setup, names):
    batch, client, _ = setup
    with pytest.raises(SessionError):
        batch.preview(names)
    client.request.assert_not_called()


def test_keychain_failure_is_item_scoped_and_does_not_echo_backend_details(
    setup, monkeypatch
):
    from claude_swap.macos_keychain import KeychainError

    batch, client, _ = setup
    monkeypatch.setattr(Platform, "detect", lambda: Platform.MACOS)
    monkeypatch.setattr(
        "claude_swap.macos_keychain.get_password",
        Mock(side_effect=KeychainError("synthetic-secret-detail")),
    )
    result = batch.preview(["one", "two"])
    assert [item["state"] for item in result["items"]] == ["unavailable", "unavailable"]
    assert "synthetic-secret-detail" not in json.dumps(result)
    client.request.assert_not_called()
