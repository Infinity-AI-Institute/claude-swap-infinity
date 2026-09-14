"""Crash and cancellation cases use only isolated synthetic native stores."""

import json
import os
import uuid
from unittest.mock import Mock

import pytest

from claude_swap.exceptions import SessionError
from claude_swap.models import Platform
from claude_swap.vision import VisionClient, VisionError
from claude_swap.vision_handoff import ManagedLoginHandoff, _write_private
from claude_swap.vision_registration import RegistrationClient


def material(suffix="a"):
    return json.dumps(
        {
            "claudeAiOauth": {
                "accessToken": "synthetic-access-" + suffix,
                "refreshToken": "synthetic-refresh-" + suffix,
            }
        }
    )


def receipt(request, state="pending_handoff"):
    return {
        "version": 1,
        "request_id": request,
        "provider": "claude",
        "kind": "login_oauth",
        "state": state,
        "account_id": "aia_00000000-0000-4000-8000-000000000002",
        "login_id": "ail_00000000-0000-4000-8000-000000000003",
        "email": "synthetic@example.invalid",
        "organization_id": "organization",
        "expected_generation": 0,
        "generation": 1 if state == "committed" else None,
        "expires_at": "2030-01-01T00:00:00Z",
        "capabilities": ["identity", "inference"],
    }


@pytest.fixture
def setup(temp_home, monkeypatch):
    monkeypatch.setattr(Platform, "detect", lambda: Platform.LINUX)
    monkeypatch.setattr(
        "claude_swap.vision_handoff.profile_is_quiescent", lambda _: True
    )
    registry = RegistrationClient(
        VisionClient("https://vision.example.invalid", "synthetic-key")
    )
    registry.prepare_registration = Mock(
        side_effect=lambda request, proof, credential: receipt(request)
    )
    registry.confirm_registration = Mock(
        side_effect=lambda request, proof, **kwargs: receipt(request, "committed")
    )
    registry.registration_status = Mock(
        side_effect=lambda request, proof: receipt(request, "committed")
    )
    registry.cancel_registration = Mock(
        side_effect=lambda request, proof: receipt(request, "cancelled")
    )
    lease = ManagedLoginHandoff(temp_home / "backup", str(uuid.uuid4()), registry)
    with lease:
        _write_private(lease.auth_path, material())
    return lease, registry


def test_success_removes_native_refresh_material_and_retires_escrow(setup):
    lease, registry = setup
    with lease:
        result = lease.upload()
        assert result["state"] == "committed"
        assert not lease.auth_path.exists()
        assert "synthetic-refresh" not in lease.journal_path.read_text()
        assert lease.upload() == result
    registry.prepare_registration.assert_called_once()
    registry.confirm_registration.assert_called_once()


def test_lost_confirm_reply_recovers_original_request_without_restoring_grant(setup):
    lease, registry = setup
    registry.confirm_registration.side_effect = VisionError("service_unavailable")
    with lease:
        with pytest.raises(VisionError):
            lease.upload()
        assert not lease.auth_path.exists()
        saved = json.loads(lease.journal_path.read_text())
    with lease:
        assert lease.upload()["state"] == "committed"
        assert not lease.auth_path.exists()
    registry.registration_status.assert_called_once_with(
        saved["request_id"], saved["proof"]
    )
    registry.prepare_registration.assert_called_once()


def test_lost_prepare_reply_replays_same_proof_and_exact_credential(setup):
    lease, registry = setup
    registry.prepare_registration.side_effect = VisionError("service_unavailable")
    with lease:
        with pytest.raises(VisionError):
            lease.upload()
        first = registry.prepare_registration.call_args
        assert lease.auth_path.read_text() == material()
        registry.prepare_registration.side_effect = lambda request, proof, credential: (
            receipt(request)
        )
        lease.upload()
    assert registry.prepare_registration.call_args == first


def test_confirmed_cancel_restores_original_native_grant(setup):
    lease, registry = setup
    registry.confirm_registration.side_effect = VisionError("service_unavailable")
    with lease:
        with pytest.raises(VisionError):
            lease.upload()
        assert not lease.auth_path.exists()
        assert lease.cancel()["state"] == "cancelled"
        assert lease.auth_path.read_text() == material()


def test_cancel_cannot_overwrite_newer_native_login(setup):
    lease, registry = setup
    registry.confirm_registration.side_effect = VisionError("service_unavailable")
    with lease:
        with pytest.raises(VisionError):
            lease.upload()
        _write_private(lease.auth_path, material("b"))
        lease.cancel()
        assert lease.auth_path.read_text() == material("b")


def test_native_writer_appearing_during_fence_prevents_confirmation(setup, monkeypatch):
    lease, registry = setup
    checks = iter([True, True, False])
    monkeypatch.setattr(
        "claude_swap.vision_handoff.profile_is_quiescent", lambda _: next(checks)
    )
    with lease, pytest.raises(SessionError, match="Exit"):
        lease.upload()
    registry.confirm_registration.assert_not_called()
    assert "synthetic-refresh" in lease.journal_path.read_text()


def test_new_login_after_commit_gets_new_request_and_credential(setup):
    lease, registry = setup
    with lease:
        first = lease.upload()
        _write_private(lease.auth_path, material("b"))
        second = lease.upload()
    assert first["request_id"] != second["request_id"]
    assert (
        registry.prepare_registration.call_args.args[2]["refreshToken"]
        == "synthetic-refresh-b"
    )


def test_keychain_and_plaintext_copies_both_survive_a_cancel(setup, monkeypatch):
    lease, registry = setup
    monkeypatch.setattr(Platform, "detect", lambda: Platform.MACOS)
    keychain = {"value": material("keychain")}
    monkeypatch.setattr(
        "claude_swap.macos_keychain.get_password", lambda *args: keychain["value"]
    )
    monkeypatch.setattr(
        "claude_swap.macos_keychain.delete_password",
        lambda *args: keychain.update(value=None),
    )
    monkeypatch.setattr(
        "claude_swap.macos_keychain.set_password",
        lambda service, account, value: keychain.update(value=value),
    )
    registry.confirm_registration.side_effect = VisionError("service_unavailable")
    with lease:
        with pytest.raises(VisionError):
            lease.upload()
        assert keychain["value"] is None
        assert not lease.auth_path.exists()
        lease.cancel()
    assert keychain["value"] == material("keychain")
    assert lease.auth_path.read_text() == material()
    assert (
        registry.prepare_registration.call_args.args[2]["refreshToken"]
        == "synthetic-refresh-keychain"
    )


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode protection")
def test_public_credential_file_is_not_read_or_uploaded(setup):
    lease, registry = setup
    lease.auth_path.chmod(0o644)
    with lease, pytest.raises(SessionError, match="private"):
        lease.upload()
    registry.prepare_registration.assert_not_called()


def test_lost_committed_reply_cannot_delete_a_new_login(setup):
    lease, registry = setup
    registry.confirm_registration.side_effect = VisionError("service_unavailable")
    with lease:
        with pytest.raises(VisionError):
            lease.upload()
        _write_private(lease.auth_path, material("b"))
        assert lease.upload()["state"] == "committed"
        assert lease.auth_path.read_text() == material("b")
        assert "synthetic-refresh-a" not in lease.journal_path.read_text()


def test_native_refresh_locks_cover_registration_and_release_after_failure(setup):
    lease, registry = setup
    primary = lease.profile / ".oauth_refresh.lock"
    legacy = lease.profile.with_name(lease.profile.name + ".lock")

    def prepare(request, proof, credential):
        assert primary.is_dir()
        assert legacy.is_dir()
        raise VisionError("service_unavailable")

    registry.prepare_registration.side_effect = prepare
    with lease, pytest.raises(VisionError):
        lease.upload()
    assert not primary.exists()
    assert not legacy.exists()
    assert lease.auth_path.read_text() == material()


def test_busy_native_refresh_prevents_registration(setup, monkeypatch):
    from claude_swap.exceptions import ClaudeCodeLockTimeout

    lease, registry = setup
    primary = lease.profile / ".oauth_refresh.lock"
    primary.mkdir()
    monkeypatch.setattr("claude_swap.claude_locks.DEFAULT_TIMEOUT_S", 0)
    try:
        with lease, pytest.raises(ClaudeCodeLockTimeout):
            lease.upload()
        registry.prepare_registration.assert_not_called()
        assert not lease.journal_path.exists()
        assert lease.auth_path.read_text() == material()
    finally:
        primary.rmdir()


def test_cancel_restores_credential_under_native_refresh_locks(setup, monkeypatch):
    lease, registry = setup
    registry.confirm_registration.side_effect = VisionError("service_unavailable")
    with lease:
        with pytest.raises(VisionError):
            lease.upload()
        assert not lease.auth_path.exists()
        original_write = _write_private

        def checked_write(path, value):
            if path == lease.auth_path:
                assert (lease.profile / ".oauth_refresh.lock").is_dir()
                assert lease.profile.with_name(lease.profile.name + ".lock").is_dir()
            original_write(path, value)

        monkeypatch.setattr("claude_swap.vision_handoff._write_private", checked_write)
        lease.cancel()
        assert lease.auth_path.read_text() == material()
        assert not (lease.profile / ".oauth_refresh.lock").exists()
