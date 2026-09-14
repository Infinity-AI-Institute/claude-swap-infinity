"""Existing-login transfer uses isolated synthetic backends and registry replies."""

from unittest.mock import Mock

import pytest

from claude_swap.exceptions import SessionError
from claude_swap.models import Platform
from claude_swap.switcher import ClaudeAccountSwitcher
from claude_swap.vision import VisionClient, VisionError
from claude_swap.vision_existing_handoff import ExistingLoginHandoff
from claude_swap.vision_handoff import _write_private
from claude_swap.vision_registration import RegistrationClient
from tests.test_vision_handoff import material, receipt
from tests.test_vision_inventory import put


@pytest.fixture
def setup(temp_home, monkeypatch):
    monkeypatch.setattr(Platform, "detect", lambda: Platform.LINUX)
    monkeypatch.setattr(
        "claude_swap.vision_inventory.profile_is_quiescent", lambda _: True
    )
    for key in ("CLAUDE_CONFIG_DIR", "CLAUDE_SECURESTORAGE_CONFIG_DIR"):
        monkeypatch.delenv(key, raising=False)
    switcher = ClaudeAccountSwitcher()
    native = temp_home / ".claude" / ".credentials.json"
    backup = switcher.credentials_dir / ".creds-1-test@example.invalid.enc"
    other = switcher.credentials_dir / ".creds-2-other@example.invalid.enc"
    put(native, material())
    put(backup, material(), True)
    put(other, material("b"), True)
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
    transaction = ExistingLoginHandoff.new(switcher, registry)
    source = next(
        item.source.id
        for item in transaction._capture().copies
        if item.source.location == str(native)
    )
    preview = transaction.preview(source)
    return transaction, registry, source, preview["confirmation"], native, backup, other


def test_transfer_removes_all_matching_copies_preserves_unselected_and_retires_escrow(
    setup,
):
    transaction, registry, source, confirmation, native, backup, other = setup
    other_bytes = other.read_bytes()
    result = transaction.upload(source, confirmation)
    assert result["state"] == "committed"
    assert not native.exists() and not backup.exists()
    assert other.read_bytes() == other_bytes
    assert "synthetic-refresh" not in transaction.path.read_text()
    registry.confirm_registration.assert_called_once()
    assert transaction.upload() == result


def test_lost_confirm_reply_recovers_without_restoring_transferred_grant(setup):
    transaction, registry, source, confirmation, native, backup, _ = setup
    registry.confirm_registration.side_effect = VisionError("service_unavailable")
    with pytest.raises(VisionError):
        transaction.upload(source, confirmation)
    assert not native.exists() and not backup.exists()
    assert "synthetic-refresh" in transaction.path.read_text()
    assert transaction.upload()["state"] == "committed"
    assert not native.exists() and not backup.exists()
    registry.prepare_registration.assert_called_once()


def test_cancel_restores_exact_backend_bytes_after_partial_fence(setup, monkeypatch):
    transaction, registry, source, confirmation, native, backup, _ = setup
    originals = {path: path.read_bytes() for path in (native, backup)}
    delete = transaction._delete
    calls = []

    def interrupted(source):
        delete(source)
        calls.append(source.id)
        if len(calls) == 1:
            raise OSError("synthetic interruption")

    monkeypatch.setattr(transaction, "_delete", interrupted)
    with pytest.raises(OSError):
        transaction.upload(source, confirmation)
    registry.confirm_registration.assert_not_called()
    assert transaction.cancel()["state"] == "cancelled"
    assert all(path.read_bytes() == raw for path, raw in originals.items())
    assert "synthetic-refresh" not in transaction.path.read_text()


def test_cancel_does_not_resurrect_old_fallback_when_new_native_login_exists(setup):
    transaction, registry, source, confirmation, native, backup, _ = setup
    registry.confirm_registration.side_effect = VisionError("service_unavailable")
    with pytest.raises(VisionError):
        transaction.upload(source, confirmation)
    _write_private(native, material("new"))
    transaction.cancel()
    assert native.read_text() == material("new")
    assert not backup.exists()


def test_changed_preview_aborts_before_registration(setup):
    transaction, registry, source, confirmation, native, _, _ = setup
    _write_private(native, material("new"))
    with pytest.raises(SessionError, match="preview changed"):
        transaction.upload(source, confirmation)
    registry.prepare_registration.assert_not_called()
    assert not transaction.path.exists()


def test_new_copy_during_prepare_prevents_confirmation(setup):
    transaction, registry, source, confirmation, _, _, _ = setup

    def prepare(request, proof, credential):
        put(
            transaction.switcher.credentials_dir / ".unclaimed-new.enc",
            material(),
            True,
        )
        return receipt(request)

    registry.prepare_registration.side_effect = prepare
    with pytest.raises(SessionError, match="Another refresh copy"):
        transaction.upload(source, confirmation)
    registry.confirm_registration.assert_not_called()


def test_live_session_prevents_registration(setup, monkeypatch):
    transaction, registry, source, confirmation, _, _, _ = setup
    monkeypatch.setattr(
        "claude_swap.vision_inventory.profile_is_quiescent", lambda _: False
    )
    with pytest.raises(SessionError, match="Exit"):
        transaction.upload(source, confirmation)
    registry.prepare_registration.assert_not_called()


def test_lost_prepare_reply_reuses_exact_request_and_proof(setup):
    transaction, registry, source, confirmation, _, _, _ = setup
    registry.prepare_registration.side_effect = VisionError("service_unavailable")
    with pytest.raises(VisionError):
        transaction.upload(source, confirmation)
    original = registry.prepare_registration.call_args
    registry.prepare_registration.side_effect = lambda request, proof, credential: (
        receipt(request)
    )
    transaction.upload()
    assert registry.prepare_registration.call_args == original


def test_registry_http_does_not_hold_roster_lock_but_holds_native_locks(setup):
    from claude_swap.locking import FileLock

    transaction, registry, source, confirmation, native, _, _ = setup

    def prepare(request, proof, credential):
        lock = FileLock(transaction.switcher.lock_file, timeout=0)
        assert lock.acquire()
        lock.release()
        assert (native.parent / ".oauth_refresh.lock").is_dir()
        assert native.parent.with_name(native.parent.name + ".lock").is_dir()
        return receipt(request)

    registry.prepare_registration.side_effect = prepare
    transaction.upload(source, confirmation)


def test_roster_reassignment_blocks_rollback_instead_of_restoring_to_new_owner(setup):
    transaction, registry, source, confirmation, native, backup, _ = setup
    registry.confirm_registration.side_effect = VisionError("service_unavailable")
    with pytest.raises(VisionError):
        transaction.upload(source, confirmation)
    transaction.switcher._write_json(
        transaction.switcher.sequence_file,
        {"accounts": {"1": {"email": "reassigned@example.invalid"}}},
    )
    with pytest.raises(SessionError, match="roster changed"):
        transaction.cancel()
    assert not native.exists() and not backup.exists()
    assert "synthetic-refresh" in transaction.path.read_text()


def test_journal_cannot_redirect_restoration_outside_inventory(setup):
    import json

    transaction, registry, source, confirmation, native, _, _ = setup
    registry.confirm_registration.side_effect = VisionError("service_unavailable")
    with pytest.raises(VisionError):
        transaction.upload(source, confirmation)
    journal = json.loads(transaction.path.read_text())
    journal["copies"][0]["source"]["location"] = str(native.parent / "unrelated.json")
    _write_private(transaction.path, json.dumps(journal))
    with pytest.raises(SessionError, match="journal needs repair"):
        transaction.cancel()
    registry.cancel_registration.assert_not_called()


def test_keychain_and_file_copies_transfer_together(setup, monkeypatch):
    from claude_swap.session import keychain_service_name

    transaction, _, source, _, native, backup, _ = setup
    monkeypatch.setattr(Platform, "detect", lambda: Platform.MACOS)
    service = keychain_service_name(native.parent)
    keychain = {service: material()}
    monkeypatch.setattr(
        "claude_swap.vision_inventory.macos_keychain.get_password",
        lambda service, account: keychain.get(service),
    )
    monkeypatch.setattr(
        "claude_swap.vision_existing_handoff.macos_keychain.delete_password",
        lambda service, account: keychain.pop(service, None),
    )
    preview = transaction.preview(source)
    assert len(preview["selected"]["matching_sources"]) == 3
    transaction.upload(source, preview["confirmation"])
    assert not keychain
    assert not native.exists() and not backup.exists()


def test_busy_local_consumer_prevents_registration(setup, monkeypatch):
    from claude_swap.exceptions import LockError
    from claude_swap.locking import FileLock

    transaction, registry, source, _, _, _, _ = setup
    switcher = transaction.switcher
    switcher._write_json(
        switcher.sequence_file, {"accounts": {"1": {"email": "test@example.invalid"}}}
    )
    preview = transaction.preview(source)
    original = FileLock.acquire

    def immediate(self, timeout=None):
        return original(self, timeout=0)

    monkeypatch.setattr(FileLock, "acquire", immediate)
    with (
        FileLock(switcher.credentials_dir / ".consume-1.lock"),
        pytest.raises(LockError),
    ):
        transaction.upload(source, preview["confirmation"])
    registry.prepare_registration.assert_not_called()
