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


def routed_fixture(setup):
    transaction, registry, source, _, _native, _backup, _other = setup
    switcher = transaction.switcher
    switcher._write_json(
        switcher.sequence_file,
        {
            "accounts": {
                "1": {"email": "test@example.invalid", "alias": "work"},
                "2": {"email": "other@example.invalid", "alias": "other"},
            },
            "sequence": [1, 2],
            "activeAccountNumber": 1,
        },
    )
    registry.client.discover = Mock(
        return_value=[
            {
                "account_id": "aia_00000000-0000-4000-8000-000000000002",
                "login_id": "ail_00000000-0000-4000-8000-000000000003",
                "provider": "claude",
                "email": "synthetic@example.invalid",
                "organization_id": "organization",
                "subscription": {"plan": "max"},
                "login_generation": 1,
            }
        ]
    )
    return transaction, registry, source


def test_committed_route_uses_verified_identity_preserves_alias_and_other_account(
    setup,
):
    transaction, registry, source = routed_fixture(setup)
    preview = transaction.preview(source)
    transaction.upload(source, preview["confirmation"])
    transaction.route_committed()
    switcher = transaction.switcher
    number, email, organization = switcher.resolve_account("work")
    assert int(number) > 2
    assert (email, organization) == (
        "synthetic@example.invalid",
        "organization",
    )
    assert switcher.resolve_account("other")[0] == "2"
    assert "1" not in switcher._get_sequence_data()["accounts"]
    assert switcher._get_sequence_data()["activeAccountNumber"] is None
    # Ordinary rediscovery must preserve migration alias routing.
    switcher.sync_vision_accounts = Mock()
    from claude_swap.vision_registry import RegistryPool

    RegistryPool(switcher, registry.client).sync(force=True)
    assert switcher.resolve_account("work")[0] == number


def test_lost_routing_journal_write_replays_without_reassigning_slots(
    setup, monkeypatch
):
    transaction, _, source = routed_fixture(setup)
    transaction.upload(source, transaction.preview(source)["confirmation"])
    save = transaction._save
    monkeypatch.setattr(
        transaction, "_save", Mock(side_effect=OSError("interrupted journal save"))
    )
    with pytest.raises(OSError):
        transaction.route_committed()
    first = transaction.switcher.resolve_account("work")
    monkeypatch.setattr(transaction, "_save", save)
    transaction.route_committed()
    assert transaction.switcher.resolve_account("work") == first


def test_migration_cli_preview_apply_and_recovery_are_bound_to_one_request(
    setup, monkeypatch
):
    from claude_swap.vision_cli import run_command

    transaction, registry, source = routed_fixture(setup)
    switcher = transaction.switcher
    monkeypatch.setattr(
        "claude_swap.vision_cli.configured_client", lambda: registry.client
    )
    monkeypatch.setattr("claude_swap.vision_cli.RegistrationClient", lambda _: registry)
    preview = run_command(["migrate-login", source], switcher)
    registry.prepare_registration.assert_not_called()
    result = run_command(
        [
            "migrate-login",
            source,
            "--request-id",
            preview["request_id"],
            "--confirm",
            preview["confirmation"],
        ],
        switcher,
    )
    assert result["state"] == "committed"
    assert switcher.resolve_account("work")[1] == "synthetic@example.invalid"
    assert run_command(["recover-migration", preview["request_id"]], switcher) == result
    registry.prepare_registration.assert_called_once()


def test_duplicate_local_slots_keep_aliases_on_one_central_account(setup):
    from claude_swap.exceptions import AccountNotFoundError, ConfigError

    transaction, _registry, source = routed_fixture(setup)
    switcher = transaction.switcher
    roster = switcher._get_sequence_data()
    roster["accounts"]["3"] = {"email": "duplicate@example.invalid", "alias": "second"}
    roster["sequence"].append(3)
    switcher._write_json(switcher.sequence_file, roster)
    put(
        switcher.credentials_dir / ".creds-3-duplicate@example.invalid.enc",
        material(),
        True,
    )
    transaction.upload(source, transaction.preview(source)["confirmation"])
    transaction.route_committed()
    number = switcher.resolve_account("work")[0]
    assert switcher.resolve_account("second")[0] == number
    assert len(switcher._get_sequence_data()["accounts"]) == 2
    assert {row[1] for row in switcher.list_aliases()} == {"work", "second", "other"}
    with pytest.raises(ConfigError):
        switcher.set_alias("other", "second")
    switcher.set_alias("work", "renamed")
    assert switcher.resolve_account("renamed")[0] == number
    with pytest.raises(AccountNotFoundError):
        switcher.resolve_account("second")


def test_migration_preserves_an_existing_remote_alias_and_disabled_preference(setup):
    from claude_swap.vision_registry import merge_accounts

    transaction, registry, source = routed_fixture(setup)
    switcher = transaction.switcher
    roster = merge_accounts(
        switcher._get_sequence_data(), registry.client.discover(), registry.client.url
    )
    remote = next(
        row for row in roster["accounts"].values() if row.get("source") == "vision"
    )
    remote["alias"] = "shared"
    remote["disabled"] = True
    switcher._write_json(switcher.sequence_file, roster)
    transaction.upload(source, transaction.preview(source)["confirmation"])
    transaction.route_committed()
    number = switcher.resolve_account("work")[0]
    assert switcher.resolve_account("shared")[0] == number
    remote = switcher._get_sequence_data()["accounts"][number]
    assert remote["alias"] == "shared"
    assert remote["disabled"] is True


def test_credential_handoff_leaves_native_history_in_place(
    setup,
):
    transaction, _registry, source = routed_fixture(setup)
    native_profile = next(
        item.source.location
        for item in transaction._capture().copies
        if item.source.id == source
    )
    from pathlib import Path

    profile = Path(native_profile).parent
    transcript = profile / "projects" / "project" / "session.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text("original conversation\n")
    transaction.upload(source, transaction.preview(source)["confirmation"])
    transcript.write_text("original conversation\nnew local session\n")
    transaction.route_committed()
    assert transcript.read_text().endswith("new local session\n")
    assert not transaction.history_path.exists()


def test_unrelated_live_profile_does_not_block_selected_idle_backup(setup, monkeypatch):
    transaction, registry, _, _, native, _, other = setup
    routed_fixture(setup)
    source = next(
        item.source.id
        for item in transaction._capture().copies
        if item.source.location == str(other)
    )
    preview = transaction.preview(source)
    monkeypatch.setattr(
        "claude_swap.vision_inventory.profile_is_quiescent",
        lambda profile: profile != native.parent,
    )
    original = native.read_bytes()
    assert transaction.upload(source, preview["confirmation"])["state"] == "committed"
    assert native.read_bytes() == original
    registry.confirm_registration.assert_called_once()


def test_selected_saved_slot_writer_blocks_even_without_native_credential_file(
    setup, monkeypatch
):
    transaction, registry, _, _, _, _, other = setup
    routed_fixture(setup)
    source = next(
        item.source.id
        for item in transaction._capture().copies
        if item.source.location == str(other)
    )
    slot_home = transaction.switcher._session_dir("2", "other@example.invalid")
    preview = transaction.preview(source)
    monkeypatch.setattr(
        "claude_swap.vision_inventory.profile_is_quiescent",
        lambda profile: profile != slot_home,
    )
    with pytest.raises(SessionError, match="selected login"):
        transaction.upload(source, preview["confirmation"])
    registry.prepare_registration.assert_not_called()


def test_another_native_profile_holding_selected_grant_is_fenced(setup, monkeypatch):
    transaction, registry, source, _, native, _, _ = setup
    copy_home = native.parent.parent / "another-profile"
    put(copy_home / ".credentials.json", material())
    transaction.extra_profiles = (str(copy_home),)
    preview = transaction.preview(source)
    monkeypatch.setattr(
        "claude_swap.vision_inventory.profile_is_quiescent",
        lambda profile: profile != copy_home,
    )
    with pytest.raises(SessionError, match="selected login"):
        transaction.upload(source, preview["confirmation"])
    registry.prepare_registration.assert_not_called()


@pytest.mark.parametrize("terminal", [False, True])
def test_recovery_retains_selected_writer_scope_after_credentials_are_deleted(
    setup, monkeypatch, terminal
):
    transaction, registry, source, confirmation, native, backup, _ = setup
    if not terminal:
        registry.confirm_registration.side_effect = VisionError("service_unavailable")
        with pytest.raises(VisionError):
            transaction.upload(source, confirmation)
    else:
        transaction.upload(source, confirmation)
    assert not native.exists() and not backup.exists()
    monkeypatch.setattr(
        "claude_swap.vision_inventory.profile_is_quiescent",
        lambda profile: profile != native.parent,
    )
    registry.registration_status.reset_mock()
    with pytest.raises(SessionError, match="selected login"):
        transaction.upload()
    registry.registration_status.assert_not_called()


def test_legacy_journal_recovery_keeps_conservative_all_profile_fencing(
    setup, monkeypatch
):
    import json

    transaction, registry, _, _, native, _, other = setup
    source = next(
        item.source.id
        for item in transaction._capture().copies
        if item.source.location == str(other)
    )
    registry.prepare_registration.side_effect = VisionError("service_unavailable")
    with pytest.raises(VisionError):
        transaction.upload(source, transaction.preview(source)["confirmation"])
    journal = json.loads(transaction.path.read_text())
    journal["version"] = 1
    del journal["affected_profiles"]
    _write_private(transaction.path, json.dumps(journal))
    monkeypatch.setattr(
        "claude_swap.vision_inventory.profile_is_quiescent",
        lambda profile: profile != native.parent,
    )
    with pytest.raises(SessionError, match="selected login"):
        transaction.upload()


def test_unreadable_unrelated_process_record_still_blocks_handoff(setup, monkeypatch):
    transaction, registry, _, _, native, _, other = setup
    source = next(
        item.source.id
        for item in transaction._capture().copies
        if item.source.location == str(other)
    )
    preview = transaction.preview(source)
    monkeypatch.setattr(
        "claude_swap.vision_existing_handoff.scan_live_sessions",
        lambda profile: ([], 1 if profile == native.parent else 0),
    )
    with pytest.raises(SessionError, match="unreadable"):
        transaction.upload(source, preview["confirmation"])
    registry.prepare_registration.assert_not_called()


def test_unreadable_unrelated_credential_source_still_blocks_handoff(setup):
    transaction, registry, source, confirmation, _, _, other = setup
    other.write_text("not-valid-base64")
    with pytest.raises(SessionError, match="needs repair"):
        transaction.upload(source, confirmation)
    registry.prepare_registration.assert_not_called()


def test_affected_scope_cannot_be_removed_from_recovery_journal(setup):
    import json

    transaction, registry, source, confirmation, _, _, _ = setup
    registry.confirm_registration.side_effect = VisionError("service_unavailable")
    with pytest.raises(VisionError):
        transaction.upload(source, confirmation)
    journal = json.loads(transaction.path.read_text())
    journal["affected_profiles"] = []
    _write_private(transaction.path, json.dumps(journal))
    with pytest.raises(SessionError, match="journal needs repair"):
        transaction.upload()


def test_new_profile_during_provider_request_blocks_handoff(setup):
    transaction, registry, source, confirmation, _, _, _ = setup

    def prepare(request, proof, credential):
        put(
            transaction.switcher.backup_dir
            / "vision-logins"
            / "new-profile"
            / ".credentials.json",
            material("unrelated"),
        )
        return receipt(request)

    registry.prepare_registration.side_effect = prepare
    with pytest.raises(SessionError, match="profiles changed"):
        transaction.upload(source, confirmation)
    registry.confirm_registration.assert_not_called()


def test_new_grant_copy_appearing_during_deletion_blocks_central_release(
    setup, monkeypatch
):
    transaction, registry, source, confirmation, _, _, _ = setup
    delete = transaction._delete

    def racing_copy(source):
        delete(source)
        put(
            transaction.switcher.credentials_dir / ".unclaimed-racing.enc",
            material(),
            True,
        )

    monkeypatch.setattr(transaction, "_delete", racing_copy)
    with pytest.raises(SessionError, match="Another refresh copy"):
        transaction.upload(source, confirmation)
    registry.confirm_registration.assert_not_called()
