"""Inventory sees independent native, backup and recovery copies without writes."""

import base64
import json
from unittest.mock import Mock

import pytest

from claude_swap.exceptions import SessionError
from claude_swap.models import Platform
from claude_swap.switcher import ClaudeAccountSwitcher
from claude_swap.vision_handoff import _write_private
from claude_swap.vision_inventory import capture_inventory


def material(refresh="synthetic-refresh"):
    return json.dumps(
        {"claudeAiOauth": {"accessToken": "synthetic-access", "refreshToken": refresh}}
    )


def put(path, value, encoded=False):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if encoded:
        value = base64.b64encode(value.encode()).decode()
    _write_private(path, value)


@pytest.fixture
def setup(temp_home, monkeypatch):
    monkeypatch.setattr(Platform, "detect", lambda: Platform.LINUX)
    monkeypatch.setattr(
        "claude_swap.vision_inventory.profile_is_quiescent", lambda _: True
    )
    for key in ("CLAUDE_CONFIG_DIR", "CLAUDE_SECURESTORAGE_CONFIG_DIR"):
        monkeypatch.delenv(key, raising=False)
    return ClaudeAccountSwitcher(), temp_home


def test_inventory_finds_orphaned_backups_previous_and_stashed_copies(setup):
    switcher, home = setup
    paths = [home / ".claude" / ".credentials.json"]
    for name in (
        ".creds-9-other@example.invalid.enc",
        ".creds-9-other@example.invalid.enc.prev",
        ".unclaimed-orphan.enc",
        "interrupted.tmp",
    ):
        paths.append(switcher.credentials_dir / name)
    for index, path in enumerate(paths):
        put(path, material(), encoded=index > 0)
    before = {path: path.read_bytes() for path in paths}
    inventory = capture_inventory(switcher)
    public = inventory.public()
    assert len(public["candidates"]) == 5
    assert all(len(row["matching_sources"]) == 5 for row in public["candidates"])
    assert "synthetic-refresh" not in json.dumps(public)
    assert "synthetic-access" not in repr(inventory)
    assert before == {path: path.read_bytes() for path in paths}


def test_keychain_and_file_backups_are_inspected_independently(setup, monkeypatch):
    switcher, _ = setup
    switcher.backup_dir.mkdir(parents=True, exist_ok=True)
    switcher._write_json(
        switcher.sequence_file, {"accounts": {"1": {"email": "one@example.invalid"}}}
    )
    put(
        switcher.credentials_dir / ".creds-1-one@example.invalid.enc",
        material("file-grant"),
        True,
    )
    monkeypatch.setattr(Platform, "detect", lambda: Platform.MACOS)
    lookup = Mock(
        side_effect=lambda service, account: (
            material("keychain-grant")
            if account == "account-1-one@example.invalid"
            else None
        )
    )
    monkeypatch.setattr(
        "claude_swap.vision_inventory.macos_keychain.get_password", lookup
    )
    candidates = capture_inventory(switcher).public()["candidates"]
    assert len(candidates) == 2
    assert all(len(row["matching_sources"]) == 1 for row in candidates)


def test_unreadable_keychain_is_not_hidden_by_readable_file(setup, monkeypatch):
    switcher, home = setup
    put(home / ".claude" / ".credentials.json", material())
    monkeypatch.setattr(Platform, "detect", lambda: Platform.MACOS)
    monkeypatch.setattr(
        "claude_swap.vision_inventory.macos_keychain.get_password",
        Mock(side_effect=OSError("secret backend details")),
    )
    with pytest.raises(SessionError, match="unreadable") as error:
        capture_inventory(switcher)
    assert "secret backend" not in str(error.value)


def test_explicit_and_detached_native_profiles_are_included(setup):
    switcher, home = setup
    paths = [
        home / "external",
        switcher.backup_dir / "sessions" / "detached",
        switcher.backup_dir / "vision-sessions" / "origin" / "login",
    ]
    for path in paths:
        put(path / ".credentials.json", material())
    public = capture_inventory(switcher, [paths[0]]).public()
    assert len(public["candidates"]) == 3


@pytest.mark.parametrize(
    "raw", ["invalid-base64", base64.b64encode(b"invalid-json").decode()]
)
def test_damaged_recovery_copy_aborts_inventory(setup, raw):
    switcher, _ = setup
    put(switcher.credentials_dir / ".unclaimed-broken.enc", raw)
    with pytest.raises(SessionError, match="repair"):
        capture_inventory(switcher)


def test_symlink_profile_is_rejected(setup):
    switcher, home = setup
    target = home / "target"
    target.mkdir()
    link = home / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(SessionError, match="symbolic"):
        capture_inventory(switcher, [link])


def test_live_profile_is_reported_without_claiming_ownership(setup, monkeypatch):
    switcher, home = setup
    monkeypatch.setattr(
        "claude_swap.vision_inventory.profile_is_quiescent", lambda _: False
    )
    public = capture_inventory(switcher).public()
    assert public["state"] == "inventory_only"
    assert str(home / ".claude") in public["live_or_unreadable_profiles"]


def test_managed_escrow_requires_reconciliation_before_legacy_inventory(setup):
    switcher, _ = setup
    profile = switcher.backup_dir / "vision-logins" / "managed"
    put(
        profile / ".vision-handoff.json",
        json.dumps({"stores": {"keychain": None, "file": material()}}),
    )
    with pytest.raises(SessionError, match="Reconcile"):
        capture_inventory(switcher)


def test_keychain_lookup_preserves_raw_relative_profile_spelling(setup, monkeypatch):
    from claude_swap.session import keychain_service_name

    switcher, home = setup
    monkeypatch.chdir(home)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "relative-profile")
    monkeypatch.setattr(Platform, "detect", lambda: Platform.MACOS)
    lookup = Mock(return_value=None)
    monkeypatch.setattr(
        "claude_swap.vision_inventory.macos_keychain.get_password", lookup
    )
    capture_inventory(switcher)
    services = {call.args[0] for call in lookup.call_args_list}
    assert keychain_service_name("relative-profile") in services


def test_readonly_inventory_cli_returns_no_credential_material(setup):
    from claude_swap.vision_cli import run_command

    switcher, home = setup
    external = home / "external"
    put(external / ".credentials.json", material())
    result = run_command(["existing-logins", "--profile", str(external)], switcher)
    assert len(result["candidates"]) == 1
    assert "synthetic-refresh" not in json.dumps(result)
