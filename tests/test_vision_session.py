"""Remote launch receives access-only auth and preserves native session identity."""

from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from claude_swap.exceptions import SessionError
from claude_swap.session import SessionManager
from claude_swap.switcher import ClaudeAccountSwitcher
from claude_swap.vision import VisionClient, VisionError
from claude_swap.vision_session import acquire_credential, prepare_launch


@pytest.fixture
def setup(temp_home, monkeypatch):
    monkeypatch.delenv("VISION_API_KEY", raising=False)
    switcher = ClaudeAccountSwitcher()
    manager = SessionManager(switcher)
    registry = VisionClient("https://vision.example.invalid", "synthetic-vision-key")
    login = "ail_00000000-0000-4000-8000-000000000001"
    account = "aia_00000000-0000-4000-8000-000000000001"
    record = {
        "source": "vision",
        "visionUrl": registry.url,
        "visionLoginId": login,
        "visionAccountId": account,
        "visionGeneration": 4,
        "email": "synthetic@example.invalid",
        "organizationUuid": "organization",
        "alias": "one",
    }
    token = {
        "email": record["email"],
        "organization_id": record["organizationUuid"],
        "login_id": login,
        "account_id": account,
        "generation": 4,
        "expires_at": datetime.fromtimestamp(2_000_000_000, UTC).isoformat(),
        "accessToken": "synthetic-access-token",
    }
    registry.credential = Mock(return_value=token)
    registry.refresh = Mock()
    return manager, registry, record, token


def test_launch_passes_only_access_token_and_scrubs_inherited_auth_routes(
    setup, monkeypatch
):
    manager, registry, record, _ = setup
    for key in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "VISION_API_KEY",
    ):
        monkeypatch.setenv(key, "must-not-inherit")
    manager.switcher._store._read_account_credentials = Mock(
        side_effect=AssertionError("local credential read")
    )
    launch = prepare_launch(manager, record, registry, share=False, share_history=False)
    assert launch.env["CLAUDE_CODE_OAUTH_TOKEN"] == "synthetic-access-token"
    assert "must-not-inherit" not in launch.env.values()
    assert launch.env["CLAUDE_CONFIG_DIR"] == str(launch.directory)
    assert "synthetic-access-token" not in repr(launch)
    assert not (launch.directory / ".credentials.json").exists()
    assert all(
        "synthetic-access-token" not in p.read_text()
        for p in launch.directory.rglob("*")
        if p.is_file()
    )
    registry.refresh.assert_not_called()


def test_relogin_generation_and_alias_changes_preserve_native_history(setup):
    manager, registry, record, token = setup
    first = prepare_launch(manager, record, registry, share=False, share_history=False)
    history = first.directory / "history.jsonl"
    history.write_text("synthetic native conversation\n")
    registry.credential.return_value = {
        **token,
        "generation": 5,
        "accessToken": "new-synthetic-token",
    }
    second = prepare_launch(
        manager,
        {**record, "alias": "renamed"},
        registry,
        share=False,
        share_history=False,
    )
    assert second.directory == first.directory
    assert second.generation == 5
    assert history.read_text() == "synthetic native conversation\n"


def test_unavailable_access_requests_only_central_refresh_with_original_generation(
    setup,
):
    _, registry, record, token = setup
    registry.credential.side_effect = [VisionError("credential_unavailable"), token]
    registry.refresh.return_value = {"state": "queued"}
    assert acquire_credential(registry, record) == token
    registry.refresh.assert_called_once_with(
        record["visionAccountId"], record["visionLoginId"], 4
    )


def test_revocation_does_not_trigger_refresh_or_create_profile(setup):
    manager, registry, record, _ = setup
    registry.credential.side_effect = VisionError("not_permitted", 403)
    with pytest.raises(VisionError):
        prepare_launch(manager, record, registry, share=False, share_history=False)
    registry.refresh.assert_not_called()
    assert not (manager.switcher.backup_dir / "vision-sessions").exists()


def test_pending_refresh_has_bounded_wait(setup):
    _, registry, record, _ = setup
    registry.credential.side_effect = VisionError("credential_unavailable")
    registry.refresh.return_value = {"state": "running"}
    with pytest.raises(SessionError, match="pending"):
        acquire_credential(registry, record, wait_seconds=0)
    registry.refresh.assert_called_once()


@pytest.mark.parametrize(
    "change", [{"disabled": True}, {"visionUrl": "https://another.example.invalid"}]
)
def test_disabled_or_other_registry_selection_never_delivers_credentials(setup, change):
    manager, registry, record, _ = setup
    with pytest.raises(SessionError):
        prepare_launch(
            manager, {**record, **change}, registry, share=False, share_history=False
        )
    registry.credential.assert_not_called()


def test_existing_native_login_is_preserved_and_requires_handoff(setup):
    manager, registry, record, _ = setup
    launch = prepare_launch(manager, record, registry, share=False, share_history=False)
    auth = launch.directory / ".credentials.json"
    auth.write_text("synthetic native login")
    with pytest.raises(SessionError, match="handoff"):
        prepare_launch(manager, record, registry, share=False, share_history=False)
    assert auth.read_text() == "synthetic native login"


def test_remote_run_bypasses_local_bootstrap_and_preserves_resume_arguments(
    setup, monkeypatch
):
    manager, registry, record, _ = setup
    manager.switcher.backup_dir.mkdir(parents=True, exist_ok=True)
    manager.switcher._write_json(
        manager.switcher.sequence_file,
        {
            "accounts": {"1": record},
            "sequence": [1],
            "activeAccountNumber": None,
        },
    )
    manager.switcher.sync_vision_accounts = Mock()
    monkeypatch.setattr("claude_swap.vision.configured_client", lambda: registry)
    monkeypatch.setattr(
        "claude_swap.session.shutil.which", lambda _: "/synthetic/claude"
    )
    manager.setup_session = Mock(side_effect=AssertionError("local bootstrap"))
    manager._exec = Mock(side_effect=SystemExit(0))
    with pytest.raises(SystemExit):
        manager.run(
            "1", ["--resume", "synthetic-session", "--model", "sonnet"], share=False
        )
    assert manager._exec.call_args.args == (
        "/synthetic/claude",
        ["--resume", "synthetic-session", "--model", "sonnet"],
    )
    assert (
        manager._exec.call_args.kwargs["env"]["CLAUDE_CODE_OAUTH_TOKEN"]
        == "synthetic-access-token"
    )


@pytest.mark.parametrize("unreadable", [False, True])
def test_keychain_native_login_or_unknown_store_blocks_remote_launch(
    setup, monkeypatch, unreadable
):
    from claude_swap import macos_keychain
    from claude_swap.models import Platform
    from claude_swap.session import keychain_service_name

    manager, registry, record, _ = setup
    monkeypatch.setattr(Platform, "detect", lambda: Platform.LINUX)
    first = prepare_launch(manager, record, registry, share=False, share_history=False)
    monkeypatch.setattr(Platform, "detect", lambda: Platform.MACOS)
    lookup = Mock(return_value="synthetic Keychain login")
    if unreadable:
        lookup.side_effect = OSError("secret-bearing backend details")
    monkeypatch.setattr(macos_keychain, "get_password", lookup)
    monkeypatch.setattr(macos_keychain, "keychain_account_name", lambda: "test-user")
    manager._sync_sharing = Mock()
    with pytest.raises(SessionError) as error:
        prepare_launch(manager, record, registry, share=False, share_history=False)
    assert "secret-bearing" not in str(error.value)
    lookup.assert_called_once_with(keychain_service_name(first.directory), "test-user")
    manager._sync_sharing.assert_not_called()
