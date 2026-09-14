"""Managed login commands exercise the real local handoff with synthetic stores."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from claude_swap.exceptions import SessionError
from claude_swap.models import Platform
from claude_swap.switcher import ClaudeAccountSwitcher
from claude_swap.vision import VisionClient, VisionError
from claude_swap.vision_cli import ManagedProfiles, run_command
from claude_swap.vision_handoff import _write_private


@pytest.fixture
def setup(temp_home, monkeypatch):
    monkeypatch.setattr(Platform, "detect", lambda: Platform.LINUX)
    monkeypatch.setattr(
        "claude_swap.vision_handoff.profile_is_quiescent", lambda _: True
    )
    monkeypatch.setattr(
        "claude_swap.vision_cli.shutil.which", lambda _: "/synthetic/claude"
    )
    switcher = ClaudeAccountSwitcher()
    client = VisionClient("https://vision.example.invalid", "synthetic-key")
    monkeypatch.setattr("claude_swap.vision_cli.configured_client", lambda: client)
    item = {
        "account_id": "aia_00000000-0000-4000-8000-000000000002",
        "login_id": "ail_00000000-0000-4000-8000-000000000003",
        "provider": "claude",
        "email": "synthetic@example.invalid",
        "organization_id": "organization",
        "subscription": {"plan": "max"},
    }
    client.discover = Mock(return_value=[item])

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
            **{
                key: item[key]
                for key in ("account_id", "login_id", "email", "organization_id")
            },
            "expected_generation": 0,
            "generation": 1 if committed else None,
            "expires_at": "2030-01-01T00:00:00Z",
            "capabilities": ["identity", "inference"],
        }

    client.request = Mock(side_effect=request)

    def login(argv, *, env, check):
        assert argv == ["/synthetic/claude", "auth", "login"]
        assert "VISION_API_KEY" not in env and "CLAUDE_CODE_OAUTH_TOKEN" not in env
        _write_private(
            Path(env["CLAUDE_CONFIG_DIR"]) / ".credentials.json",
            json.dumps(
                {
                    "claudeAiOauth": {
                        "accessToken": "synthetic-access",
                        "refreshToken": "synthetic-refresh",
                    }
                }
            ),
        )
        return SimpleNamespace(returncode=0)

    native = Mock(side_effect=login)
    monkeypatch.setattr("claude_swap.vision_cli.subprocess.run", native)
    return switcher, client, native


def test_successful_login_uploads_by_default_and_alias_runs_verified_remote(setup):
    switcher, client, native = setup
    result = run_command(["account-login", "work"], switcher)
    assert result["state"] == "committed"
    assert switcher.resolve_account("work")[1] == "synthetic@example.invalid"
    assert not list((switcher.backup_dir / "vision-logins").rglob(".credentials.json"))
    native.assert_called_once()
    assert len(client.request.call_args_list) == 2


def test_persistent_opt_out_keeps_native_login_local(setup):
    switcher, client, _ = setup
    assert run_command(["auto-register", "off"], switcher) == {
        "auto_register": False, "url": client.url
    }
    assert ManagedProfiles(switcher.backup_dir).auto_register(client.url) is False
    assert run_command(["account-login", "work"], switcher)["state"] == "local"
    client.request.assert_not_called()
    assert list((switcher.backup_dir / "vision-logins").rglob(".credentials.json"))
    assert run_command(["upload", "work"], switcher)["state"] == "committed"


def test_failed_login_never_uploads(setup):
    switcher, client, native = setup
    native.side_effect = None
    native.return_value = SimpleNamespace(returncode=7)
    with pytest.raises(SystemExit) as result:
        run_command(["account-login", "work"], switcher)
    assert result.value.code == 7
    client.request.assert_not_called()


def test_no_op_login_never_uploads(setup):
    switcher, client, native = setup
    native.side_effect = None
    native.return_value = SimpleNamespace(returncode=0)
    assert run_command(["account-login", "work"], switcher)["state"] == "unchanged"
    client.request.assert_not_called()


def test_pending_upload_blocks_new_login_even_after_opt_out(setup):
    switcher, client, native = setup
    client.request.side_effect = VisionError("service_unavailable")
    with pytest.raises(VisionError):
        run_command(["account-login", "work"], switcher)
    run_command(["auto-register", "off"], switcher)
    with pytest.raises(SessionError, match="Recover or cancel"):
        run_command(["account-login", "work"], switcher)
    native.assert_called_once()


def test_corrupt_preferences_cannot_reenable_upload(setup):
    switcher, client, native = setup
    run_command(["auto-register", "off"], switcher)
    _write_private(switcher.backup_dir / "vision-profiles.json", "{")
    with pytest.raises(SessionError, match="refusing to reset"):
        run_command(["account-login", "work"], switcher)
    native.assert_not_called()
    client.request.assert_not_called()


def test_reused_profile_alias_follows_new_verified_login(setup):
    switcher, client, _ = setup
    first = run_command(["account-login", "work"], switcher)
    original_request = client.request.side_effect
    old_item = client.discover.return_value[0]
    new_item = {
        **old_item,
        "account_id": "aia_00000000-0000-4000-8000-000000000004",
        "login_id": "ail_00000000-0000-4000-8000-000000000005",
        "email": "another@example.invalid",
    }
    client.discover.return_value = [old_item, new_item]

    def request(*args, **kwargs):
        value = original_request(*args, **kwargs)
        return {
            **value,
            **{key: new_item[key] for key in ("account_id", "login_id", "email")},
        }

    client.request.side_effect = request
    second = run_command(["account-login", "work"], switcher)
    assert first["account_id"] != second["account_id"]
    assert switcher.resolve_account("work")[1] == "another@example.invalid"


def test_opted_out_profile_runs_native_resume_without_registry_upload(setup):
    switcher, client, native = setup
    run_command(["auto-register", "off"], switcher)
    run_command(["account-login", "work"], switcher)
    login_env = native.call_args.kwargs["env"]
    native.side_effect = None
    native.return_value = SimpleNamespace(returncode=17)
    with pytest.raises(SystemExit) as exited:
        run_command(
            ["account-run", "work", "--", "--resume", "conversation-id"], switcher
        )
    assert exited.value.code == 17
    assert native.call_args.args[0] == [
        "/synthetic/claude",
        "--resume",
        "conversation-id",
    ]
    assert native.call_args.kwargs["env"] == login_env
    assert not (Path(login_env["CLAUDE_CONFIG_DIR"]) / ".oauth_refresh.lock").exists()
    client.request.assert_not_called()


def test_pending_handoff_blocks_local_run_even_with_upload_disabled(setup):
    switcher, client, native = setup
    client.request.side_effect = VisionError("service_unavailable")
    with pytest.raises(VisionError):
        run_command(["account-login", "work"], switcher)
    run_command(["auto-register", "off"], switcher)
    with pytest.raises(SessionError, match="Recover or cancel"):
        run_command(["account-run", "work"], switcher)
    native.assert_called_once()


def test_committed_profile_cannot_restart_native_with_retired_refresh_grant(setup):
    switcher, _, native = setup
    run_command(["account-login", "work"], switcher)
    with pytest.raises(SessionError, match="central"):
        run_command(["account-run", "work"], switcher)
    native.assert_called_once()


def test_local_run_does_not_require_vision_configuration_when_opted_out(
    setup, monkeypatch
):
    switcher, _, native = setup
    run_command(["auto-register", "off"], switcher)
    run_command(["account-login", "work"], switcher)
    monkeypatch.setattr(
        "claude_swap.vision_cli.configured_client",
        Mock(side_effect=AssertionError("local launch must not load registry auth")),
    )
    native.side_effect = None
    native.return_value = SimpleNamespace(returncode=0)
    with pytest.raises(SystemExit) as exited:
        run_command(["account-run", "work"], switcher)
    assert exited.value.code == 0


def test_changed_native_login_uploads_only_after_process_exit_when_enabled(setup):
    switcher, client, native = setup
    run_command(["auto-register", "off"], switcher)
    run_command(["account-login", "work"], switcher)
    run_command(["auto-register", "on"], switcher)

    def relogin(argv, *, env, check):
        from claude_swap.locking import FileLock

        profile = Path(env["CLAUDE_CONFIG_DIR"])
        competing_upload = FileLock(profile / ".vision-auth.lock", timeout=0)
        try:
            assert not competing_upload.acquire()
        finally:
            competing_upload.release()
        client.request.assert_not_called()
        path = Path(env["CLAUDE_CONFIG_DIR"]) / ".credentials.json"
        _write_private(
            path, path.read_text().replace("synthetic-refresh", "new-refresh")
        )
        assert not (path.parent / ".oauth_refresh.lock").exists()
        return SimpleNamespace(returncode=0)

    native.side_effect = relogin
    with pytest.raises(SystemExit) as exited:
        run_command(["account-run", "work"], switcher)
    assert exited.value.code == 0
    assert len(client.request.call_args_list) == 2
    assert not list((switcher.backup_dir / "vision-logins").rglob(".credentials.json"))


def test_auto_registration_choices_are_independent_per_normalized_deployment(setup):
    switcher, client, _ = setup
    a = client.url
    b = "https://other-vision.example.invalid"
    run_command(["--url", a + "/", "auto-register", "off"], switcher)
    preferences = ManagedProfiles(switcher.backup_dir)
    assert preferences.auto_register(a) is False
    assert preferences.auto_register(b) is True
    run_command(["--url", b, "auto-register", "off"], switcher)
    run_command(["--url", a, "auto-register", "on"], switcher)
    reopened = ManagedProfiles(switcher.backup_dir)
    assert reopened.auto_register(a) is True
    assert reopened.auto_register(b) is False


@pytest.mark.parametrize("legacy", [False, True])
def test_legacy_upload_preference_remains_fallback_after_origin_override(setup, legacy):
    switcher, client, _ = setup
    profiles = ManagedProfiles(switcher.backup_dir)
    profiles.root.mkdir(parents=True, exist_ok=True, mode=0o700)
    _write_private(profiles.path, json.dumps({
        "version": 1, "auto_register": legacy, "profiles": {},
    }))
    other = "https://other-vision.example.invalid"
    assert profiles.auto_register(client.url) is legacy
    assert profiles.auto_register(other) is legacy
    profiles.set_auto_register(not legacy, client.url)
    reopened = ManagedProfiles(switcher.backup_dir)
    assert reopened.read()["version"] == 2
    assert reopened.auto_register(client.url) is not legacy
    assert reopened.auto_register(other) is legacy


def test_login_uses_current_origin_preference_and_discloses_before_native(setup, capsys):
    switcher, client, native = setup
    profiles = ManagedProfiles(switcher.backup_dir)
    profiles.set_auto_register(False, client.url)
    original_native = native.side_effect

    def launch(*args, **kwargs):
        disclosure = capsys.readouterr().err
        assert "disabled" in disclosure and client.url in disclosure
        assert f"cswap vision --url {client.url} auto-register off" in disclosure
        return original_native(*args, **kwargs)

    native.side_effect = launch
    assert run_command(["account-login", "work"], switcher)["state"] == "local"
    client.request.assert_not_called()
    assert profiles.auto_register("https://other-vision.example.invalid") is True


def test_vision_key_setup_discloses_enabled_destination_and_opt_out(setup, monkeypatch, capsys):
    switcher, client, _ = setup
    monkeypatch.setenv("VISION_API_KEY", "synthetic-key")
    assert run_command(["login"], switcher)["state"] == "configured"
    text = capsys.readouterr().err
    assert "enabled" in text and client.url in text
    assert f"cswap vision --url {client.url} auto-register off" in text
    client.request.assert_not_called()


def test_malformed_origin_preference_cannot_reset_existing_opt_out(setup):
    switcher, client, _ = setup
    profiles = ManagedProfiles(switcher.backup_dir)
    profiles.root.mkdir(parents=True, exist_ok=True, mode=0o700)
    _write_private(profiles.path, json.dumps({
        "version": 2, "auto_register": False, "profiles": {}, "bindings": {},
        "auto_register_by_origin": {client.url: "true"},
    }))
    with pytest.raises(SessionError, match="refusing to reset"):
        profiles.auto_register(client.url)


def test_unconfigured_provider_login_discloses_local_only(setup, monkeypatch, capsys):
    switcher, client, _ = setup
    monkeypatch.setattr("claude_swap.vision_cli.configured_client", lambda: None)
    assert run_command(["account-login", "work"], switcher)["state"] == "local"
    disclosure = capsys.readouterr().err
    assert "Vision is not configured; this login stays local" in disclosure
    assert "auto-register off" in disclosure
    client.request.assert_not_called()
