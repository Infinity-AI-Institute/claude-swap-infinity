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
    assert run_command(["auto-register", "off"], switcher) == {"auto_register": False}
    assert ManagedProfiles(switcher.backup_dir).read()["auto_register"] is False
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
