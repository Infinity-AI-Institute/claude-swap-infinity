"""A Vision key and central accounts suffice for an ordinary native launch."""

from unittest.mock import Mock

import pytest

from claude_swap.exceptions import ClaudeSwitchError, SessionError
from claude_swap.session import SessionManager
from claude_swap.switcher import ClaudeAccountSwitcher
from claude_swap.vision import VisionClient, VisionError
from tests.test_vision_registry import item


@pytest.fixture
def setup(temp_home, monkeypatch):
    switcher = ClaudeAccountSwitcher()
    manager = SessionManager(switcher)
    client = VisionClient("https://vision.example.invalid", "synthetic-key")
    client.discover = Mock(return_value=[item(1), item(2)])
    monkeypatch.setattr("claude_swap.vision.configured_client", lambda: client)
    manager.run = Mock(side_effect=SystemExit(0))
    manager._exec = Mock(side_effect=AssertionError("uncredentialed native launch"))
    switcher._get_current_account = Mock(
        side_effect=AssertionError("local login lookup")
    )
    return manager, client


def test_empty_home_launch_discovers_account_without_login_or_explicit_selection(setup):
    manager, client = setup
    arguments = ["--resume", "native-conversation", "--model", "sonnet"]
    with pytest.raises(SystemExit) as exited:
        manager.exec_default(arguments)
    assert exited.value.code == 0
    client.discover.assert_called_once()
    manager.run.assert_called_once_with("1", arguments, share=True, share_history=False)
    assert not list(manager.switcher.credentials_dir.glob("*"))


def test_disabled_account_is_skipped(setup):
    manager, client = setup
    from claude_swap.vision_registry import RegistryPool

    RegistryPool(manager.switcher, client).sync()
    roster = manager.switcher._get_sequence_data()
    roster["accounts"]["1"]["disabled"] = True
    manager.switcher._write_json(manager.switcher.sequence_file, roster)
    with pytest.raises(SystemExit):
        manager.exec_default([])
    manager.run.assert_called_once_with("2", [], share=True, share_history=False)


def test_empty_authorized_pool_does_not_fall_back_to_provider_login(setup):
    manager, client = setup
    client.discover.return_value = []
    with pytest.raises(SessionError, match="authorized through Vision"):
        manager.exec_default([])
    manager.run.assert_not_called()
    manager._exec.assert_not_called()


def test_registry_failure_does_not_launch_an_uncredentialed_native_process(setup):
    manager, client = setup
    client.discover.side_effect = VisionError("not_permitted", status=403)
    with pytest.raises(VisionError):
        manager.exec_default([])
    manager.run.assert_not_called()
    manager._exec.assert_not_called()


def test_automatic_selection_preserves_sharing_options(setup):
    manager, _ = setup
    with pytest.raises(SystemExit):
        manager.exec_default(["--continue"], share=False, share_history=True)
    manager.run.assert_called_once_with(
        "1", ["--continue"], share=False, share_history=True
    )


def test_empty_authorized_pool_names_who_can_grant_access(setup):
    manager, client = setup
    client.discover.return_value = []
    with pytest.raises(SessionError, match="Vision admin"):
        manager.exec_default([])


def test_explicit_launch_explains_a_refused_key_before_the_lookup_fails(
    temp_home, monkeypatch, capsys
):
    monkeypatch.delenv("VISION_API_KEY", raising=False)
    switcher = ClaudeAccountSwitcher()
    client = VisionClient("https://vision.example.invalid", "synthetic-key")
    client.discover = Mock(side_effect=VisionError("not_permitted", status=403))
    monkeypatch.setattr(
        "claude_swap.switcher.configured_vision_client", lambda: client
    )
    monkeypatch.setattr("claude_swap.session.shutil.which", lambda _: "/synthetic/claude")
    manager = SessionManager(switcher)
    manager._exec = Mock(side_effect=AssertionError("uncredentialed native launch"))
    with pytest.raises(ClaudeSwitchError, match="does not exist"):
        manager.run("1", [])
    assert "Vision admin" in capsys.readouterr().err
