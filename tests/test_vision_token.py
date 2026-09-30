"""Shared-key startup contract; all key material is synthetic."""

import io
import json
import os
from pathlib import Path

import pytest

from claude_swap.exceptions import SessionError
from claude_swap.vision import configured_client
from claude_swap.vision_token import save_token, saved_token_client, setup_command, token_path

KEY = "vsk_" + "a" * 40


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    monkeypatch.delenv("VISION_API_KEY", raising=False)
    monkeypatch.delenv("VISION_API_URL", raising=False)


def test_save_and_resolve_without_environment_or_native_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    path = save_token(KEY)
    assert path == tmp_path / "vision" / "credentials.json"
    assert json.loads(path.read_text()) == {
        "version": 1, "url": "https://vision.infinity.inc", "api_key": KEY,
    }
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600
        assert path.parent.stat().st_mode & 0o777 == 0o700
    assert configured_client().api_key == KEY
    assert not (Path.home() / ".claude-swap-backup").exists()


def test_shared_key_precedes_legacy_and_environment_precedes_shared(monkeypatch):
    save_token(KEY)
    monkeypatch.setattr("claude_swap.vision_state.VisionState.read", lambda *a: pytest.fail("legacy key was read"))
    assert configured_client().api_key == KEY
    monkeypatch.setenv("VISION_API_KEY", "environment-key")
    token_path().write_text("malformed")
    assert configured_client().api_key == "environment-key"


def test_saved_origin_cannot_be_redirected(monkeypatch):
    monkeypatch.setenv("VISION_API_URL", "https://vision.example.invalid/")
    save_token(KEY)
    assert saved_token_client().url == "https://vision.example.invalid"
    monkeypatch.setenv("VISION_API_URL", "https://other.example.invalid")
    with pytest.raises(SessionError, match="another origin"):
        configured_client()


@pytest.mark.parametrize("payload", ["{", "[]", '{"version":true}', '{"version":1,"url":"https://vision.infinity.inc","api_key":"bad"}'])
def test_invalid_shared_file_fails_without_echo(payload):
    save_token(KEY)
    token_path().write_text(payload)
    with pytest.raises(SessionError, match="needs repair") as error:
        configured_client()
    assert payload not in str(error.value)


def test_private_prompt_saves_and_never_prints_token(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("getpass.getpass", lambda prompt: KEY)
    setup_command([])
    assert configured_client().api_key == KEY
    assert KEY not in capsys.readouterr().out


def test_noninteractive_prompt_does_not_read_or_save(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(SystemExit):
        setup_command([])
    assert not token_path().exists()


@pytest.mark.parametrize("arguments", [["--set-vision-token", KEY], ["--set-vision-token=" + KEY]])
def test_cli_flag_exits_before_switcher(monkeypatch, capsys, arguments):
    from claude_swap.cli import main
    monkeypatch.setattr("sys.argv", ["cswap", *arguments])
    monkeypatch.setattr("claude_swap.switcher.ClaudeAccountSwitcher", lambda *a: pytest.fail("native switcher constructed"))
    main()
    assert configured_client().api_key == KEY
    assert KEY not in capsys.readouterr().out


def test_invalid_token_preserves_previous_secret(capsys):
    save_token(KEY)
    with pytest.raises(SystemExit):
        setup_command(["secret-invalid"])
    assert configured_client().api_key == KEY
    assert "secret-invalid" not in capsys.readouterr().err


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions and symlinks")
def test_rejects_public_or_symlink_storage(tmp_path):
    save_token(KEY)
    path = token_path()
    path.chmod(0o644)
    with pytest.raises(SessionError):
        saved_token_client()
    path.unlink()
    target = tmp_path / "target"
    target.write_text("unchanged")
    path.symlink_to(target)
    with pytest.raises(SessionError):
        save_token(KEY)
    assert target.read_text() == "unchanged"
    path.unlink()
    path.parent.chmod(0o755)
    with pytest.raises(SessionError):
        save_token(KEY)


def test_login_reports_shared_configuration_instead_of_starting_browser(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from claude_swap.vision_cli import run_command
    save_token(KEY)
    monkeypatch.setattr("claude_swap.vision_cli.disclose_registration", lambda *a: None)
    monkeypatch.setattr("claude_swap.vision_signin.VisionSignIn.begin", lambda *a: pytest.fail("browser started"))
    assert run_command(["login"], SimpleNamespace(backup_dir=tmp_path)) == {
        "state": "configured", "source": "shared_config", "url": "https://vision.infinity.inc",
    }


def test_prompt_refuses_echo_fallback(monkeypatch, capsys):
    import getpass
    import warnings

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    def unsafe_prompt(prompt):
        warnings.warn("Cannot control echo", getpass.GetPassWarning)
        pytest.fail("Echo fallback was reached")

    monkeypatch.setattr("getpass.getpass", unsafe_prompt)
    with pytest.raises(SystemExit):
        setup_command([])
    assert not token_path().exists()


def test_native_passthrough_is_not_a_setup_command(monkeypatch):
    from claude_swap.cli import main
    received = []
    monkeypatch.setattr("sys.argv", ["cswap", "run", "--", "--set-vision-token", KEY])
    monkeypatch.setattr("claude_swap.cli._run_command", lambda args: received.extend(args))
    main()
    assert received == ["--", "--set-vision-token", KEY]
    assert not token_path().exists()


def test_status_reports_saved_key_before_pending_browser_flow(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from claude_swap.vision_cli import run_command

    save_token(KEY)
    monkeypatch.setattr("claude_swap.vision_signin.VisionSignIn.poll", lambda *a: pytest.fail("old flow polled"))
    result = run_command(["status"], SimpleNamespace(backup_dir=tmp_path))
    assert result["state"] == "configured"
    assert result["source"] == "shared_config"


class _TerminalInput(io.StringIO):
    """A terminal on stdin: typed input would be echoed back to the screen."""

    def isatty(self):
        return True


@pytest.mark.parametrize("arguments", [["--set-vision-token", "-"], ["--set-vision-token=-"]])
@pytest.mark.parametrize("piped", [KEY + "\n", KEY + "\r\n", KEY, KEY + "\nsecond line is not read\n"])
def test_dash_reads_the_key_from_the_first_line_of_stdin(monkeypatch, capsys, arguments, piped):
    from claude_swap.cli import main
    monkeypatch.setattr("sys.stdin", io.StringIO(piped))
    monkeypatch.setattr("sys.argv", ["cswap", *arguments])
    monkeypatch.setattr("claude_swap.switcher.ClaudeAccountSwitcher", lambda *a: pytest.fail("native switcher constructed"))
    monkeypatch.setattr("getpass.getpass", lambda prompt: pytest.fail("prompted instead of reading stdin"))
    main()
    assert configured_client().api_key == KEY
    output = capsys.readouterr()
    assert KEY not in output.out + output.err


def test_dash_with_empty_stdin_does_not_save(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    with pytest.raises(SystemExit) as exit_info:
        setup_command(["-"])
    assert exit_info.value.code == 1
    assert "No Vision API key on standard input" in capsys.readouterr().err
    assert not token_path().exists()


def test_dash_with_invalid_line_keeps_previous_key_and_does_not_echo(monkeypatch, capsys):
    save_token(KEY)
    monkeypatch.setattr("sys.stdin", io.StringIO("secret-invalid\n"))
    with pytest.raises(SystemExit):
        setup_command(["-"])
    assert configured_client().api_key == KEY
    output = capsys.readouterr()
    assert "secret-invalid" not in output.out + output.err
    assert "Vision Settings" in output.err


def test_dash_refuses_a_terminal_so_the_key_is_never_echoed(monkeypatch, capsys):
    terminal = _TerminalInput(KEY + "\n")
    monkeypatch.setattr("sys.stdin", terminal)
    with pytest.raises(SystemExit):
        setup_command(["-"])
    assert terminal.tell() == 0, "a terminal line was read"
    assert "hidden prompt" in capsys.readouterr().err
    assert not token_path().exists()


def test_noninteractive_prompt_points_to_the_stdin_form(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(SystemExit):
        setup_command([])
    assert "--set-vision-token -" in capsys.readouterr().err


def test_dash_through_a_real_pipe_keeps_the_key_out_of_argv(tmp_path):
    import subprocess
    import sys

    environment = {
        **os.environ,
        "HOME": str(tmp_path),
        "USERPROFILE": str(tmp_path),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
    }
    environment.pop("VISION_API_KEY", None)
    environment.pop("VISION_API_URL", None)
    arguments = [sys.executable, "-c", "from claude_swap.cli import main; main()", "--set-vision-token", "-"]
    completed = subprocess.run(arguments, input=KEY + "\n", env=environment, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    assert KEY not in completed.stdout + completed.stderr + " ".join(arguments)
    saved = json.loads((tmp_path / "config" / "vision" / "credentials.json").read_text())
    assert saved["api_key"] == KEY
