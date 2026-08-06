"""CLI-level tests for the `cswap codex|kimi ...` namespaces."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from claude_swap import cli
from claude_swap.tool_switcher import TOOLS, live_credential_path

from tests.test_tool_switcher import codex_auth, kimi_auth


def _write_live(temp_home: Path, tool: str, content: str) -> None:
    path = live_credential_path(TOOLS[tool])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture(autouse=True)
def _no_usage_network(monkeypatch):
    """`list`/`status` must never hit the network from CLI tests."""
    monkeypatch.setattr(cli, "_tool_usage_lines", lambda store: {})


class TestToolCli:
    def test_add_and_list(self, temp_home, capsys):
        _write_live(temp_home, "codex", codex_auth("one@example.com"))
        cli._tool_command("codex", ["add"])
        out = capsys.readouterr().out
        assert "Added" in out and "one@example.com" in out

        cli._tool_command("codex", ["list"])
        out = capsys.readouterr().out
        assert "Codex CLI accounts:" in out
        assert "1: one@example.com [active]" in out

    def test_list_json(self, temp_home, capsys):
        _write_live(temp_home, "kimi", kimi_auth())
        cli._tool_command("kimi", ["add", "--label", "work"])
        capsys.readouterr()
        cli._tool_command("kimi", ["list", "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["tool"] == "kimi"
        assert payload["activeAccountNumber"] == 1
        assert payload["accounts"][0]["label"] == "work"

    def test_switch_and_status(self, temp_home, capsys):
        _write_live(temp_home, "codex", codex_auth("one@example.com"))
        cli._tool_command("codex", ["add"])
        _write_live(temp_home, "codex", codex_auth("two@example.com", token="b"))
        cli._tool_command("codex", ["add"])
        capsys.readouterr()

        cli._tool_command("codex", ["switch", "1"])
        out = capsys.readouterr().out
        assert "Switched to" in out and "one@example.com" in out

        cli._tool_command("codex", ["status", "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["activeAccountNumber"] == 1
        assert payload["label"] == "one@example.com"

    def test_remove(self, temp_home, capsys):
        _write_live(temp_home, "kimi", kimi_auth())
        cli._tool_command("kimi", ["add"])
        capsys.readouterr()
        cli._tool_command("kimi", ["remove", "1"])
        assert "Removed" in capsys.readouterr().out
        cli._tool_command("kimi", ["list"])
        assert "No Kimi Code CLI accounts stored" in capsys.readouterr().out

    def test_error_exit_code(self, temp_home, capsys):
        # No login present: add must fail cleanly with exit 1.
        with pytest.raises(SystemExit) as exc:
            cli._tool_command("codex", ["add"])
        assert exc.value.code == 1
        assert "Error:" in capsys.readouterr().err

    def test_json_error_is_machine_readable(self, temp_home, capsys):
        with pytest.raises(SystemExit) as exc:
            cli._tool_command("codex", ["status", "--json"])
        assert exc.value.code == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["schemaVersion"] == 1
        assert "error" in payload

    def test_run_requires_target(self, temp_home):
        with pytest.raises(SystemExit):
            cli._tool_command("codex", ["run"])

    def test_forward_only_for_run(self, temp_home):
        with pytest.raises(SystemExit):
            cli._tool_command("codex", ["list", "--", "x"])

    def test_main_dispatches_tool_namespace(self, temp_home, monkeypatch, capsys):
        _write_live(temp_home, "codex", codex_auth("one@example.com"))
        monkeypatch.setattr("sys.argv", ["cswap", "codex", "add"])
        monkeypatch.setattr(
            "claude_swap.update_check.check_for_update", lambda *_: None
        )
        cli.main()
        assert "Added" in capsys.readouterr().out

    def test_help_mentions_tool_namespaces(self, temp_home, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["cswap", "help"])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "codex add|switch|list|status|remove|run" in out
        assert "kimi add|switch|list|status|remove|run" in out
