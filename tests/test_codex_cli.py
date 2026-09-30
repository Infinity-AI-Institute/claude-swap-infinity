"""CLI-level tests for the `cswap codex ...` namespace."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from claude_swap import cli
from claude_swap.codex_accounts import default_codex_home

from tests.test_codex_accounts import codex_auth


def _write_live(content: str) -> Path:
    path = default_codex_home() / "auth.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _no_usage_network(monkeypatch):
    """`list` must never hit the network from CLI tests."""
    monkeypatch.setattr(cli, "_codex_usage_lines", lambda store: {})


class TestCodexCli:
    def test_add_and_list(self, temp_home, capsys):
        _write_live(codex_auth("one@example.com"))
        cli._codex_command(["add"])
        out = capsys.readouterr().out
        assert "Added" in out and "one@example.com" in out

        cli._codex_command(["list"])
        out = capsys.readouterr().out
        assert "Codex CLI accounts:" in out
        assert "1: one@example.com [active]" in out

    def test_list_json(self, temp_home, capsys):
        _write_live(codex_auth("one@example.com"))
        cli._codex_command(["add", "--label", "work"])
        capsys.readouterr()
        cli._codex_command(["list", "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["tool"] == "codex"
        assert payload["activeAccountNumber"] == 1
        assert payload["accounts"][0]["label"] == "work"

    def test_switch_and_status(self, temp_home, capsys):
        _write_live(codex_auth("one@example.com"))
        cli._codex_command(["add"])
        _write_live(codex_auth("two@example.com", token="b"))
        cli._codex_command(["add"])
        capsys.readouterr()

        cli._codex_command(["switch", "1"])
        out = capsys.readouterr().out
        assert "Switched to" in out and "one@example.com" in out
        # The no-hot-reload warning is part of the interface: without it a
        # user reasonably assumes the live TUI rotated.
        assert "caches auth in memory" in out

        cli._codex_command(["status", "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["activeAccountNumber"] == 1
        assert payload["label"] == "one@example.com"

    def test_remove(self, temp_home, capsys):
        _write_live(codex_auth("one@example.com"))
        cli._codex_command(["add"])
        capsys.readouterr()
        cli._codex_command(["remove", "1"])
        assert "Removed" in capsys.readouterr().out
        cli._codex_command(["list"])
        assert "No Codex CLI accounts stored" in capsys.readouterr().out

    def test_error_exit_code(self, temp_home, capsys):
        # No login present: add must fail cleanly with exit 1.
        with pytest.raises(SystemExit) as exc:
            cli._codex_command(["add"])
        assert exc.value.code == 1
        assert "Error:" in capsys.readouterr().err

    def test_json_error_is_machine_readable(self, temp_home, capsys):
        with pytest.raises(SystemExit) as exc:
            cli._codex_command(["status", "--json"])
        assert exc.value.code == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["schemaVersion"] == 1
        assert "error" in payload

    def test_remove_requires_target(self, temp_home):
        with pytest.raises(SystemExit):
            cli._codex_command(["remove"])

    def test_handoff_flags_only_for_handoff(self, temp_home):
        with pytest.raises(SystemExit):
            cli._codex_command(["list", "--if-needed"])
        with pytest.raises(SystemExit):
            cli._codex_command(["list", "--dry-run"])

    def test_handoff_takes_no_target(self, temp_home):
        with pytest.raises(SystemExit):
            cli._codex_command(["handoff", "2"])

    def test_codex_home_setting_redirects_the_store(self, temp_home, capsys):
        from claude_swap.paths import get_backup_root
        from claude_swap.settings import set_setting

        other_home = temp_home / "custom-codex"
        (other_home / "auth.json").parent.mkdir(parents=True)
        (other_home / "auth.json").write_text(codex_auth("cfg@example.com"))
        set_setting(get_backup_root(), "codex.home", str(other_home))
        cli._codex_command(["add"])
        assert "cfg@example.com" in capsys.readouterr().out

    def test_main_dispatches_codex_namespace(self, temp_home, monkeypatch, capsys):
        _write_live(codex_auth("one@example.com"))
        monkeypatch.setattr("sys.argv", ["cswap", "codex", "add"])
        monkeypatch.setattr(
            "claude_swap.update_check.check_for_update", lambda *_: None
        )
        cli.main()
        assert "Added" in capsys.readouterr().out

    def test_help_mentions_codex_namespace(self, temp_home, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["cswap", "help"])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "codex add|switch|list|status|remove" in out
        assert "codex handoff|watch" in out


class TestCodexCliHandoff:
    def test_handoff_without_tmux_target_refuses(self, temp_home, capsys):
        _write_live(codex_auth("one@example.com"))
        cli._codex_command(["add"])
        capsys.readouterr()
        with pytest.raises(SystemExit) as exc:
            cli._codex_command(["handoff"])
        assert exc.value.code == 1
        assert "codex.tmux_target" in capsys.readouterr().err

    def _configured(self, capsys) -> None:
        from claude_swap.paths import get_backup_root
        from claude_swap.settings import set_setting

        _write_live(codex_auth("one@example.com"))
        cli._codex_command(["add"])
        set_setting(get_backup_root(), "codex.tmux_target", "agent:0.0")
        capsys.readouterr()

    def test_handoff_blocked_exits_3(self, temp_home, monkeypatch, capsys):
        from claude_swap.codex_handoff import CodexHandoff, HandoffBlocked

        self._configured(capsys)

        def blocked(self, *, if_needed=False, dry_run=False):
            raise HandoffBlocked("no codex slot with positive margin")

        monkeypatch.setattr(CodexHandoff, "execute", blocked)
        with pytest.raises(SystemExit) as exc:
            cli._codex_command(["handoff"])
        assert exc.value.code == 3
        assert "positive margin" in capsys.readouterr().err

    def test_handoff_error_exits_1(self, temp_home, monkeypatch, capsys):
        from claude_swap.codex_handoff import CodexHandoff, HandoffError

        self._configured(capsys)

        def failing(self, *, if_needed=False, dry_run=False):
            raise HandoffError("tmux target 'agent:0.0' does not exist")

        monkeypatch.setattr(CodexHandoff, "execute", failing)
        with pytest.raises(SystemExit) as exc:
            cli._codex_command(["handoff"])
        assert exc.value.code == 1

    def test_handoff_success_exits_0_and_forwards_flags(
        self, temp_home, monkeypatch, capsys
    ):
        from claude_swap.codex_handoff import CodexHandoff

        self._configured(capsys)
        seen: list[dict] = []

        def succeeding(self, *, if_needed=False, dry_run=False):
            seen.append({"if_needed": if_needed, "dry_run": dry_run})
            return 0

        monkeypatch.setattr(CodexHandoff, "execute", succeeding)
        with pytest.raises(SystemExit) as exc:
            cli._codex_command(["handoff", "--if-needed", "--dry-run"])
        assert exc.value.code == 0
        assert seen == [{"if_needed": True, "dry_run": True}]

    def test_watch_runs_the_loop(self, temp_home, monkeypatch, capsys):
        from claude_swap import codex_handoff as handoff_module

        self._configured(capsys)
        seen: list[object] = []

        def fake_run_watch(handoff, **kwargs):
            seen.append(handoff)
            return 0

        # cli imports run_watch from the module at call time.
        monkeypatch.setattr(handoff_module, "run_watch", fake_run_watch)
        with pytest.raises(SystemExit) as exc:
            cli._codex_command(["watch"])
        assert exc.value.code == 0
        assert len(seen) == 1
        assert "watch running" in capsys.readouterr().out
