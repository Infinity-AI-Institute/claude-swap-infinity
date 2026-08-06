"""Tests for the non-Claude tool account store (Codex / Kimi)."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pytest

from claude_swap.exceptions import (
    AccountNotFoundError,
    CredentialReadError,
    SessionError,
)
from claude_swap.paths import get_backup_root
from claude_swap.tool_switcher import (
    TOOLS,
    ToolAccountStore,
    live_credential_path,
)


def _jwt(payload: dict) -> str:
    """Build an unsigned JWT with the given payload (identity claims only)."""

    def seg(data: dict) -> str:
        raw = json.dumps(data).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    return f"{seg({'alg': 'none'})}.{seg(payload)}."


def codex_auth(email: str, account_id: str = "acct-1", token: str = "tok") -> str:
    id_token = _jwt(
        {
            "https://api.openai.com/profile": {"email": email},
            "https://api.openai.com/auth": {"chatgpt_account_id": account_id},
        }
    )
    return json.dumps(
        {
            "auth_mode": "chatgpt",
            "tokens": {
                "id_token": id_token,
                "access_token": f"access-{token}",
                "refresh_token": f"refresh-{token}",
                "account_id": account_id,
            },
            "last_refresh": "2026-01-01T00:00:00Z",
        }
    )


def kimi_auth(token: str = "tok") -> str:
    return json.dumps(
        {
            "access_token": f"kimi-access-{token}",
            "refresh_token": f"kimi-refresh-{token}",
            "expires_at": 9999999999,
            "scope": "kimi-for-coding",
            "token_type": "Bearer",
        }
    )


def write_live(temp_home: Path, tool: str, content: str) -> Path:
    path = live_credential_path(TOOLS[tool])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


class TestAdd:
    def test_add_snapshots_live_login_and_extracts_email(self, temp_home):
        live = write_live(temp_home, "codex", codex_auth("one@example.com"))
        store = ToolAccountStore("codex")
        slot, label = store.add()
        assert slot == 1
        assert label == "one@example.com"
        assert store._slot_path(1).read_text() == live.read_text()
        assert store._load()["activeSlot"] == 1

    def test_add_same_login_refreshes_slot_instead_of_duplicating(self, temp_home):
        write_live(temp_home, "codex", codex_auth("one@example.com"))
        store = ToolAccountStore("codex")
        store.add()
        slot, _ = store.add()
        assert slot == 1
        assert len(store._load()["accounts"]) == 1

    def test_add_second_login_gets_next_slot(self, temp_home):
        write_live(temp_home, "codex", codex_auth("one@example.com"))
        store = ToolAccountStore("codex")
        store.add()
        write_live(temp_home, "codex", codex_auth("two@example.com", token="b"))
        slot, label = store.add()
        assert (slot, label) == (2, "two@example.com")

    def test_add_explicit_slot_and_label(self, temp_home):
        write_live(temp_home, "kimi", kimi_auth())
        store = ToolAccountStore("kimi")
        slot, label = store.add(slot=3, label="work")
        assert (slot, label) == (3, "work")
        assert store._slot_path(3).exists()

    def test_add_without_login_raises(self, temp_home):
        store = ToolAccountStore("codex")
        with pytest.raises(CredentialReadError):
            store.add()

    def test_kimi_label_falls_back_to_generic(self, temp_home):
        write_live(temp_home, "kimi", kimi_auth())
        store = ToolAccountStore("kimi")
        slot, label = store.add()
        assert label == f"kimi-account-{slot}"

    def test_backup_root_is_per_tool(self, temp_home):
        assert ToolAccountStore("codex").root == get_backup_root() / "codex"
        assert ToolAccountStore("kimi").root == get_backup_root() / "kimi"


class TestSwitch:
    def _two_accounts(self, temp_home) -> ToolAccountStore:
        write_live(temp_home, "codex", codex_auth("one@example.com"))
        store = ToolAccountStore("codex")
        store.add()
        write_live(temp_home, "codex", codex_auth("two@example.com", token="b"))
        store.add()
        return store

    def test_bare_switch_rotates(self, temp_home):
        store = self._two_accounts(temp_home)
        slot, label = store.switch()
        assert (slot, label) == (1, "one@example.com")
        live = live_credential_path(TOOLS["codex"]).read_text()
        assert json.loads(live)["tokens"]["refresh_token"] == "refresh-tok"
        slot, _ = store.switch()
        assert slot == 2

    def test_switch_to_target_by_number_and_label(self, temp_home):
        store = self._two_accounts(temp_home)
        slot, _ = store.switch("2")
        assert slot == 2
        slot, _ = store.switch("one@example.com")
        assert slot == 1

    def test_switch_unknown_target_raises(self, temp_home):
        store = self._two_accounts(temp_home)
        with pytest.raises(AccountNotFoundError):
            store.switch("9")
        with pytest.raises(AccountNotFoundError):
            store.switch("nobody@example.com")

    def test_switch_with_no_accounts_raises(self, temp_home):
        store = ToolAccountStore("codex")
        with pytest.raises(AccountNotFoundError):
            store.switch()

    def test_switch_snapshots_unknown_live_login_first(self, temp_home):
        # Slot 1 stored, then the user logs in with a different, never-added
        # account. Switching must not silently discard that login.
        write_live(temp_home, "codex", codex_auth("one@example.com"))
        store = ToolAccountStore("codex")
        store.add()
        live = write_live(temp_home, "codex", codex_auth("stray@example.com", token="z"))
        slot, _ = store.switch("1")
        assert slot == 1
        data = store._load()
        assert len(data["accounts"]) == 2  # stray was snapshotted to slot 2
        assert "one@example.com" not in live.read_text()  # email is JWT-encoded
        assert json.loads(live.read_text())["tokens"]["refresh_token"] == "refresh-tok"

    def test_switch_does_not_touch_other_tools(self, temp_home):
        write_live(temp_home, "codex", codex_auth("one@example.com"))
        kimi_live = write_live(temp_home, "kimi", kimi_auth())
        store = ToolAccountStore("codex")
        store.add()
        before = kimi_live.read_text()
        store.switch("1")
        assert kimi_live.read_text() == before


class TestListStatusRemove:
    def test_list_marks_active(self, temp_home):
        write_live(temp_home, "codex", codex_auth("one@example.com"))
        store = ToolAccountStore("codex")
        store.add()
        write_live(temp_home, "codex", codex_auth("two@example.com", token="b"))
        store.add()
        live_slot, rows = store.list_accounts()
        assert live_slot == 2
        assert [r.number for r in rows] == [1, 2]
        assert [r.active for r in rows] == [False, True]

    def test_status_unmanaged_login(self, temp_home):
        write_live(temp_home, "codex", codex_auth("stray@example.com"))
        store = ToolAccountStore("codex")
        slot, note = store.status()
        assert slot is None
        assert "unmanaged" in note

    def test_status_no_login_raises(self, temp_home):
        store = ToolAccountStore("codex")
        with pytest.raises(CredentialReadError):
            store.status()

    def test_remove_deletes_slot_but_not_live_login(self, temp_home):
        live = write_live(temp_home, "codex", codex_auth("one@example.com"))
        store = ToolAccountStore("codex")
        store.add()
        live_content = live.read_text()
        slot, label = store.remove("1")
        assert (slot, label) == (1, "one@example.com")
        assert not store._slot_path(1).exists()
        assert store._load()["accounts"] == {}
        assert live.read_text() == live_content

    def test_remove_unknown_raises(self, temp_home):
        store = ToolAccountStore("codex")
        with pytest.raises(AccountNotFoundError):
            store.remove("1")


class TestRunSession:
    def _store_with_account(self, temp_home, tool: str) -> ToolAccountStore:
        if tool == "codex":
            write_live(temp_home, tool, codex_auth("one@example.com"))
        else:
            write_live(temp_home, tool, kimi_auth())
        store = ToolAccountStore(tool)
        store.add()
        return store

    def test_missing_binary_raises(self, temp_home, monkeypatch):
        store = self._store_with_account(temp_home, "codex")
        monkeypatch.setattr("shutil.which", lambda _: None)
        with pytest.raises(SessionError):
            store.run_session("1", [])

    @pytest.mark.parametrize(
        "tool,env_var,cred_rel",
        [
            ("codex", "CODEX_HOME", ("auth.json",)),
            ("kimi", "KIMI_CODE_HOME", ("credentials", "kimi-code.json")),
        ],
    )
    def test_run_sets_home_env_and_copies_credential(
        self, temp_home, monkeypatch, tool, env_var, cred_rel
    ):
        store = self._store_with_account(temp_home, tool)
        monkeypatch.setattr("shutil.which", lambda _: f"/fake/{tool}")

        captured = {}

        if sys.platform == "win32":

            class _Result:
                returncode = 3

            def fake_run(argv, env=None, **kw):
                captured["argv"] = argv
                captured["env"] = env
                return _Result()

            monkeypatch.setattr("subprocess.run", fake_run)
            with pytest.raises(SystemExit) as exc:
                store.run_session("1", ["--help"])
            assert exc.value.code == 3
        else:

            def fake_execvpe(binary, argv, env):
                captured["argv"] = argv
                captured["env"] = env
                raise SystemExit(0)

            monkeypatch.setattr("os.execvpe", fake_execvpe)
            with pytest.raises(SystemExit):
                store.run_session("1", ["--help"])

        env = captured["env"]
        session_home = Path(env[env_var])
        assert session_home == store.sessions_dir / "1"
        session_cred = session_home.joinpath(*cred_rel)
        assert session_cred.read_text() == store._slot_path(1).read_text()
        # The live credential path was not redirected.
        assert env[env_var] != str(live_credential_path(TOOLS[tool]).parent)
        assert captured["argv"][1:] == ["--help"]
