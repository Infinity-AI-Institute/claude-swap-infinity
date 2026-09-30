"""Tests for the Codex account slot store (codex_accounts.py)."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from claude_swap.codex_accounts import (
    CodexAccountStore,
    default_codex_home,
    resolve_codex_home,
)
from claude_swap.exceptions import (
    AccountNotFoundError,
    CredentialReadError,
    ValidationError,
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


def make_store(tmp_path: Path) -> CodexAccountStore:
    """A store with an explicit backup root and codex home under tmp_path."""
    home = tmp_path / "codex-home"
    home.mkdir(parents=True, exist_ok=True)
    return CodexAccountStore(backup_root=tmp_path / "backup", home=home)


def write_live(store: CodexAccountStore, content: str) -> Path:
    path = store.live_credential_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


class TestHomeResolution:
    def test_default_home_is_dot_codex(self, temp_home):
        assert default_codex_home() == temp_home / ".codex"

    def test_env_codex_home_wins_over_default(self, temp_home, monkeypatch):
        monkeypatch.setenv("CODEX_HOME", str(temp_home / "elsewhere"))
        assert default_codex_home() == temp_home / "elsewhere"

    def test_configured_home_wins_over_env(self, temp_home, monkeypatch):
        monkeypatch.setenv("CODEX_HOME", str(temp_home / "elsewhere"))
        assert resolve_codex_home(str(temp_home / "cfg")) == temp_home / "cfg"

    def test_configured_home_expands_tilde(self, temp_home):
        assert resolve_codex_home("~/mycodex") == temp_home / "mycodex"

    def test_unconfigured_falls_back_to_default(self, temp_home):
        assert resolve_codex_home(None) == temp_home / ".codex"


class TestAdd:
    def test_add_snapshots_live_login_and_extracts_email(self, tmp_path):
        store = make_store(tmp_path)
        live = write_live(store, codex_auth("one@example.com"))
        slot, label = store.add()
        assert slot == 1
        assert label == "one@example.com"
        assert store._slot_path(1).read_text() == live.read_text()
        assert store._load()["activeSlot"] == 1

    def test_add_same_login_refreshes_slot_instead_of_duplicating(self, tmp_path):
        store = make_store(tmp_path)
        write_live(store, codex_auth("one@example.com"))
        store.add()
        slot, _ = store.add()
        assert slot == 1
        assert len(store._load()["accounts"]) == 1

    def test_add_second_login_gets_next_slot(self, tmp_path):
        store = make_store(tmp_path)
        write_live(store, codex_auth("one@example.com"))
        store.add()
        write_live(store, codex_auth("two@example.com", token="b"))
        slot, label = store.add()
        assert (slot, label) == (2, "two@example.com")

    def test_add_explicit_slot_and_label(self, tmp_path):
        store = make_store(tmp_path)
        write_live(store, codex_auth("one@example.com"))
        slot, label = store.add(slot=3, label="work")
        assert (slot, label) == (3, "work")
        assert store._slot_path(3).exists()

    def test_add_rejects_non_positive_slot(self, tmp_path):
        store = make_store(tmp_path)
        write_live(store, codex_auth("one@example.com"))
        with pytest.raises(ValidationError):
            store.add(slot=0)

    def test_add_without_login_raises(self, tmp_path):
        store = make_store(tmp_path)
        with pytest.raises(CredentialReadError):
            store.add()

    def test_unparseable_credential_gets_generic_label(self, tmp_path):
        store = make_store(tmp_path)
        write_live(store, "not json")
        slot, label = store.add()
        assert label == f"codex-account-{slot}"

    def test_store_lives_in_codex_namespace(self, tmp_path):
        store = make_store(tmp_path)
        assert store.root == tmp_path / "backup" / "codex"


class TestSwitch:
    def _two_accounts(self, tmp_path) -> CodexAccountStore:
        store = make_store(tmp_path)
        write_live(store, codex_auth("one@example.com"))
        store.add()
        write_live(store, codex_auth("two@example.com", token="b"))
        store.add()
        return store

    def test_bare_switch_rotates(self, tmp_path):
        store = self._two_accounts(tmp_path)
        slot, label = store.switch()
        assert (slot, label) == (1, "one@example.com")
        live = store.live_credential_path.read_text()
        assert json.loads(live)["tokens"]["refresh_token"] == "refresh-tok"
        slot, _ = store.switch()
        assert slot == 2

    def test_switch_to_target_by_number_and_label(self, tmp_path):
        store = self._two_accounts(tmp_path)
        slot, _ = store.switch("2")
        assert slot == 2
        slot, _ = store.switch("one@example.com")
        assert slot == 1

    def test_switch_unknown_target_raises(self, tmp_path):
        store = self._two_accounts(tmp_path)
        with pytest.raises(AccountNotFoundError):
            store.switch("9")
        with pytest.raises(AccountNotFoundError):
            store.switch("nobody@example.com")

    def test_switch_with_no_accounts_raises(self, tmp_path):
        store = make_store(tmp_path)
        with pytest.raises(AccountNotFoundError):
            store.switch()

    def test_switch_snapshots_stray_live_login_first(self, tmp_path):
        # Slot 1 stored, then the user logs in by hand with a different,
        # never-added account. Switching must not silently discard that
        # login — and must NOT mistake it for slot 1's token refresh (the
        # identities differ), which would overwrite slot 1's snapshot.
        store = make_store(tmp_path)
        write_live(store, codex_auth("one@example.com"))
        store.add()
        live = write_live(store, codex_auth("stray@example.com", token="z"))
        slot, _ = store.switch("1")
        assert slot == 1
        data = store._load()
        assert len(data["accounts"]) == 2  # stray was snapshotted to slot 2
        slot1 = json.loads(store._slot_path(1).read_text())
        assert slot1["tokens"]["refresh_token"] == "refresh-tok"  # not clobbered
        assert json.loads(live.read_text())["tokens"]["refresh_token"] == "refresh-tok"

    def test_switch_after_in_place_refresh_resnapshots_active_slot(self, tmp_path):
        """Codex rotates tokens in place; the refreshed live file matches no
        snapshot by content but must be treated as the ACTIVE account's
        freshest credential, not filed as a brand-new login."""
        store = self._two_accounts(tmp_path)  # active: slot 2
        refreshed = codex_auth("two@example.com", token="b-rotated")
        write_live(store, refreshed)
        slot, _ = store.switch("1")
        assert slot == 1
        data = store._load()
        assert len(data["accounts"]) == 2  # no third slot appeared
        assert store._slot_path(2).read_text() == refreshed

    def test_bare_switch_with_single_account_is_noop(self, tmp_path):
        store = make_store(tmp_path)
        live = write_live(store, codex_auth("one@example.com"))
        store.add()
        before = live.read_text()
        slot, _ = store.switch()
        assert slot == 1
        assert live.read_text() == before

    def test_switch_writes_live_0600(self, tmp_path):
        import sys

        if sys.platform == "win32":
            pytest.skip("POSIX file modes")
        store = self._two_accounts(tmp_path)
        store.switch("1")
        assert (store.live_credential_path.stat().st_mode & 0o777) == 0o600


class TestSnapshotSlot:
    def test_snapshot_slot_refreshes_content_keeping_label(self, tmp_path):
        store = make_store(tmp_path)
        write_live(store, codex_auth("one@example.com"))
        store.add()
        rotated = codex_auth("one@example.com", token="rotated")
        write_live(store, rotated)
        store.snapshot_slot(1)
        assert store._slot_path(1).read_text() == rotated
        assert store._load()["accounts"]["1"]["label"] == "one@example.com"

    def test_snapshot_unknown_slot_raises(self, tmp_path):
        store = make_store(tmp_path)
        write_live(store, codex_auth("one@example.com"))
        with pytest.raises(AccountNotFoundError):
            store.snapshot_slot(7)


class TestListStatusRemove:
    def test_list_marks_active(self, tmp_path):
        store = make_store(tmp_path)
        write_live(store, codex_auth("one@example.com"))
        store.add()
        write_live(store, codex_auth("two@example.com", token="b"))
        store.add()
        live_slot, rows = store.list_accounts()
        assert live_slot == 2
        assert [r.number for r in rows] == [1, 2]
        assert [r.active for r in rows] == [False, True]

    def test_list_active_survives_in_place_token_refresh(self, tmp_path):
        store = make_store(tmp_path)
        write_live(store, codex_auth("one@example.com"))
        store.add()
        write_live(store, codex_auth("one@example.com", token="rotated"))
        live_slot, rows = store.list_accounts()
        assert live_slot == 1
        assert rows[0].active is True

    def test_status_unmanaged_login(self, tmp_path):
        store = make_store(tmp_path)
        write_live(store, codex_auth("stray@example.com"))
        slot, note = store.status()
        assert slot is None
        assert "unmanaged" in note

    def test_status_no_login_raises(self, tmp_path):
        store = make_store(tmp_path)
        with pytest.raises(CredentialReadError):
            store.status()

    def test_remove_deletes_slot_but_not_live_login(self, tmp_path):
        store = make_store(tmp_path)
        live = write_live(store, codex_auth("one@example.com"))
        store.add()
        live_content = live.read_text()
        slot, label = store.remove("1")
        assert (slot, label) == (1, "one@example.com")
        assert not store._slot_path(1).exists()
        assert store._load()["accounts"] == {}
        assert live.read_text() == live_content

    def test_remove_unknown_raises(self, tmp_path):
        store = make_store(tmp_path)
        with pytest.raises(AccountNotFoundError):
            store.remove("1")

    def test_duplicate_label_requires_slot_number(self, tmp_path):
        store = make_store(tmp_path)
        write_live(store, codex_auth("one@example.com"))
        store.add(slot=1, label="same")
        write_live(store, codex_auth("two@example.com", token="b"))
        store.add(slot=2, label="same")
        with pytest.raises(ValidationError, match="slot number"):
            store.resolve("same")
