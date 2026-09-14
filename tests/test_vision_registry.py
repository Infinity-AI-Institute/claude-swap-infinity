"""Registry membership is metadata, never a local refresh-token backup."""

import json
from unittest.mock import Mock

import pytest

from claude_swap.exceptions import ConfigError
from claude_swap.locking import FileLock
from claude_swap.switcher import ClaudeAccountSwitcher
from claude_swap.vision import VisionClient, VisionError
from claude_swap.vision_registry import RegistryPool, merge_accounts


def item(index=1):
    return {
        "account_id": f"aia_00000000-0000-4000-8000-{index:012d}",
        "login_id": f"ail_00000000-0000-4000-8000-{index:012d}",
        "email": f"user{index}@example.invalid",
        "organization_id": "organization",
        "subscription": {"plan": "max"},
        "login_generation": 1,
    }


@pytest.fixture
def switcher(temp_home, monkeypatch):
    monkeypatch.delenv("VISION_API_KEY", raising=False)
    instance = ClaudeAccountSwitcher()
    instance.backup_dir.mkdir(parents=True, exist_ok=True)
    return instance


def client(items=None, key="synthetic-private-key"):
    registry = VisionClient("https://vision.example.invalid", key)
    registry.discover = Mock(return_value=[item()] if items is None else items)
    return registry


def roster(switcher):
    return switcher._get_sequence_data()


def test_sync_preserves_local_profiles_and_remote_preferences(switcher):
    local = {"email": "local@example.invalid", "alias": "local", "disabled": True}
    original = {"accounts": {"4": local}, "sequence": [4], "activeAccountNumber": 4}
    switcher._write_json(switcher.sequence_file, original)
    registry = client()
    pool = RegistryPool(switcher, registry)
    assert pool.sync()
    data = roster(switcher)
    assert data["accounts"]["4"] == local
    assert data["sequence"] == [4, 5]
    assert data["activeAccountNumber"] == 4
    data["accounts"]["5"].update(alias="preferred", disabled=True)
    switcher._write_json(switcher.sequence_file, data)
    registry.discover.return_value = [{**item(), "login_generation": 2}, item(2)]
    assert pool.sync(force=True)
    data = roster(switcher)
    assert data["accounts"]["5"]["alias"] == "preferred"
    assert data["accounts"]["5"]["disabled"] is True
    assert data["accounts"]["5"]["visionGeneration"] == 2
    assert data["sequence"] == [4, 5, 6]
    assert "synthetic-private-key" not in switcher.sequence_file.read_text()
    assert "synthetic-private-key" not in pool.state_path.read_text()
    assert not list(switcher.credentials_dir.glob("*"))


def test_revocation_removes_remote_rows_and_never_reuses_retired_slots(switcher):
    registry = client()
    pool = RegistryPool(switcher, registry)
    pool.sync()
    registry.discover.return_value = []
    pool.sync(force=True)
    assert roster(switcher)["accounts"] == {}
    registry.discover.return_value = [item(2)]
    pool.sync(force=True)
    assert roster(switcher)["sequence"] == [2]


def test_failed_discovery_preserves_roster_bytes(switcher):
    registry = client()
    pool = RegistryPool(switcher, registry)
    pool.sync()
    before = switcher.sequence_file.read_bytes()
    registry.discover.side_effect = VisionError("unavailable")
    with pytest.raises(VisionError):
        pool.sync(force=True)
    assert switcher.sequence_file.read_bytes() == before


def test_network_releases_account_lock_and_late_response_cannot_replace_newer_sync(
    switcher,
):
    old = client([item(1)])
    new = client([item(2)])

    def finish_newer_request():
        lock = FileLock(switcher.lock_file, timeout=0)
        assert lock.acquire(), "Registry HTTP must not hold the account lock"
        lock.release()
        assert RegistryPool(switcher, new).sync(force=True)
        return [item(1)]

    old.discover.side_effect = finish_newer_request
    assert RegistryPool(switcher, old).sync(force=True) is False
    assert [a["email"] for a in roster(switcher)["accounts"].values()] == [
        item(2)["email"]
    ]


def test_key_change_bypasses_discovery_ttl(switcher):
    registry = client()
    pool = RegistryPool(switcher, registry, now=lambda: 100)
    assert pool.sync()
    assert pool.sync() is False
    registry.discover.assert_called_once()
    new = client([], key="another-synthetic-key")
    assert RegistryPool(switcher, new, now=lambda: 101).sync()
    assert roster(switcher)["accounts"] == {}


def test_fresh_registry_user_is_discovered_before_first_run_prompt(
    switcher, monkeypatch
):
    registry = client()
    monkeypatch.setattr(
        "claude_swap.switcher.configured_vision_client", lambda: registry
    )
    switcher._first_run_setup = Mock(side_effect=AssertionError("must not prompt"))
    result = switcher.list_accounts(json_output=True)
    assert len(result["accounts"]) == 1
    registry.discover.assert_called_once()


def test_remote_rows_never_read_or_write_local_credential_backups(switcher):
    RegistryPool(switcher, client()).sync()
    switcher._store._read_account_credentials = Mock(
        side_effect=AssertionError("local read")
    )
    switcher._store._read_account_credentials_ex = Mock(
        side_effect=AssertionError("local read")
    )
    switcher._store._write_account_credentials = Mock(
        side_effect=AssertionError("local write")
    )
    assert switcher._read_account_credentials("1", item()["email"]) == ""
    assert switcher._read_account_credentials_ex("1", item()["email"]) == ("", False)
    assert switcher._build_accounts_info()[0][5] == ""
    with pytest.raises(ConfigError, match="cannot be stored locally"):
        switcher._write_account_credentials("1", item()["email"], "synthetic-token")


@pytest.mark.parametrize(
    "data",
    [
        {"accounts": {"01": {}}},
        {"accounts": {"²": {}}},
        {"accounts": {"1": {"source": "vision", "visionLoginId": []}}},
        {"accounts": {"1": {"alias": []}}},
        {"sequence": [True]},
        {"sequence": [1, 1]},
        {"sequence": [{}]},
    ],
)
def test_malformed_roster_is_not_partially_rewritten(data):
    before = json.dumps(data)
    with pytest.raises(ConfigError):
        merge_accounts(data, [item()], "https://vision.example.invalid")
    assert json.dumps(data) == before
