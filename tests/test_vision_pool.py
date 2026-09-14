"""Request-time central selection uses quota evidence, never local credentials."""

from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from claude_swap.exceptions import ConfigError, SessionError
from claude_swap.settings import AutoSwitchSettings
from claude_swap.switcher import ClaudeAccountSwitcher
from claude_swap.usage_store import UsageEntry
from claude_swap.vision import VisionClient, VisionError
from claude_swap.vision_pool import CentralPoolCredential, request_models, retry_delay

NOW = 1_800_000_000.0


def item(index):
    return {
        "account_id": f"aia_00000000-0000-4000-8000-{index:012d}",
        "login_id": f"ail_00000000-0000-4000-8000-{index:012d}",
        "email": f"user{index}@example.invalid",
        "organization_id": "organization",
        "subscription": {"plan": "max"},
        "login_generation": 1,
    }


def observation(index, pct, *, blocked=False, age=10, reset=900, scoped=None):
    row = item(index)
    usage = {
        "five_hour": {"pct": pct},
        "seven_day": {
            "pct": pct,
            "resets_at": datetime.fromtimestamp(NOW + reset, UTC).isoformat(),
        },
        "scoped": scoped or [],
    }
    return row["account_id"], UsageEntry(
        last_good=usage, age_s=age, decision_blocked=blocked
    )


@pytest.fixture
def setup(temp_home, monkeypatch):
    monkeypatch.delenv("VISION_API_KEY", raising=False)
    switcher = ClaudeAccountSwitcher()
    switcher.backup_dir.mkdir(parents=True, exist_ok=True)
    client = VisionClient("https://vision.example.invalid", "synthetic-key")
    client.discover = Mock(return_value=[item(1), item(2), item(3)])

    def credential(account, login):
        index = int(login[-12:])
        row = item(index)
        return {
            "account_id": account,
            "login_id": login,
            "email": row["email"],
            "organization_id": row["organization_id"],
            "generation": 1,
            "kind": "login_oauth",
            "accessToken": f"synthetic-token-{index}",
        }

    client.credential = Mock(side_effect=credential)
    client.refresh = Mock(side_effect=AssertionError("unexpected refresh"))
    observations = {
        item(1)["login_id"]: observation(1, 95),
        item(2)["login_id"]: observation(2, 20),
        item(3)["login_id"]: observation(3, 40),
    }
    read = Mock(side_effect=lambda *_args, **_kwargs: observations)
    monkeypatch.setattr("claude_swap.vision_pool.read_usage", read)
    clock = Mock(return_value=NOW)
    pool = CentralPoolCredential(
        switcher, client, {"visionLoginId": item(1)["login_id"]}, clock=clock
    )
    return pool, switcher, client, observations, clock, read


def test_threshold_switch_uses_complete_pool_without_changing_global_active(setup):
    pool, switcher, client, _, _, read = setup
    assert pool.get(model="claude-sonnet-4-6")["accessToken"] == "synthetic-token-2"
    assert pool.get(model="claude-sonnet-4-6")["accessToken"] == "synthetic-token-2"
    assert client.discover.call_count == 1
    assert read.call_count == client.credential.call_count == 2
    assert switcher._get_sequence_data().get("activeAccountNumber") is None
    assert not list(switcher.credentials_dir.glob("*"))
    client.refresh.assert_not_called()


def test_disabled_local_preference_is_read_on_every_request(setup):
    pool, switcher, _, _, _, _ = setup
    pool.pool.sync()
    data = switcher._get_sequence_data()
    data["accounts"]["2"]["disabled"] = True
    switcher._write_json(switcher.sequence_file, data)
    assert pool.get()["accessToken"] == "synthetic-token-3"
    data["accounts"]["3"]["disabled"] = True
    switcher._write_json(switcher.sequence_file, data)
    assert pool.get()["accessToken"] == "synthetic-token-1"


@pytest.mark.parametrize("age,blocked", [(400, False), (10, True)])
def test_stale_or_unknown_candidate_is_not_selected(setup, age, blocked):
    pool, _, _, observations, _, _ = setup
    observations[item(2)["login_id"]] = observation(2, 0, age=age, blocked=blocked)
    assert pool.get()["accessToken"] == "synthetic-token-3"


def test_usage_account_identity_must_match_registry(setup):
    pool, _, _, observations, _, _ = setup
    observations[item(2)["login_id"]] = observation(3, 0)
    assert pool.get()["accessToken"] == "synthetic-token-3"


def test_revoked_membership_is_removed_at_next_discovery(setup):
    pool, _, client, _, clock, _ = setup
    assert pool.get()["accessToken"] == "synthetic-token-2"
    client.discover.return_value = [item(1), item(3)]
    clock.return_value += 31
    assert pool.get()["accessToken"] == "synthetic-token-3"


def test_cooldown_prevents_proactive_flap_but_not_escape_from_hard_limit(setup):
    pool, _, _, observations, _, _ = setup
    assert pool.get()["accessToken"] == "synthetic-token-2"
    observations[item(2)["login_id"]] = observation(2, 95)
    observations[item(1)["login_id"]] = observation(1, 0)
    assert pool.get()["accessToken"] == "synthetic-token-2"
    observations[item(2)["login_id"]] = observation(2, 100)
    assert pool.get()["accessToken"] == "synthetic-token-1"


def test_request_model_weekly_limit_binds_selection(setup):
    pool, _, _, observations, _, _ = setup
    observations[item(1)["login_id"]] = observation(
        1, 0, scoped=[{"name": "Sonnet", "pct": 100}]
    )
    assert pool.get(model="claude-sonnet-4-6")["accessToken"] == "synthetic-token-2"


def test_observation_outage_keeps_current_authorized_login(setup):
    pool, _, client, _, _, read = setup
    read.side_effect = VisionError("service_unavailable")
    assert pool.get()["accessToken"] == "synthetic-token-1"
    client.credential.assert_called_once_with(
        item(1)["account_id"], item(1)["login_id"]
    )


def test_no_known_quota_fails_without_provider_request(setup):
    pool, _, client, observations, _, _ = setup
    for index in range(1, 4):
        observations[item(index)["login_id"]] = observation(index, 100)
    with pytest.raises(SessionError, match="known quota"):
        pool.get()
    client.credential.assert_not_called()


def test_failed_credential_does_not_commit_account_switch(setup):
    pool, _, client, _, _, _ = setup
    client.credential.side_effect = VisionError("forbidden", status=403)
    with pytest.raises(VisionError):
        pool.get()
    assert pool.current == item(1)["login_id"]
    assert pool.last_switch is None


def test_consume_first_uses_soonest_weekly_reset(setup, monkeypatch):
    pool, _, _, observations, _, _ = setup
    monkeypatch.setattr(
        "claude_swap.vision_pool.load_settings",
        lambda _: AutoSwitchSettings(strategy="consume-first"),
    )
    observations[item(1)["login_id"]] = observation(1, 20, reset=900)
    observations[item(2)["login_id"]] = observation(2, 20, reset=600)
    observations[item(3)["login_id"]] = observation(3, 30, reset=300)
    assert pool.get()["accessToken"] == "synthetic-token-3"


def test_unknown_model_includes_all_scoped_windows():
    assert request_models("new-provider-model", None) == ("all",)
    assert set(request_models("claude-sonnet-4-6", "Opus")) == {"Opus", "sonnet"}


def test_malformed_threshold_settings_fail_before_credential_issue(setup, monkeypatch):
    pool, _, client, _, _, _ = setup
    monkeypatch.setattr(
        "claude_swap.vision_pool.load_settings",
        lambda _: AutoSwitchSettings(thresholds="not-a-threshold"),
    )
    with pytest.raises(ConfigError, match="threshold settings"):
        pool.get()
    client.credential.assert_not_called()


def test_rejected_token_uses_another_account_and_avoids_immediate_return(setup):
    pool, _, client, observations, _, _ = setup
    client.refresh.side_effect = None
    client.refresh.return_value = {"state": "current"}
    rejected = pool.get()
    assert rejected["accessToken"] == "synthetic-token-2"
    assert pool.recover(rejected)["accessToken"] == "synthetic-token-3"
    observations[item(3)["login_id"]] = observation(3, 100)
    assert pool.get()["accessToken"] == "synthetic-token-1"
    client.refresh.assert_called_once_with(
        item(2)["account_id"], item(2)["login_id"], 1
    )


def test_new_generation_can_reenter_pool_before_rejection_cooldown(setup):
    pool, _, client, observations, _, _ = setup
    client.refresh.side_effect = None
    client.refresh.return_value = {"state": "current"}
    rejected = pool.get()
    assert pool.recover(rejected)["accessToken"] == "synthetic-token-3"
    previous_credential = client.credential.side_effect

    def credential(account, login):
        token = previous_credential(account, login)
        if login == rejected["login_id"]:
            token.update(generation=2, accessToken="synthetic-successor")
        return token

    client.credential.side_effect = credential
    client.discover.return_value[1]["login_generation"] = 2
    pool.pool.sync(force=True)
    observations[item(3)["login_id"]] = observation(3, 100)
    assert pool.get()["accessToken"] == "synthetic-successor"


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, 60),
        ("invalid", 60),
        ("120", 120),
        ("0", 1),
        ("１２", 60),
        ("99999999999", 60),
    ],
)
def test_retry_after_delta_parsing(value, expected):
    assert retry_delay(value, NOW) == expected


def test_retry_after_http_date():
    from email.utils import format_datetime

    value = format_datetime(datetime.fromtimestamp(NOW + 125, UTC), usegmt=True)
    assert retry_delay(value, NOW) == 125


def test_rate_limit_selects_another_account_without_refresh(setup):
    pool, _, client, _, _, _ = setup
    credential = pool.get()
    assert pool.rate_limited(credential, "120")["accessToken"] == "synthetic-token-3"
    assert pool.rate_limits[credential["account_id"]] == NOW + 120
    client.refresh.assert_not_called()


def test_rate_limit_survives_new_generation_until_retry_deadline(setup):
    pool, _, client, observations, clock, _ = setup
    credential = pool.get()
    pool.rate_limited(credential, "120")
    client.discover.return_value[1]["login_generation"] = 2
    pool.pool.sync(force=True)
    observations[item(3)["login_id"]] = observation(3, 100)
    assert pool.get()["accessToken"] == "synthetic-token-1"
    clock.return_value = NOW + 121
    observations[item(1)["login_id"]] = observation(1, 100)
    assert pool.get()["accessToken"] == "synthetic-token-2"


def test_all_accounts_limited_returns_remaining_retry_delay(setup):
    pool, _, client, _, clock, _ = setup
    first = pool.get()
    second = pool.rate_limited(first, "120")
    third = pool.rate_limited(second, "180")
    assert pool.rate_limited(third, "240") is None
    calls = client.credential.call_count
    clock.return_value += 30
    with pytest.raises(VisionError) as error:
        pool.get()
    assert error.value.status == 429
    assert error.value.retry_after_seconds == 90
    assert client.credential.call_count == calls


def test_all_logins_for_limited_account_are_excluded(setup):
    pool, _, client, _, _, _ = setup
    client.discover.return_value[2]["account_id"] = item(2)["account_id"]
    credential = pool.get()
    assert pool.rate_limited(credential, "120")["accessToken"] == "synthetic-token-1"
