"""Request-time central selection uses quota evidence, never local credentials."""

from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from claude_swap.exceptions import ConfigError, SessionError
from claude_swap.settings import AutoSwitchSettings
from claude_swap.switcher import ClaudeAccountSwitcher
from claude_swap.usage_store import UsageEntry
from claude_swap.vision import VisionClient, VisionError
from claude_swap.vision_pool import (
    RELOGIN_USAGE_ERROR,
    CentralPoolCredential,
    NoCentralQuota,
    ProviderRateLimited,
    request_models,
    retry_delay,
)

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
    with pytest.raises(SessionError, match="can serve this request now"):
        pool.get()
    client.credential.assert_not_called()


def test_spent_pool_explains_each_login_and_when_one_returns(setup):
    pool, switcher, _, observations, _, _ = setup
    pool.pool.sync()
    data = switcher._get_sequence_data()
    data["accounts"]["2"]["disabled"] = True
    switcher._write_json(switcher.sequence_file, data)
    observations[item(1)["login_id"]] = observation(1, 100, reset=900)
    del observations[item(3)["login_id"]]
    with pytest.raises(NoCentralQuota) as refused:
        pool.get()
    reasons = {login.email: login.reason for login in refused.value.logins}
    assert reasons == {
        "user1@example.invalid": (
            "5h window at 100% and 7d window at 100%, "
            "resets 2027-01-15 08:15 UTC, in 15m"
        ),
        "user2@example.invalid": "disabled here; `cswap enable 2` re-enables it",
        "user3@example.invalid": "no current usage reading to select it on",
    }
    assert refused.value.available_at == NOW + 900
    assert refused.value.retry_after_seconds == 900
    assert "user2@example.invalid" in str(refused.value)


def test_launch_check_neither_issues_a_credential_nor_switches(setup):
    pool, _, client, _, _, _ = setup
    pool.check_launch()
    assert pool.current == item(1)["login_id"]
    assert pool.last_switch is None
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
    with pytest.raises(ProviderRateLimited) as error:
        pool.get()
    assert error.value.retry_after_seconds == 90
    assert not error.value.needs_action
    # Each login is named with its own window, not a bare "rate_limited".
    assert str(error.value).count("rate-limited by the provider until") == 3
    assert client.credential.call_count == calls


def test_all_logins_for_limited_account_are_excluded(setup):
    pool, _, client, _, _, _ = setup
    client.discover.return_value[2]["account_id"] = item(2)["account_id"]
    credential = pool.get()
    assert pool.rate_limited(credential, "120")["accessToken"] == "synthetic-token-1"


def _needs_relogin(client, *indexes):
    """Vision cannot issue these logins, and its refresh says they need a re-login."""
    issue = client.credential.side_effect
    blocked = {item(index)["login_id"] for index in indexes}

    def credential(account, login):
        if login in blocked:
            raise VisionError("credential_unavailable", status=409)
        return issue(account, login)

    client.credential.side_effect = credential
    client.refresh.side_effect = None
    client.refresh.return_value = {"version": 1, "generation": 1, "state": "reauth_required"}


def test_login_needing_relogin_is_skipped_within_the_same_request(setup):
    """P1: the request that finds a dead login is served by the next one."""
    pool, _, client, _, _, _ = setup
    _needs_relogin(client, 2)
    assert pool.get()["accessToken"] == "synthetic-token-3"
    calls = client.credential.call_count
    assert pool.get()["accessToken"] == "synthetic-token-3"
    # The dead login is not asked again on the next request.
    assert client.credential.call_count == calls + 1
    excluded = pool._records()[1]
    why = excluded[item(2)["login_id"]]
    assert why.command is not None and why.command.startswith(
        "cswap vision account-login "
    )
    assert why.command in why.reason
    assert why.available_at is None


def test_login_vision_cannot_issue_is_skipped_and_retried_later(setup):
    pool, _, client, observations, clock, _ = setup
    issue = client.credential.side_effect

    def credential(account, login):
        if login == item(2)["login_id"]:
            raise VisionError("unavailable", status=404)
        return issue(account, login)

    client.credential.side_effect = credential
    assert pool.get()["accessToken"] == "synthetic-token-3"
    why = pool._records()[1][item(2)["login_id"]]
    assert why.command is None and why.available_at is not None
    client.credential.side_effect = issue
    observations[item(1)["login_id"]] = observation(1, 100)
    observations[item(3)["login_id"]] = observation(3, 100)
    with pytest.raises(NoCentralQuota):
        pool.get()  # Login 2 is still set aside; 1 and 3 are now spent.
    clock.return_value = why.available_at + 1
    assert pool.get()["accessToken"] == "synthetic-token-2"


def test_identity_change_of_one_login_does_not_stop_the_pool(setup):
    pool, _, client, _, _, _ = setup
    issue = client.credential.side_effect

    def credential(account, login):
        token = issue(account, login)
        if login == item(2)["login_id"]:
            token["email"] = "someone-else@example.invalid"
        return token

    client.credential.side_effect = credential
    assert pool.get()["accessToken"] == "synthetic-token-3"


def test_every_login_failing_names_each_with_its_command(setup):
    pool, _, client, _, _, _ = setup
    _needs_relogin(client, 1, 2, 3)
    with pytest.raises(NoCentralQuota) as refused:
        pool.get()
    assert len(refused.value.logins) == 3
    assert all(login.command for login in refused.value.logins)
    assert str(refused.value).count("cswap vision account-login ") == 3
    assert refused.value.needs_action


def test_refused_key_is_not_mistaken_for_a_dead_login(setup):
    pool, _, client, _, _, _ = setup
    client.credential.side_effect = VisionError("unauthorized", status=401)
    with pytest.raises(VisionError) as refused:
        pool.get()
    assert refused.value.code == "unauthorized"
    assert client.credential.call_count == 1


def test_relogin_returns_the_login_to_the_pool_at_its_new_generation(setup):
    pool, _, client, observations, _, _ = setup
    issue = client.credential.side_effect
    _needs_relogin(client, 2)
    assert pool.get()["accessToken"] == "synthetic-token-3"
    client.credential.side_effect = issue
    client.discover.return_value[1]["login_generation"] = 2
    pool.pool.sync(force=True)
    observations[item(3)["login_id"]] = observation(3, 100)
    assert pool.get()["accessToken"] == "synthetic-token-2"


def test_expired_usage_reading_names_the_relogin_command(setup):
    """Vision's usage check sees an expired login before any request fails."""
    pool, _, _, observations, _, _ = setup
    account, entry = observations[item(2)["login_id"]]
    observations[item(2)["login_id"]] = (
        account,
        UsageEntry(last_error=RELOGIN_USAGE_ERROR, decision_blocked=True),
    )
    for index in (1, 3):
        observations[item(index)["login_id"]] = observation(index, 100)
    with pytest.raises(NoCentralQuota) as refused:
        pool.get()
    reasons = {login.email: login for login in refused.value.logins}
    expired = reasons["user2@example.invalid"]
    assert expired.command and expired.command in expired.reason
    assert pool.relogin_notices() == [
        f"cswap: user2@example.invalid needs re-login: run `{expired.command}`"
    ]


def test_relogin_notices_are_empty_when_every_login_is_fine(setup):
    pool, _, _, _, _, _ = setup
    assert pool.relogin_notices() == []


def test_relogin_notices_never_fail_a_launch(setup):
    pool, _, client, _, _, read = setup
    client.discover.side_effect = VisionError("service_unavailable")
    read.side_effect = VisionError("service_unavailable")
    pool.pool.state_path.unlink(missing_ok=True)
    assert pool.relogin_notices() == []


def _expire(observations, *indexes):
    for index in indexes:
        account, _ = observations[item(index)["login_id"]]
        observations[item(index)["login_id"]] = (
            account,
            UsageEntry(last_error=RELOGIN_USAGE_ERROR, decision_blocked=True),
        )


def test_adapter_start_prints_one_line_per_login_needing_relogin(setup, capsys):
    """P5: the warning appears before native starts, and only when needed."""
    from claude_swap.vision_proxy import InferenceProxy

    pool, _, _, observations, _, _ = setup
    with InferenceProxy(pool):
        pass
    assert capsys.readouterr().err == ""
    _expire(observations, 2)
    with InferenceProxy(pool):
        pass
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("cswap: user2@example.invalid needs re-login: run `")


def test_relogin_notices_follow_a_subclass_that_narrows_the_pool(setup):
    """The yoco launcher narrows ``_records()``; its hidden logins stay unmentioned."""
    pool, switcher, client, observations, clock, _ = setup

    class Narrowed(CentralPoolCredential):
        def _records(self):
            records, excluded = super()._records()
            keep = item(1)["email"]
            return (
                {k: v for k, v in records.items() if v.get("email") == keep},
                {k: v for k, v in excluded.items() if v.email == keep},
            )

    narrowed = Narrowed(
        switcher, client, {"visionLoginId": item(1)["login_id"]}, clock=clock
    )
    _expire(observations, 1, 2)
    assert narrowed.relogin_notices() == [
        f"cswap: {item(1)['email']} needs re-login: run "
        f"`{narrowed.relogin_command({'login_id': item(1)['login_id']})}`"
    ]


def test_relogin_notices_give_up_quickly_when_vision_hangs(setup):
    import threading
    import time as real_time

    pool, _, _, _, _, read = setup
    release = threading.Event()

    def hang(*_args, **_kwargs):
        release.wait(5)
        return {}

    read.side_effect = hang
    started = real_time.monotonic()
    try:
        assert pool.relogin_notices(timeout_seconds=0.2) == []
    finally:
        release.set()
    assert real_time.monotonic() - started < 2
