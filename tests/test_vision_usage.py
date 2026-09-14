"""Central usage keeps provider observation age and never refreshes locally."""

import copy
from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from claude_swap.usage_store import STALE_OK_S
from claude_swap.vision import VisionClient, VisionError
from claude_swap.vision_usage import collect_usage, project_status, read_usage

NOW = 1_789_388_000.0


def iso(value):
    return datetime.fromtimestamp(value, UTC).isoformat()


def status(age=10, state="fresh"):
    return {
        "state": state,
        "observed_at": iso(NOW - age),
        "last_attempt_at": iso(NOW - 2),
        "next_attempt_at": iso(NOW + 30),
        "error_code": None,
        "snapshot": {
            "provider": "claude",
            "observedAt": iso(NOW - age),
            "ordinaryUsageAllowed": None,
            "windows": [
                {
                    "bucket": "five_hour",
                    "slot": "five_hour",
                    "kind": "five_hour",
                    "usedPercent": 34,
                    "resetsAt": iso(NOW + 300),
                },
                {
                    "bucket": "seven_day",
                    "slot": "seven_day",
                    "kind": "weekly",
                    "usedPercent": 61,
                    "resetsAt": iso(NOW + 900),
                },
                {
                    "bucket": "Fable",
                    "slot": "limits",
                    "kind": "weekly",
                    "usedPercent": 99,
                    "resetsAt": None,
                },
            ],
        },
    }


def row(index=1):
    return {
        "provider": "claude",
        "login_id": f"ail_00000000-0000-4000-8000-{index:012d}",
        "account_id": f"aia_00000000-0000-4000-8000-{index:012d}",
        "status": status(),
    }


def client():
    return VisionClient("https://vision.example.invalid", "synthetic-key")


def test_projection_preserves_observation_time_and_scoped_quota():
    entry = project_status(status(), now=NOW)
    assert entry.fetched_at == NOW - 10
    assert entry.age_s == 10
    assert entry.last_good["five_hour"]["pct"] == 34
    assert entry.last_good["seven_day"]["pct"] == 61
    assert entry.last_good["scoped"] == [{"name": "Fable", "pct": 99.0}]
    assert entry.decision_value() == entry.last_good


@pytest.mark.parametrize("age,state", [(10, "stale"), (STALE_OK_S + 1, "fresh")])
def test_reading_stale_observation_does_not_renew_decision_trust(age, state):
    entry = project_status(status(age, state), now=NOW)
    assert entry.last_good["five_hour"]["pct"] == 34
    assert entry.age_s == age
    assert entry.decision_value() is None


def test_unknown_future_window_preserves_display_but_blocks_selection():
    value = status()
    value["snapshot"]["windows"].append(
        {
            "bucket": "new-limit",
            "slot": "limits",
            "kind": "other",
            "usedPercent": 100,
            "resetsAt": None,
        }
    )
    entry = project_status(value, now=NOW)
    assert entry.last_good is not None
    assert entry.decision_value() is None


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s.update(observed_at=iso(NOW + 60)),
        lambda s: s["snapshot"].update(provider="codex"),
        lambda s: s["snapshot"]["windows"][0].update(usedPercent=float("nan")),
        lambda s: s["snapshot"]["windows"][0].update(usedPercent=True),
        lambda s: s["snapshot"]["windows"].append(
            copy.deepcopy(s["snapshot"]["windows"][0])
        ),
        lambda s: s.update(next_attempt_at="2026-09-14T01:00:00"),
    ],
)
def test_malformed_observations_fail_as_a_unit(mutation):
    value = status()
    mutation(value)
    with pytest.raises(VisionError):
        project_status(value, now=NOW)


def test_usage_pagination_uses_after_and_returns_complete_pool():
    registry = client()
    registry.request = Mock(
        side_effect=[
            {"accounts": [row(1)], "next_cursor": row(1)["login_id"]},
            {"accounts": [row(2)], "next_cursor": None},
        ]
    )
    assert len(read_usage(registry, now=NOW)) == 2
    assert "after=ail_" in registry.request.call_args_list[1].args[1]
    assert all(call.args[0] == "GET" for call in registry.request.call_args_list)


def test_bad_later_page_never_returns_partial_usage():
    registry = client()
    registry.request = Mock(
        side_effect=[
            {"accounts": [row(1)], "next_cursor": row(1)["login_id"]},
            {"accounts": [row(1)], "next_cursor": None},
        ]
    )
    with pytest.raises(VisionError):
        read_usage(registry, now=NOW)


def test_usage_is_bound_to_registry_origin_and_account_identity(monkeypatch):
    registry = client()
    registry.request = Mock(return_value={"accounts": [row()], "next_cursor": None})
    monkeypatch.setattr("claude_swap.vision_usage.time.time", lambda: NOW)
    record = {
        "visionUrl": registry.url,
        "visionLoginId": row()["login_id"],
        "visionAccountId": row()["account_id"],
    }
    values = collect_usage(
        {
            "1": record,
            "2": {**record, "visionUrl": "https://another.example.invalid"},
            "3": {**record, "visionAccountId": row(2)["account_id"]},
        },
        registry,
    )
    assert values["1"].decision_value() is not None
    assert values["2"].decision_value() is None
    assert values["3"].decision_value() is None


def test_remote_collection_never_enters_local_refresh_pipeline(temp_home, monkeypatch):
    from claude_swap.switcher import ClaudeAccountSwitcher
    from claude_swap.vision_registry import RegistryPool

    monkeypatch.delenv("VISION_API_KEY", raising=False)
    switcher = ClaudeAccountSwitcher()
    registry = client()
    registry.discover = Mock(
        return_value=[
            {
                **{key: row()[key] for key in ("login_id", "account_id")},
                "email": "synthetic@example.invalid",
                "organization_id": "organization",
                "subscription": {"plan": "max"},
            }
        ]
    )
    RegistryPool(switcher, registry).sync()
    registry.request = Mock(return_value={"accounts": [row()], "next_cursor": None})
    monkeypatch.setattr(
        "claude_swap.switcher.configured_vision_client", lambda: registry
    )
    monkeypatch.setattr("claude_swap.vision_usage.time.time", lambda: NOW)
    switcher._run_usage_fetches = Mock(
        side_effect=AssertionError("local provider pipeline")
    )
    entry = switcher.usage_entries_by_account()["1"]
    assert entry.last_good["five_hour"]["pct"] == 34
    assert entry.decision_value() is not None
