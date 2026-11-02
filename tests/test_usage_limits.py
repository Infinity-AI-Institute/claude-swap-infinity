"""Vision's per-key usage limits, as the local gate reads and enforces them.

The shapes are vision PR #421's: ``limits`` in every report answer and in
``GET /api/ai-accounts/usage/limits/self``, and the 429 a credential fetch
gets at a limit (``error.code == "usage_limit_reached"``).
"""

from datetime import UTC, datetime, timedelta

import pytest

from claude_swap.usage_limits import (
    LimitGate,
    LimitsFormatError,
    parse_limits,
    pacific_time,
)

NOW = datetime(2026, 11, 2, 21, 0, tzinfo=UTC)  # Mon Nov 2, 1:00 PM PST
DAY_END = "2026-11-03T08:00:00Z"  # Tue Nov 3, 12:00 AM PST
WEEK_END = "2026-11-09T08:00:00Z"  # Mon Nov 9, 12:00 AM PST


class Clock:
    def __init__(self, at=NOW):
        self.now = at.timestamp()

    def __call__(self):
        return self.now


def window(name="day", *, state="ok", used=10.0, limit=50.0, resets_at=DAY_END):
    return {
        "window": name,
        "resets_at": resets_at,
        "limit": {"unit": "usd_estimated", "value": limit},
        "used": {"value": used, "as_of": "2026-11-02T20:55:00Z"},
        "state": state,
    }


def vision_refusal(name="week", *, used=31.4, limit=30.0, resets_at=WEEK_END):
    """The ``error`` object of Vision's 429 at a limit, as PR #421 builds it."""
    return {
        "message": (
            f"This API key reached its {name} usage limit ({used:.2f} of "
            f"{limit:.2f} USD at API list price, not what Infinity pays). "
            f"Credentials are refused until {resets_at}, when the window "
            "resets; ask its owner or an admin to raise the limit."
        ),
        "code": "usage_limit_reached",
        "window": name,
        "resets_at": resets_at,
        "limit": {"unit": "usd_estimated", "value": limit},
        "used": {"value": used, "as_of": "2026-11-02T20:55:00Z"},
    }


def gate(clock=None):
    return LimitGate(tool="cswap", clock=clock or Clock())


# -- parsing ------------------------------------------------------------------


def test_null_means_the_key_has_no_limit():
    assert parse_limits(None) is None


def test_windows_are_parsed_in_the_contract_shape():
    [day, week] = parse_limits([window(), window("week", resets_at=WEEK_END)])
    assert (day.window, day.limit_usd, day.used_usd, day.state) == ("day", 50.0, 10.0, "ok")
    assert day.resets_at == datetime(2026, 11, 3, 8, 0, tzinfo=UTC)
    assert week.window == "week"


@pytest.mark.parametrize(
    "broken",
    [
        "nope",
        [],
        [window(state="maybe")],
        [window("year")],
        [{**window(), "resets_at": "2026-11-03 08:00"}],  # no offset
        [{**window(), "limit": {"unit": "eur", "value": 5}}],
        [{**window(), "limit": {"unit": "usd_estimated", "value": 0}}],
        [{**window(), "used": {"value": -1, "as_of": "2026-11-02T20:55:00Z"}}],
        [{**window(), "used": {"value": True, "as_of": "2026-11-02T20:55:00Z"}}],
        [window(), "x"],
    ],
)
def test_anything_else_is_a_format_error(broken):
    with pytest.raises(LimitsFormatError):
        parse_limits(broken)


def test_pacific_time_names_the_day_and_zone():
    assert pacific_time(datetime(2026, 11, 3, 8, 0, tzinfo=UTC)) == "Tue Nov 3, 12:00 AM PST"
    assert pacific_time(datetime(2026, 10, 1, 19, 30, tzinfo=UTC)) == "Thu Oct 1, 12:30 PM PDT"


# -- refusing -----------------------------------------------------------------


def test_no_limits_never_refuses():
    limits = gate()
    assert limits.refusal() is None
    limits.update(None)
    assert limits.refusal() is None


def test_ok_and_warning_windows_do_not_refuse():
    limits = gate()
    limits.update([window(state="ok"), window("week", state="warning", resets_at=WEEK_END)])
    assert limits.refusal() is None


def test_an_exceeded_window_refuses_naming_it_the_dollars_and_the_pacific_reset():
    limits = gate()
    limits.update([window(state="exceeded", used=52.5, limit=50.0)])
    assert limits.refusal() == (
        "cswap: this Vision API key reached its daily usage limit (52.50 of 50.00 "
        "USD at API list price, not what Infinity pays). cswap refuses new model "
        "requests until Tue Nov 3, 12:00 AM PST, when the day resets; ask the "
        "key's owner or a Vision admin to raise the limit."
    )


def test_with_several_exceeded_the_one_resetting_last_is_named():
    limits = gate()
    limits.update(
        [
            window(state="exceeded", used=60, limit=50),
            window("week", state="exceeded", used=31.4, limit=30, resets_at=WEEK_END),
        ]
    )
    assert "weekly usage limit (31.40 of 30.00 USD" in limits.refusal()
    assert "until Mon Nov 9, 12:00 AM PST, when the week resets" in limits.refusal()


def test_the_refusal_lifts_by_itself_at_the_reset():
    clock = Clock()
    limits = gate(clock)
    limits.update([window(state="exceeded")])
    clock.now = datetime(2026, 11, 3, 7, 59, 59, tzinfo=UTC).timestamp()
    assert limits.refusal() is not None
    clock.now = datetime(2026, 11, 3, 8, 0, tzinfo=UTC).timestamp()
    assert limits.refusal() is None


def test_a_fresh_read_that_is_not_exceeded_lifts_the_refusal():
    limits = gate()
    limits.update([window(state="exceeded")])
    limits.update([window(state="ok", limit=100)])  # an admin raised it
    assert limits.refusal() is None


def test_a_limit_removed_lifts_the_refusal():
    limits = gate()
    limits.update([window(state="exceeded")])
    limits.update(None)
    assert limits.refusal() is None


def test_a_malformed_read_keeps_the_last_known_limits():
    limits = gate()
    limits.update([window(state="exceeded")])
    limits.update("garbled")
    assert limits.refusal() is not None


# -- Vision's 429 -------------------------------------------------------------


def test_visions_refusal_is_shown_as_is_with_its_reset_in_pacific_time():
    limits = gate()
    shown = limits.refuse_from_vision(vision_refusal())
    assert shown == (
        "cswap: " + vision_refusal()["message"] + " (Resets Mon Nov 9, 12:00 AM PST.)"
    )


def test_visions_refusal_refuses_locally_until_its_reset():
    clock = Clock()
    limits = gate(clock)
    limits.refuse_from_vision(vision_refusal())
    assert "weekly usage limit (31.40 of 30.00 USD" in limits.refusal()
    clock.now = datetime(2026, 11, 9, 8, 0, tzinfo=UTC).timestamp()
    assert limits.refusal() is None


def test_a_fresh_read_after_visions_refusal_decides():
    limits = gate()
    limits.refuse_from_vision(vision_refusal())
    limits.update([window("week", state="ok", limit=100, resets_at=WEEK_END)])
    assert limits.refusal() is None


def test_a_refusal_without_usable_fields_still_refuses_for_a_while():
    clock = Clock()
    limits = gate(clock)
    shown = limits.refuse_from_vision({"code": "usage_limit_reached", "message": "Limit reached."})
    assert shown == "cswap: Limit reached."
    assert limits.refusal() == "cswap: Limit reached."
    clock.now += 301
    assert limits.refusal() is None  # the next limits read or 429 decides again


def test_visions_message_is_bounded_and_stripped_of_control_characters():
    limits = gate()
    shown = limits.refuse_from_vision(
        {**vision_refusal(), "message": "Over\x1b[2J the limit.\n" + "x" * 5000}
    )
    assert "\x1b" not in shown and "\n" not in shown
    assert len(shown) < 1200


# -- warnings -----------------------------------------------------------------


def test_a_warning_is_one_line_once_per_window():
    limits = gate()
    limits.update([window(state="warning", used=41, limit=50)])
    assert limits.take_notices() == [
        "cswap: this Vision API key has used 82% of its daily usage limit (41.00 of "
        "50.00 USD at API list price, not what Infinity pays). At the limit, cswap "
        "refuses new model requests until the day resets (Tue Nov 3, 12:00 AM PST)."
    ]
    limits.update([window(state="warning", used=45, limit=50)])
    assert limits.take_notices() == []


def test_the_next_window_warns_again():
    limits = gate()
    limits.update([window(state="warning", used=41, limit=50)])
    limits.take_notices()
    limits.update([window(state="warning", used=41, limit=50, resets_at="2026-11-04T08:00:00Z")])
    assert len(limits.take_notices()) == 1


def test_ok_and_exceeded_windows_hold_no_warning():
    limits = gate()
    limits.update([window(state="ok"), window("week", state="exceeded", resets_at=WEEK_END)])
    assert limits.take_notices() == []


def test_a_week_ahead_reset_reads_its_own_date():
    later = (NOW + timedelta(days=6)).replace(hour=8, minute=0)
    assert pacific_time(later) == "Sun Nov 8, 12:00 AM PST"


def test_without_a_time_zone_database_the_reset_reads_in_utc(monkeypatch):
    from zoneinfo import ZoneInfoNotFoundError

    def missing(_key):
        raise ZoneInfoNotFoundError("no tzdata")

    monkeypatch.setattr("claude_swap.usage_limits.ZoneInfo", missing)
    assert pacific_time(datetime(2026, 11, 3, 8, 0, tzinfo=UTC)) == "Tue Nov 3, 8:00 AM UTC"
