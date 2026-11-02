"""Vision's per-key usage limits, enforced on this machine (vision PR #421).

A Vision API key may carry a spend limit per Pacific calendar window (day
from midnight PT, week from Monday, month from the 1st), in USD at API LIST
PRICE: an estimate from the reported tokens at public prices, never what
Infinity pays, and every figure shown says so.

Vision states each window's standing in a ``limits`` block, in every usage
report answer and in ``GET /api/ai-accounts/usage/limits/self``::

    null                                   the key has no limit
    [{"window": "day" | "week" | "month",
      "resets_at": "<UTC instant: the end of the Pacific window>",
      "limit": {"unit": "usd_estimated", "value": 50},
      "used": {"value": 31.4, "as_of": "<UTC instant>"},
      "state": "ok" | "warning" | "exceeded"}, ...]   day -> week -> month

and refuses a credential fetch at a limit with HTTP 429, ``error.code ==
"usage_limit_reached"`` plus that window's fields and a message for people.

``LimitGate`` keeps the latest of both and answers, before each model
request, whether it must be refused here:

- an ``exceeded`` window refuses until its ``resets_at`` passes on this
  machine's clock, or until a fresh read no longer says ``exceeded``;
- a Vision 429 refuses the same way, from its own fields;
- ``null`` (no limit) never refuses, and neither does a key whose limits were
  never read;
- a read that is not the contract's shape changes nothing, so a garbled
  answer can never lift a refusal (Vision still refuses credentials itself);
- a ``warning`` window (from 80%) yields one notice line per window per
  process, which the caller shows where it does not disturb the agent.

This module is the same in claude-swap and codex-swap apart from its logger name.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

USAGE_LIMIT_REACHED = "usage_limit_reached"
WindowName = Literal["day", "week", "month"]
LimitState = Literal["ok", "warning", "exceeded"]
WINDOW_NAMES: dict[str, str] = {"day": "daily", "week": "weekly", "month": "monthly"}
STATES = ("ok", "warning", "exceeded")
LIST_PRICE = "USD at API list price, not what Infinity pays"
# How long a Vision refusal without a usable reset refuses locally. The next
# limits read (every few minutes) or the next refusal then decides again.
FALLBACK_REFUSAL_SECONDS = 300
# Vision's message is shown to people; it is kept to a sane length.
MAX_MESSAGE_CHARS = 1000

_log = logging.getLogger("claude-swap.usage_limits")
_log.addHandler(logging.NullHandler())


class LimitsFormatError(ValueError):
    """A ``limits`` value that is not the contract's shape."""


@dataclass(frozen=True)
class LimitWindow:
    """One configured window of the key, as Vision last stated it."""

    window: WindowName
    resets_at: datetime  # timezone-aware
    limit_usd: float
    used_usd: float
    state: LimitState


def _instant(value: object) -> datetime:
    if not isinstance(value, str) or len(value) > 64:
        raise LimitsFormatError("an instant must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise LimitsFormatError(f"not an ISO-8601 instant: {value!r}") from None
    if parsed.utcoffset() is None:
        raise LimitsFormatError(f"an instant needs a UTC offset: {value!r}")
    return parsed


def _amount(value: object, *, positive: bool) -> float:
    if type(value) not in (int, float):
        raise LimitsFormatError("an amount must be a number")
    amount = float(cast(float, value))
    if not math.isfinite(amount) or amount < 0 or (positive and amount == 0):
        raise LimitsFormatError(f"amount out of range: {amount}")
    return amount


def parse_window(value: object, *, state: object | None = None) -> LimitWindow:
    """One window entry; ``state`` overrides an entry that carries none (a 429)."""
    if not isinstance(value, Mapping):
        raise LimitsFormatError("a window must be an object")
    name = value.get("window")
    if name not in WINDOW_NAMES:
        raise LimitsFormatError(f"unknown window: {name!r}")
    limit = value.get("limit")
    used = value.get("used")
    if not isinstance(limit, Mapping) or limit.get("unit") != "usd_estimated":
        raise LimitsFormatError("limit must be {unit: usd_estimated, value}")
    if not isinstance(used, Mapping):
        raise LimitsFormatError("used must be {value, as_of}")
    standing = value.get("state") if state is None else state
    if standing not in STATES:
        raise LimitsFormatError(f"unknown state: {standing!r}")
    return LimitWindow(
        window=cast(WindowName, name),
        resets_at=_instant(value.get("resets_at")),
        limit_usd=_amount(limit.get("value"), positive=True),
        used_usd=_amount(used.get("value"), positive=False),
        state=cast(LimitState, standing),
    )


def parse_limits(value: object) -> tuple[LimitWindow, ...] | None:
    """The ``limits`` block: None when the key has no limit, else 1-3 windows."""
    if value is None:
        return None
    if not isinstance(value, list) or not 1 <= len(value) <= 3:
        raise LimitsFormatError("limits must be null or a list of 1-3 windows")
    return tuple(parse_window(entry) for entry in value)


def pacific_time(instant: datetime) -> str:
    """``Tue Nov 3, 12:00 AM PST``: how a reset is shown to people."""
    try:
        local = instant.astimezone(ZoneInfo("America/Los_Angeles"))
    except ZoneInfoNotFoundError:
        # No time zone database (Windows without tzdata): say UTC plainly.
        local = instant.astimezone(UTC)
    hour = local.hour % 12 or 12
    meridiem = "AM" if local.hour < 12 else "PM"
    return f"{local:%a %b} {local.day}, {hour}:{local:%M} {meridiem} {local.tzname()}"


def _clean(message: str) -> str:
    """Vision's text, on one line, without control characters, bounded."""
    printable = "".join(" " if not ch.isprintable() else ch for ch in message)
    collapsed = " ".join(printable.split())
    if len(collapsed) > MAX_MESSAGE_CHARS:
        collapsed = collapsed[: MAX_MESSAGE_CHARS - 1] + "…"
    return collapsed


def describe_vision_refusal(error: Mapping[str, object]) -> tuple[LimitWindow | None, str]:
    """The window and the text of Vision's 429 ``usage_limit_reached`` ``error``.

    The text is Vision's own message, as it wrote it, plus the reset in
    Pacific time (Vision states it as a UTC instant). The window is None when
    the refusal's fields are not the contract's.
    """
    try:
        window: LimitWindow | None = parse_window(error, state="exceeded")
    except LimitsFormatError:
        window = None
    message = error.get("message")
    if isinstance(message, str) and message.strip():
        text = _clean(message)
        if window is not None:
            text += f" (Resets {pacific_time(window.resets_at)}.)"
        return window, text
    if window is not None:
        return window, (
            f"Vision refused this API key a credential: it reached its "
            f"{WINDOW_NAMES[window.window]} usage limit ({window.used_usd:.2f} of "
            f"{window.limit_usd:.2f} {LIST_PRICE}) until {pacific_time(window.resets_at)}."
        )
    return None, "Vision refused this API key a credential: it reached a usage limit."


class LimitGate:
    """The latest limits Vision stated for this process's key, and what they refuse.

    Thread-safe: requests ask ``refusal`` while a background reader updates it.
    ``tool`` prefixes every line shown (``cswap``, ``codex-swap``).
    """

    def __init__(self, *, tool: str, clock: Callable[[], float] = time.time) -> None:
        self.tool = tool
        self.clock = clock
        self._lock = threading.Lock()
        self._windows: tuple[LimitWindow, ...] | None = None
        # From a Vision 429: (its window or None, its message, refuse until).
        self._vision: tuple[LimitWindow | None, str, float] | None = None
        self._warned: set[tuple[str, str]] = set()
        self._notices: list[str] = []

    @property
    def windows(self) -> tuple[LimitWindow, ...] | None:
        with self._lock:
            return self._windows

    def update(self, raw: object) -> None:
        """Adopt a fresh ``limits`` read; a malformed one changes nothing."""
        try:
            windows = parse_limits(raw)
        except LimitsFormatError as error:
            _log.warning("Ignored a usage limits answer that is not the contract's: %s", error)
            return
        with self._lock:
            self._windows = windows
            self._vision = None  # a fresh read decides
            for window in windows or ():
                if window.state != "warning":
                    continue
                seen = (window.window, window.resets_at.isoformat())
                if seen in self._warned:
                    continue
                self._warned.add(seen)
                notice = self._warning(window)
                self._notices.append(notice)
                _log.warning("%s", notice)

    def refuse_from_vision(self, error: Mapping[str, object]) -> str:
        """Refuse locally from Vision's 429 ``error`` object; the line to show now."""
        window, text = describe_vision_refusal(error)
        shown = f"{self.tool}: {text}"
        until = (
            window.resets_at.timestamp()
            if window is not None
            else self.clock() + FALLBACK_REFUSAL_SECONDS
        )
        with self._lock:
            self._vision = (window, shown, until)
        _log.warning("Vision refused a credential at a usage limit: %s", shown)
        return shown

    def refusal(self) -> str | None:
        """The line to answer a new model request with, or None to serve it."""
        now = self.clock()
        with self._lock:
            exceeded = [
                window
                for window in self._windows or ()
                if window.state == "exceeded" and window.resets_at.timestamp() > now
            ]
            vision = self._vision
        if vision is not None and vision[2] > now:
            if vision[0] is None:
                return vision[1]
            exceeded.append(vision[0])
        if not exceeded:
            return None
        # Vision names the exceeded window that resets last: the refusal lasts that long.
        return self._refusal(max(exceeded, key=lambda window: window.resets_at))

    def take_notices(self) -> list[str]:
        """Warning lines not shown yet; each is returned once."""
        with self._lock:
            notices, self._notices = self._notices, []
        return notices

    def _refusal(self, window: LimitWindow) -> str:
        return (
            f"{self.tool}: this Vision API key reached its {WINDOW_NAMES[window.window]} "
            f"usage limit ({window.used_usd:.2f} of {window.limit_usd:.2f} {LIST_PRICE}). "
            f"{self.tool} refuses new model requests until {pacific_time(window.resets_at)}, "
            f"when the {window.window} resets; ask the key's owner or a Vision admin "
            "to raise the limit."
        )

    def _warning(self, window: LimitWindow) -> str:
        share = math.floor(100 * window.used_usd / window.limit_usd)
        return (
            f"{self.tool}: this Vision API key has used {share}% of its "
            f"{WINDOW_NAMES[window.window]} usage limit ({window.used_usd:.2f} of "
            f"{window.limit_usd:.2f} {LIST_PRICE}). At the limit, {self.tool} refuses "
            f"new model requests until the {window.window} resets "
            f"({pacific_time(window.resets_at)})."
        )
