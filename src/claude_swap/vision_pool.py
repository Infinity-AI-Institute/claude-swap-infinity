"""Select central logins without moving native history or local credentials."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

from claude_swap.autoswitch import rank_candidates
from claude_swap.exceptions import ConfigError, NoUsableLogin
from claude_swap.locking import FileLock
from claude_swap.oauth import (
    account_headroom,
    format_countdown,
    relevant_windows,
    switch_margin,
)
from claude_swap.poll_policy import limiting_reset_ts
from claude_swap.settings import (
    load_settings,
    parse_model_names,
    parse_window_thresholds,
)
from claude_swap.vision import VisionError
from claude_swap.vision_proxy import CentralCredential
from claude_swap.vision_registry import RegistryPool
from claude_swap.vision_session import recover_rejected_credential
from claude_swap.vision_usage import read_usage


@dataclass(frozen=True)
class IneligibleLogin:
    """One Vision login that cannot serve now, why, and when it may again."""

    email: str
    reason: str
    available_at: float | None = None  # UTC epoch seconds; None when unknown


class NoCentralQuota(NoUsableLogin):
    """No enabled account can be selected using current quota evidence.

    The message names every Vision login and why it cannot serve, so both the
    launcher and native (through the adapter's error body) can say what to do.
    """

    def __init__(self, logins: list[IneligibleLogin], *, now: float):
        self.logins = tuple(logins)
        known = [
            login.available_at
            for login in self.logins
            if login.available_at is not None
        ]
        self.available_at = min(known, default=None)
        retry_after = None
        if self.available_at is not None:
            retry_after = max(1, math.ceil(self.available_at - now))
        super().__init__(
            describe_ineligible(self.logins, self.available_at, now),
            retry_after_seconds=retry_after,
        )


def describe_moment(timestamp: float, now: float) -> str:
    """An absolute UTC time plus the wait, e.g. "2026-09-26 01:29 UTC, in 1h 52m"."""
    clock = datetime.fromtimestamp(timestamp, UTC).strftime("%Y-%m-%d %H:%M")
    return f"{clock} UTC, in {format_countdown(timestamp - now)}"


def describe_ineligible(
    logins: tuple[IneligibleLogin, ...], available_at: float | None, now: float
) -> str:
    if not logins:
        return "Vision authorizes no Claude account for this key."
    details = "; ".join(f"{login.email} ({login.reason})" for login in logins)
    message = f"No Vision Claude account has known quota available: {details}."
    if len(logins) > 1 and available_at is not None:
        message += f" Earliest reset: {describe_moment(available_at, now)}."
    return message


def spent_login(row: dict, usage: dict | None, models, now: float) -> IneligibleLogin:
    """Why a roster login with no known headroom cannot be selected."""
    email = row.get("email", "")
    if account_headroom(usage, models) is None:
        return IneligibleLogin(email, "no current usage reading to select it on")
    full = [
        f"{label} window at {pct:.0f}%"
        for label, pct, _ in relevant_windows(usage, models)
        if pct >= 100
    ]
    reason = " and ".join(full)
    available_at = limiting_reset_ts(usage, models)
    if available_at is not None:
        reason += f", resets {describe_moment(available_at, now)}"
    return IneligibleLogin(email, reason, available_at)


def retry_delay(value, now):
    """Honor provider delta-seconds and HTTP-date Retry-After values."""
    if isinstance(value, str) and len(value) <= 128:
        value = value.strip()
        if value.isascii() and value.isdecimal() and len(value) <= 10:
            return max(1, int(value))
        try:
            date = parsedate_to_datetime(value)
            if date.utcoffset() is not None:
                return max(1, math.ceil(date.timestamp() - now))
        except (ValueError, TypeError, OverflowError):
            pass
    return 60


def request_models(model, configured):
    """Include the request's model limit even without an explicit model setting.

    Unknown native model names conservatively include every scoped window. A
    new provider model must not silently bypass an unfamiliar limiting window.
    """
    names = set(parse_model_names(configured))
    family = None
    if model:
        for candidate in ("opus", "sonnet", "haiku", "fable"):
            if candidate in model.lower().split("-"):
                family = candidate
                break
    names.add(family or "all")
    return tuple(sorted(names))


def launch_models(model, configured):
    """Windows a launch must clear before native starts.

    With ``--model``, the same windows its requests will use. Without it,
    native picks its own default model, so only the account-wide windows and
    any configured models count: one model's weekly limit must not refuse a
    launch that other models could still serve.
    """
    if model:
        return request_models(model, configured)
    return tuple(sorted(parse_model_names(configured)))


class CentralPoolCredential:
    def __init__(self, switcher, client, record, *, clock=time.time):
        self.switcher = switcher
        self.client = client
        self.clock = clock
        self.pool = RegistryPool(switcher, client, now=clock)
        self.current = record["visionLoginId"]
        self.last_switch = None
        self.lock = threading.RLock()
        self.rejected = {}
        self.rate_limits = {}

    def _records(self):
        """Selectable Vision rows by login, and why every other one is not."""
        self.pool.sync()
        with FileLock(self.switcher.lock_file):
            data = self.switcher._get_sequence_data() or {}
        accounts = data.get("accounts", {})
        sequence = data.get("sequence", [])
        if not isinstance(accounts, dict) or not isinstance(sequence, list):
            raise ConfigError("The account roster needs repair.")
        records = {}
        excluded = []
        now = self.clock()
        # Preserve the user's sequence as the ranking tie-breaker. The complete
        # central discovery owns membership; local rows never enter this pool.
        for number in sequence:
            row = accounts.get(str(number))
            if row is None:
                continue
            if not isinstance(row, dict):
                raise ConfigError("The account roster needs repair.")
            if row.get("source") != "vision" or row.get("visionUrl") != self.client.url:
                continue
            email = row.get("email", "")
            limited_until = self.rate_limits.get(row["visionAccountId"], 0)
            if limited_until > now:
                excluded.append(
                    IneligibleLogin(
                        email,
                        "rate-limited by the provider until "
                        + describe_moment(limited_until, now),
                        limited_until,
                    )
                )
                continue
            login = row["visionLoginId"]
            rejected = self.rejected.get(login)
            if rejected is not None:
                generation, retry_at = rejected
                if row.get("visionGeneration", 0) <= generation and now < retry_at:
                    excluded.append(
                        IneligibleLogin(
                            email,
                            "credential rejected by the provider; retrying "
                            + describe_moment(retry_at, now),
                            retry_at,
                        )
                    )
                    continue
                del self.rejected[login]
            if row.get("disabled", False):
                excluded.append(
                    IneligibleLogin(
                        email, f"disabled here; `cswap enable {number}` re-enables it"
                    )
                )
                continue
            records[login] = row
        return records, excluded

    def get(self, *, model=None):
        with self.lock:
            record, now = self._select(model, request_models)
            credential = CentralCredential(self.client, record).get(model=model)
            selected = record["visionLoginId"]
            if selected != self.current:
                self.current = selected
                self.last_switch = now
            return credential

    def check_launch(self, *, model=None):
        """Raise NoCentralQuota when no login could serve native's first request.

        Selection only: no credential is issued and the current login is kept,
        so a launch that passes behaves exactly as it did without the check.
        """
        self._select(model, launch_models)

    def _select(self, model, window_models):
        """The row to serve ``model`` with, or NoCentralQuota saying why none can."""
        with self.lock:
            records, excluded = self._records()
            now = self.clock()
            if not records:
                self._unavailable(excluded, now)
            settings = load_settings(self.switcher.backup_dir)
            models = window_models(model, settings.model)
            try:
                walls = parse_window_thresholds(settings.thresholds)
            except ValueError:
                raise ConfigError(
                    "The autoswitch threshold settings need repair."
                ) from None
            try:
                observations = read_usage(self.client, now=now)
            except VisionError:
                # An observation outage is not evidence that another account
                # has quota. An existing authorized login can still be used;
                # every credential request independently enforces its grant.
                observations = {}
            usage = {}
            for login, row in records.items():
                match = observations.get(login)
                value = None
                if match is not None and match[0] == row["visionAccountId"]:
                    value = match[1].decision_value()
                usage[login] = value if isinstance(value, dict) else None
            headroom = {
                login: account_headroom(value, models) for login, value in usage.items()
            }
            margins = {
                login: switch_margin(value, models, settings.threshold, walls)
                for login, value in usage.items()
            }
            active_headroom = headroom.get(self.current)
            active_margin = margins.get(self.current)
            absent = self.current not in records
            exhausted = active_headroom is not None and active_headroom <= 0
            forced = absent or exhausted
            consume_first = settings.strategy == "consume-first"
            cooling = (
                self.last_switch is not None
                and 0 <= now - self.last_switch < settings.cooldown_seconds
            )
            proactive = not cooling and (
                consume_first or (active_margin is not None and active_margin <= 0)
            )
            ordered = []
            if forced or proactive:
                trigger = "proactive"
                if absent:
                    trigger = "failover"
                elif exhausted:
                    trigger = "at-limit"
                if (
                    not forced
                    and consume_first
                    and active_margin is not None
                    and active_margin > 0
                ):
                    trigger = "consume-first"
                ordered, _, _ = rank_candidates(
                    models=models,
                    trigger=trigger,
                    consume_first=consume_first,
                    oauth_candidates=[
                        login for login in records if login != self.current
                    ],
                    no_return=None,
                    usage=usage,
                    headroom=headroom,
                    current=self.current,
                    active_headroom=active_headroom,
                    settings=settings,
                    now=now,
                    margins=margins,
                    active_margin=active_margin,
                )
            selected = ordered[0] if ordered else self.current
            if selected not in records or (forced and not ordered):
                # Reached only when the current login is spent or gone and no other
                # has known headroom, so every remaining row is spent or unread.
                spent = [
                    spent_login(row, usage[login], models, now)
                    for login, row in records.items()
                ]
                self._unavailable(excluded + spent, now)
            return records[selected], now

    def recover(self, rejected, *, model=None):
        with self.lock:
            # Use the identity actually rejected by the provider, not whichever
            # login another concurrent native request most recently selected.
            candidate = recover_rejected_credential(self.client, rejected)
            if candidate is not None:
                return candidate
            self.rejected[rejected["login_id"]] = (
                rejected["generation"],
                self.clock() + 60,
            )
            self.pool.sync(force=True)
            return self.get(model=model)

    def _unavailable(self, logins, now):
        remaining = [deadline - now for deadline in self.rate_limits.values()]
        remaining = [delay for delay in remaining if delay > 0]
        if remaining:
            raise VisionError(
                "rate_limited",
                status=429,
                retry_after_seconds=max(1, math.ceil(min(remaining))),
            )
        raise NoCentralQuota(logins, now=now)

    def record_rate_limit(self, credential, retry_after):
        with self.lock:
            account = credential["account_id"]
            deadline = self.clock() + retry_delay(retry_after, self.clock())
            # Quota belongs to the account, not the token generation or login.
            # A refresh or another login must not bypass the provider's delay.
            self.rate_limits[account] = max(self.rate_limits.get(account, 0), deadline)

    def rate_limited(self, credential, retry_after, *, model=None):
        with self.lock:
            self.record_rate_limit(credential, retry_after)
            try:
                return self.get(model=model)
            except NoCentralQuota:
                return None
            except VisionError as error:
                if error.code == "rate_limited":
                    return None
                raise
