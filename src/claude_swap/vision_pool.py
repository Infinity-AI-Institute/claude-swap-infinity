"""Select central logins without moving native history or local credentials."""

from __future__ import annotations

import threading
import time

from claude_swap.autoswitch import rank_candidates
from claude_swap.exceptions import ConfigError, SessionError
from claude_swap.locking import FileLock
from claude_swap.oauth import account_headroom, switch_margin
from claude_swap.settings import (
    load_settings,
    parse_model_names,
    parse_window_thresholds,
)
from claude_swap.vision import VisionError
from claude_swap.vision_proxy import CentralCredential
from claude_swap.vision_registry import RegistryPool
from claude_swap.vision_usage import read_usage


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


class CentralPoolCredential:
    def __init__(self, switcher, client, record, *, clock=time.time):
        self.switcher = switcher
        self.client = client
        self.clock = clock
        self.pool = RegistryPool(switcher, client, now=clock)
        self.current = record["visionLoginId"]
        self.last_switch = None
        self.lock = threading.Lock()

    def _records(self):
        self.pool.sync()
        with FileLock(self.switcher.lock_file):
            data = self.switcher._get_sequence_data() or {}
        accounts = data.get("accounts", {})
        sequence = data.get("sequence", [])
        if not isinstance(accounts, dict) or not isinstance(sequence, list):
            raise ConfigError("The account roster needs repair.")
        records = {}
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
            if not row.get("disabled", False):
                records[row["visionLoginId"]] = row
        return records

    def get(self, *, model=None):
        with self.lock:
            records = self._records()
            if not records:
                raise SessionError("No enabled central Claude login is available.")
            now = self.clock()
            settings = load_settings(self.switcher.backup_dir)
            models = request_models(model, settings.model)
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
                raise SessionError(
                    "No enabled central Claude login has known quota available."
                )
            credential = CentralCredential(self.client, records[selected]).get(
                model=model
            )
            if selected != self.current:
                self.current = selected
                self.last_switch = now
            return credential
