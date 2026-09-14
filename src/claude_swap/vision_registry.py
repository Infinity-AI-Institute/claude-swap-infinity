"""Install complete authorized Vision membership without copying credentials."""

from __future__ import annotations

import copy
import hashlib
import time
from typing import Any

from claude_swap.exceptions import ConfigError
from claude_swap.locking import FileLock
from claude_swap.models import get_timestamp
from claude_swap.vision import VisionClient


def merge_accounts(
    data: dict[str, Any], items: list[dict[str, Any]], url: str
) -> dict[str, Any]:
    accounts = data.get("accounts", {})
    sequence = data.get("sequence", [])
    if (
        not isinstance(accounts, dict)
        or not isinstance(sequence, list)
        or any(
            not isinstance(number, str)
            or not number.isascii()
            or not number.isdecimal()
            or str(int(number)) != number
            or int(number) < 1
            or not isinstance(record, dict)
            or not isinstance(record.get("alias", ""), (str, type(None)))
            or not isinstance(record.get("visionMigratedAliases", []), list)
            or any(
                not isinstance(alias, str) or not alias
                for alias in record.get("visionMigratedAliases", [])
            )
            or (
                record.get("source") == "vision"
                and not isinstance(record.get("visionLoginId"), str)
            )
            for number, record in accounts.items()
        )
        or any(type(number) is not int or number < 1 for number in sequence)
        or len(set(sequence)) != len(sequence)
    ):
        raise ConfigError(
            "The account roster needs repair before Vision synchronization."
        )
    merged = copy.deepcopy(data)
    local = {
        number: record
        for number, record in accounts.items()
        if record.get("source") != "vision"
    }
    previous = {
        record.get("visionLoginId"): (number, record)
        for number, record in accounts.items()
        if record.get("source") == "vision" and record.get("visionUrl") == url
    }
    high_water = data.get("visionLastSlot", 0)
    if type(high_water) is not int or high_water < 0:
        raise ConfigError("The Vision slot counter needs repair.")
    high_water = max(high_water, *(int(number) for number in accounts), 0)
    names = {record.get("alias") for record in local.values() if record.get("alias")}
    remote = {}
    for item in items:
        old_number, old = previous.get(item["login_id"], (None, {}))
        if old_number is None:
            high_water += 1
            number = str(high_water)
        else:
            number = old_number
        default_alias = "vision-" + item["login_id"][4:].replace("-", "")
        alias = old.get("alias") or default_alias
        suffix = 1
        while alias in names:
            suffix += 1
            alias = f"{default_alias}-{suffix}"
        names.add(alias)
        migrated_aliases = old.get("visionMigratedAliases", [])
        if any(name in names and name != alias for name in migrated_aliases):
            raise ConfigError("A migrated Vision alias conflicts with another account.")
        names.update(migrated_aliases)
        remote[number] = {
            "email": item["email"],
            "uuid": item["login_id"],
            "organizationUuid": item["organization_id"],
            "organizationName": item["organization_id"],
            "added": old.get("added") or get_timestamp(),
            "alias": alias,
            "disabled": old.get("disabled", False),
            "visionMigratedAliases": old.get("visionMigratedAliases", []),
            "source": "vision",
            "visionUrl": url,
            "visionAccountId": item["account_id"],
            "visionLoginId": item["login_id"],
            "visionGeneration": item.get("login_generation"),
            "title": item.get("title"),
            "subscription": item["subscription"],
        }
    merged["accounts"] = {**local, **remote}
    retained = [int(number) for number in sequence if str(number) in local]
    retained.extend(int(number) for number in local if int(number) not in retained)
    merged["sequence"] = retained + [int(number) for number in remote]
    merged["visionLastSlot"] = high_water
    active = merged.get("activeAccountNumber")
    if active is not None and str(active) not in merged["accounts"]:
        merged["activeAccountNumber"] = None
    merged["lastUpdated"] = get_timestamp()
    return merged


class RegistryPool:
    def __init__(self, switcher, client: VisionClient, *, now=time.time):
        self.switcher = switcher
        self.client = client
        self.now = now
        self.state_path = switcher.backup_dir / "vision-sync.json"
        self.key_fingerprint = hashlib.sha256(client.api_key.encode()).hexdigest()

    def _state(self) -> dict[str, Any]:
        state = self.switcher._read_json(self.state_path, strict=True) or {}
        if (
            not isinstance(state, dict)
            or type(state.get("revision", 0)) is not int
            or state.get("revision", 0) < 0
        ):
            raise ConfigError("Vision synchronization state needs repair.")
        return state

    def sync(self, *, force: bool = False) -> bool:
        # Allocate a monotonically increasing request before networking. A late
        # response may commit only if no newer discovery has started, regardless
        # of wall-clock changes or whether the newer request succeeded.
        with FileLock(self.switcher.lock_file):
            state = self._state()
            completed = state.get("completedAt")
            if (
                not force
                and state.get("url") == self.client.url
                and state.get("keyFingerprint") == self.key_fingerprint
                and type(completed) in (int, float)
                and 0 <= self.now() - completed < 30
            ):
                return False
            revision = state.get("revision", 0) + 1
            pending = {**state, "revision": revision}
            self.switcher._write_json(self.state_path, pending)
        items = self.client.discover()
        with FileLock(self.switcher.lock_file):
            if self._state().get("revision") != revision:
                return False
            current = self.switcher._get_sequence_data() or {}
            merged = merge_accounts(current, items, self.client.url)
            self.switcher._write_json(self.switcher.sequence_file, merged)
            self.switcher._write_json(
                self.state_path,
                {
                    "revision": revision,
                    "url": self.client.url,
                    "keyFingerprint": self.key_fingerprint,
                    "completedAt": self.now(),
                    "accountCount": len(items),
                },
            )
        return True
