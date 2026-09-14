"""Adapt central observations without polling or refreshing provider credentials."""

from __future__ import annotations

import math
import time
import urllib.parse
from datetime import datetime
from typing import Any

from claude_swap.oauth import build_usage_result
from claude_swap.usage_store import UsageEntry
from claude_swap.vision import VisionClient, VisionError, registry_id


def timestamp(value: Any, *, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    try:
        if not isinstance(value, str) or len(value) > 100:
            raise ValueError()
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            raise ValueError()
        result = parsed.timestamp()
        if not math.isfinite(result):
            raise ValueError()
        return result
    except (ValueError, OverflowError):
        raise VisionError("service_unavailable") from None


def project_status(status: Any, *, now: float) -> UsageEntry:
    if not isinstance(status, dict) or status.get("state") not in (
        "fresh",
        "stale",
        "unknown",
        "unsupported",
    ):
        raise VisionError("service_unavailable")
    observed = timestamp(status.get("observed_at"), nullable=True)
    attempted = timestamp(status.get("last_attempt_at"), nullable=True)
    next_attempt = timestamp(status.get("next_attempt_at"))
    snapshot = status.get("snapshot")
    usage = None
    blocked = status["state"] != "fresh" or status.get("error_code") is not None
    if snapshot is not None:
        if not isinstance(snapshot, dict) or snapshot.get("provider") != "claude":
            raise VisionError("service_unavailable")
        if observed is None or timestamp(snapshot.get("observedAt")) != observed:
            raise VisionError("service_unavailable")
        windows = snapshot.get("windows")
        if not isinstance(windows, list) or len(windows) > 256:
            raise VisionError("service_unavailable")
        raw: dict[str, Any] = {"limits": []}
        seen = set()
        for window in windows:
            if not isinstance(window, dict):
                raise VisionError("service_unavailable")
            bucket, slot = window.get("bucket"), window.get("slot")
            pct = window.get("usedPercent")
            if (
                not isinstance(bucket, str)
                or not 1 <= len(bucket) <= 400
                or not isinstance(slot, str)
                or not 1 <= len(slot) <= 400
                or type(pct) not in (int, float)
                or not math.isfinite(pct)
                or pct < 0
                or (bucket, slot) in seen
            ):
                raise VisionError("service_unavailable")
            seen.add((bucket, slot))
            resets = window.get("resetsAt")
            timestamp(resets, nullable=True)
            if bucket in ("five_hour", "seven_day") and slot == bucket:
                expected = "five_hour" if bucket == "five_hour" else "weekly"
                if window.get("kind") != expected:
                    raise VisionError("service_unavailable")
                raw[bucket] = {"utilization": pct, "resets_at": resets}
            elif window.get("kind") == "weekly":
                name = bucket.removeprefix("seven_day_") if slot == bucket else bucket
                raw["limits"].append(
                    {
                        "scope": {"model": {"display_name": name}},
                        "percent": pct,
                        "resets_at": resets,
                    }
                )
            else:
                # Preserve uncertainty when a new limiting window cannot yet be
                # represented by the local selection model.
                blocked = True
        usage = build_usage_result(raw)
        if snapshot.get("ordinaryUsageAllowed") is False:
            blocked = True
    if observed is not None and observed > now + 5:
        raise VisionError("service_unavailable")
    return UsageEntry(
        last_good=usage,
        fetched_at=observed,
        age_s=max(0, now - observed) if observed is not None else None,
        last_attempt_at=attempted,
        next_poll_at=next_attempt,
        last_error=None if not blocked else "vision-usage-unavailable",
        decision_blocked=blocked,
    )


def read_usage(
    client: VisionClient, *, now: float
) -> dict[str, tuple[str, UsageEntry]]:
    """Return only a complete, monotonically paginated Claude observation set."""
    result = {}
    cursor = None
    while True:
        query = {"provider": "claude", "limit": "100"}
        if cursor:
            query["after"] = cursor
        page = client.request(
            "GET", "/api/ai-accounts/usage?" + urllib.parse.urlencode(query)
        )
        if (
            not isinstance(page, dict)
            or set(page) != {"accounts", "next_cursor"}
            or not isinstance(page["accounts"], list)
            or len(page["accounts"]) > 100
        ):
            raise VisionError("service_unavailable")
        previous = cursor
        for row in page["accounts"]:
            if not isinstance(row, dict) or row.get("provider") != "claude":
                raise VisionError("service_unavailable")
            login = registry_id(row.get("login_id"), "ail_")
            account = registry_id(row.get("account_id"), "aia_")
            if previous is not None and login <= previous:
                raise VisionError("service_unavailable")
            result[login] = (account, project_status(row.get("status"), now=now))
            previous = login
        following = page["next_cursor"]
        if following is None:
            return result
        registry_id(following, "ail_")
        if (
            not page["accounts"]
            or following != previous
            or (cursor and following <= cursor)
        ):
            raise VisionError("service_unavailable")
        cursor = following


def collect_usage(
    records: dict[str, dict], client: VisionClient | None
) -> dict[str, UsageEntry]:
    unavailable = UsageEntry(
        last_error="vision-usage-unavailable", decision_blocked=True
    )
    result = {number: unavailable for number in records}
    if client is None:
        return result
    try:
        observations = read_usage(client, now=time.time())
    except VisionError:
        return result
    for number, record in records.items():
        match = observations.get(record.get("visionLoginId"))
        if (
            record.get("visionUrl") == client.url
            and match is not None
            and match[0] == record.get("visionAccountId")
        ):
            result[number] = match[1]
    return result
