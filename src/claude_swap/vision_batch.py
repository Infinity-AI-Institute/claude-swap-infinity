"""Selected managed-profile uploads bound to a nonsecret review token."""

import hashlib
import hmac
import json
import re

from claude_swap.exceptions import ClaudeSwitchError, SessionError
from claude_swap.models import normalize_alias
from claude_swap.vision_handoff import ManagedLoginHandoff
from claude_swap.vision_registration import RegistrationClient


class BatchUpload:
    def __init__(self, switcher, client):
        self.switcher = switcher
        self.client = client
        self.registry = RegistrationClient(client)

    def _selection(self, names):
        from claude_swap.vision_cli import ManagedProfiles

        try:
            selected = [normalize_alias(name) for name in names]
        except (TypeError, ValueError):
            raise SessionError("Select valid managed profile names.") from None
        if not 1 <= len(selected) <= 100 or len(set(selected)) != len(selected):
            raise SessionError("Select between 1 and 100 distinct managed profiles.")
        profiles = ManagedProfiles(self.switcher.backup_dir).read()["profiles"]
        items = []
        for name in selected:
            profile_id = profiles.get(name)
            item = {"profile": name, "profile_id": profile_id}
            try:
                if profile_id is None:
                    raise SessionError("No managed login has that name.")
                with ManagedLoginHandoff(
                    self.switcher.backup_dir, profile_id, self.registry
                ) as lease:
                    fingerprint, state = lease.preview()
                item.update(fingerprint=fingerprint, state=state)
            except (ClaudeSwitchError, OSError) as error:
                item.update(
                    state="unavailable",
                    detail=str(error)
                    if isinstance(error, ClaudeSwitchError)
                    else "Profile storage is unavailable.",
                )
            items.append(item)
        return items

    def _confirmation(self, items):
        binding = {
            "version": 1,
            "url": self.client.url,
            "key_fingerprint": hashlib.sha256(self.client.api_key.encode()).hexdigest(),
            "items": items,
        }
        return hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def _public(items):
        return [
            {
                key: value
                for key, value in item.items()
                if key in {"profile", "state", "detail"}
            }
            for item in items
        ]

    def preview(self, names):
        items = self._selection(names)
        return {
            "version": 1,
            "url": self.client.url,
            "items": self._public(items),
            "confirmation": self._confirmation(items),
        }

    def apply(self, names, confirmation):
        from claude_swap.vision_cli import name_committed_login

        items = self._selection(names)
        if (
            not isinstance(confirmation, str)
            or re.fullmatch(r"[0-9a-f]{64}", confirmation) is None
            or not hmac.compare_digest(confirmation, self._confirmation(items))
        ):
            raise SessionError(
                "The selected logins or Vision key changed; review a new batch preview."
            )
        results = []
        for item in items:
            name = item["profile"]
            if item["state"] == "unavailable":
                results.extend(self._public([item]))
                continue
            try:
                with ManagedLoginHandoff(
                    self.switcher.backup_dir, item["profile_id"], self.registry
                ) as lease:
                    fingerprint, _ = lease.preview()
                    if fingerprint != item["fingerprint"]:
                        raise SessionError(
                            "The profile changed after preview; review it again."
                        )
                    receipt = lease.upload()
                    name_committed_login(self.switcher, self.client, name, receipt)
                results.append(
                    {"profile": name, "state": receipt["state"], "receipt": receipt}
                )
            except (ClaudeSwitchError, OSError) as error:
                results.append(
                    {
                        "profile": name,
                        "state": "unavailable",
                        "detail": str(error)
                        if isinstance(error, ClaudeSwitchError)
                        else "Profile storage is unavailable.",
                    }
                )
        return {"version": 1, "items": results}
