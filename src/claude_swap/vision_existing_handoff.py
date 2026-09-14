"""Recoverable transfer of an explicitly selected existing Claude refresh grant.

The transaction preserves exact backend bytes until the registry confirms its
terminal state. It never uses cached email/alias metadata as registration identity.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from pathlib import Path

from claude_swap import macos_keychain
from claude_swap.claude_locks import claude_credentials_lock
from claude_swap.exceptions import SessionError
from claude_swap.locking import FileLock
from claude_swap.vision_handoff import _read_private, _sync_directory, _write_private
from claude_swap.vision_inventory import CredentialSource, _decode, capture_inventory
from claude_swap.vision_registration import registration_proof, registration_receipt


def _fingerprint(inventory):
    return hashlib.sha256(
        json.dumps(
            [[item.source.id, item.raw] for item in inventory.copies], sort_keys=True
        ).encode()
    ).hexdigest()


class ExistingLoginHandoff:
    def __init__(self, switcher, registry, request_id, extra_profiles=()):
        registration_proof(request_id, "a" * 43)
        self.switcher = switcher
        self.registry = registry
        self.request_id = request_id
        self.extra_profiles = tuple(str(path) for path in extra_profiles)
        self.root = switcher.backup_dir / "vision-migrations"
        self.path = self.root / (request_id + ".json")

    @classmethod
    def new(cls, switcher, registry, extra_profiles=()):
        return cls(switcher, registry, str(uuid.uuid4()), extra_profiles)

    def _capture(self):
        return capture_inventory(self.switcher, self.extra_profiles)

    def _save(self, value):
        if value["receipt"] is not None:
            registration_receipt(value["receipt"], self.request_id)
        _write_private(self.path, json.dumps(value, allow_nan=False))

    def _read(self):
        raw = _read_private(self.path)
        if raw is None:
            return None
        try:
            value = json.loads(raw)
            if set(value) != {
                "version",
                "request_id",
                "proof",
                "url",
                "extra_profiles",
                "copies",
                "credential",
                "receipt",
                "roster",
            }:
                raise ValueError()
            if type(value["version"]) is not int or value["version"] != 1:
                raise ValueError()
            registration_proof(value["request_id"], value["proof"])
            if (
                value["request_id"] != self.request_id
                or value["url"] != self.registry.client.url
            ):
                raise ValueError()
            if value["extra_profiles"] != list(self.extra_profiles):
                raise ValueError()
            if not isinstance(value["copies"], list) or len(value["copies"]) > 1000:
                raise ValueError()
            known = self._capture()
            known_ids = {item.source.id for item in known.copies}
            native_files = {profile / ".credentials.json" for profile in known.profiles}
            for item in value["copies"]:
                source = CredentialSource(**item["source"])
                if source.backend not in {
                    "file",
                    "keychain",
                } or source.encoding not in {"plain", "base64"}:
                    raise ValueError()
                if not all(
                    isinstance(part, str) for part in (source.location, source.account)
                ):
                    raise TypeError()
                if source.backend == "file":
                    path = Path(source.location)
                    if not path.is_absolute() or any(
                        parent.is_symlink() for parent in (path, *path.parents)
                    ):
                        raise ValueError()
                    native = path in native_files and source.encoding == "plain"
                    backup = (
                        path.parent == self.switcher.credentials_dir
                        and source.encoding == "base64"
                        and (
                            (
                                path.name.startswith(".creds-")
                                and path.name.endswith((".enc", ".enc.prev"))
                            )
                            or (
                                path.name.startswith(".unclaimed-")
                                and path.name.endswith(".enc")
                            )
                            or path.name.endswith(".tmp")
                        )
                    )
                    if not (native or backup) or source.account:
                        raise ValueError()
                elif source.id not in known_ids or source.encoding != "plain":
                    raise ValueError()
                if not isinstance(item["raw"], str):
                    raise TypeError()
                if _decode(source, item["raw"]) != value["credential"]:
                    # Copies may have different access generations, but must
                    # represent exactly the selected refresh grant.
                    decoded = _decode(source, item["raw"])
                    if (
                        decoded is None
                        or decoded["refreshToken"]
                        != value["credential"]["refreshToken"]
                    ):
                        raise ValueError()
            if value["receipt"] is not None:
                registration_receipt(value["receipt"], self.request_id)
            return value
        except (ValueError, KeyError, TypeError, AttributeError):
            raise SessionError(
                "The existing-login transfer journal needs repair."
            ) from None

    @contextmanager
    def _lease(self):
        if self.root.is_symlink():
            raise SessionError("Migration storage cannot be a symbolic link.")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with FileLock(self.root / ".migration.lock"), ExitStack() as locks:
            # Capture names before acquiring consume locks, then revalidate once
            # all locks are held. No provider HTTP runs under the roster lock.
            initial = self._capture()
            roster = self.switcher._get_sequence_data() or {}
            slots = sorted(roster.get("accounts", {}))
            for number in slots:
                locks.enter_context(
                    FileLock(self.switcher.credentials_dir / f".consume-{number}.lock")
                )
            for profile in initial.profiles:
                if profile.parent == self.switcher.backup_dir / "vision-logins":
                    locks.enter_context(FileLock(profile / ".vision-auth.lock"))
                if profile.is_dir():
                    locks.enter_context(claude_credentials_lock(config_home=profile))
            current = self._capture()
            if current.live_profiles:
                raise SessionError(
                    "Exit the native sessions before transferring an existing login."
                )
            if (
                _fingerprint(initial) != _fingerprint(current)
                or initial.profiles != current.profiles
            ):
                raise SessionError(
                    "Credential stores changed while acquiring migration locks; retry."
                )
            yield current

    def preview(self, source_id):
        inventory = self._capture()
        item = next(
            (item for item in inventory.copies if item.source.id == source_id), None
        )
        if item is None or item.credential is None:
            raise SessionError("Select an existing refresh credential source.")
        binding = hashlib.sha256(
            json.dumps(
                [
                    self.request_id,
                    self.registry.client.url,
                    hashlib.sha256(self.registry.client.api_key.encode()).hexdigest(),
                    source_id,
                    _fingerprint(inventory),
                ]
            ).encode()
        ).hexdigest()
        public = inventory.public()
        return {
            "request_id": self.request_id,
            "confirmation": binding,
            "selected": next(
                row for row in public["candidates"] if row["source_id"] == source_id
            ),
            "live_or_unreadable_profiles": public["live_or_unreadable_profiles"],
        }

    def _delete(self, source):
        if source.backend == "file":
            path = Path(source.location)
            if path.is_symlink():
                raise SessionError("Migration source became a symbolic link.")
            path.unlink(missing_ok=True)
            _sync_directory(path.parent)
        else:
            try:
                macos_keychain.delete_password(source.location, source.account)
            except macos_keychain.KEYCHAIN_ERRORS:
                raise SessionError(
                    "Could not retire a migration Keychain copy."
                ) from None

    def _restore(self, source, raw):
        if source.backend == "file":
            _write_private(Path(source.location), raw)
        else:
            try:
                macos_keychain.set_password(source.location, source.account, raw)
            except macos_keychain.KEYCHAIN_ERRORS:
                raise SessionError(
                    "Could not restore a migration Keychain copy."
                ) from None

    def _finish(self, journal):
        state = journal["receipt"]["state"]
        if state not in {"committed", "cancelled", "expired"}:
            return
        if (
            state != "committed"
            and (self.switcher._get_sequence_data() or {}) != journal["roster"]
        ):
            raise SessionError(
                "The account roster changed; reconcile slot ownership before restoring credentials."
            )
        if self._capture().live_profiles:
            raise SessionError(
                "Exit native sessions before reconciling credential ownership."
            )
        current_copies = [
            (item, CredentialSource(**item["source"]).read())
            for item in journal["copies"]
        ]
        newer_login = any(
            current is not None and current != item["raw"]
            for item, current in current_copies
        )
        for item, current in current_copies:
            source = CredentialSource(**item["source"])
            if state == "committed":
                if current == item["raw"]:
                    self._delete(source)
            elif current is None and not newer_login:
                self._restore(source, item["raw"])
                if source.read() != item["raw"]:
                    raise SessionError(
                        "A restored credential did not persist; retry cancellation recovery."
                    )
        # Retire escrow only after every backend operation succeeded. Newer
        # local material is preserved in both commit and rollback recovery.
        journal["copies"] = []
        journal["credential"] = None
        self._save(journal)

    def upload(self, source_id=None, confirmation=None):
        with self._lease() as inventory:
            journal = self._read()
            if journal is None:
                expected = self.preview(source_id)["confirmation"]
                if not isinstance(confirmation, str) or not secrets.compare_digest(
                    expected, confirmation
                ):
                    raise SessionError(
                        "The selected credential preview changed; inspect it again."
                    )
                selected = next(
                    item for item in inventory.copies if item.source.id == source_id
                )
                matching = [
                    item
                    for item in inventory.copies
                    if item.credential is not None
                    and item.credential["refreshToken"]
                    == selected.credential["refreshToken"]
                ]
                journal = {
                    "version": 1,
                    "request_id": self.request_id,
                    "proof": secrets.token_urlsafe(32),
                    "url": self.registry.client.url,
                    "extra_profiles": list(self.extra_profiles),
                    "copies": [
                        {"source": asdict(item.source), "raw": item.raw}
                        for item in matching
                    ],
                    "credential": selected.credential,
                    "receipt": None,
                    "roster": self.switcher._get_sequence_data() or {},
                }
                self._save(journal)
                journal = self._read()
            elif journal["receipt"] is not None and journal["receipt"]["state"] in {
                "committed",
                "cancelled",
                "expired",
            }:
                self._finish(journal)
                return journal["receipt"]
            proof = journal["proof"]
            if journal["receipt"] is None or journal["receipt"]["state"] == "preparing":
                for item in journal["copies"]:
                    if CredentialSource(**item["source"]).read() != item["raw"]:
                        raise SessionError(
                            "A selected login changed before registration; cancel this transfer."
                        )
                journal["receipt"] = self.registry.prepare_registration(
                    self.request_id, proof, journal["credential"]
                )
            else:
                journal["receipt"] = self.registry.registration_status(
                    self.request_id, proof
                )
            self._save(journal)
            if journal["receipt"]["state"] == "pending_handoff":
                with FileLock(self.switcher.lock_file):
                    if (self.switcher._get_sequence_data() or {}) != journal["roster"]:
                        raise SessionError(
                            "The account roster changed during migration; reconcile this transfer."
                        )
                    current = self._capture()
                    if current.live_profiles:
                        raise SessionError(
                            "A native session started during migration; exit it before retrying."
                        )
                    known = {
                        CredentialSource(**item["source"]).id
                        for item in journal["copies"]
                    }
                    for item in current.copies:
                        if (
                            item.credential is not None
                            and item.credential["refreshToken"]
                            == journal["credential"]["refreshToken"]
                            and item.source.id not in known
                        ):
                            raise SessionError(
                                "Another refresh copy appeared; cancel and repeat inventory."
                            )
                    for item in journal["copies"]:
                        source = CredentialSource(**item["source"])
                        raw = source.read()
                        if raw is not None and raw != item["raw"]:
                            raise SessionError(
                                "A selected login changed during registration; cancel this transfer."
                            )
                    for item in journal["copies"]:
                        self._delete(CredentialSource(**item["source"]))
                    if any(
                        CredentialSource(**item["source"]).read() is not None
                        for item in journal["copies"]
                    ):
                        raise SessionError(
                            "A native credential copy reappeared during migration."
                        )
                    if self._capture().live_profiles:
                        raise SessionError(
                            "A native session started during migration; exit it before retrying."
                        )
                journal["receipt"] = self.registry.confirm_registration(
                    self.request_id, proof, local_refreshers_stopped=True
                )
                self._save(journal)
            self._finish(journal)
            return journal["receipt"]

    def cancel(self):
        with self._lease():
            journal = self._read()
            if journal is None:
                raise SessionError("No existing-login transfer has this request ID.")
            journal["receipt"] = self.registry.cancel_registration(
                self.request_id, journal["proof"]
            )
            self._save(journal)
            self._finish(journal)
            return journal["receipt"]
