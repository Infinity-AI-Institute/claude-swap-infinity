"""Recoverable transfer of an explicitly selected existing Claude refresh grant.

The transaction preserves exact backend bytes until the registry confirms its
terminal state. It never uses cached email/alias metadata as registration identity.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import shutil
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from pathlib import Path

from claude_swap import macos_keychain
from claude_swap.claude_locks import claude_credentials_lock
from claude_swap.credentials import SECURITY_SERVICE
from claude_swap.exceptions import SessionError
from claude_swap.locking import FileLock
from claude_swap.session import scan_live_sessions
from claude_swap.vision_handoff import _read_private, _sync_directory, _write_private
from claude_swap.vision_inventory import CredentialSource, _decode, capture_inventory
from claude_swap.vision_registration import registration_proof, registration_receipt
from claude_swap.vision_registry import merge_accounts


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
        self.history_path = self.root / (request_id + ".history")

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
            fields = {
                "version",
                "request_id",
                "proof",
                "url",
                "extra_profiles",
                "copies",
                "credential",
                "receipt",
                "roster",
                "local_slots",
                "routing_complete",
                "history_sources",
                "history_snapshotted",
            }
            if type(value["version"]) is not int or value["version"] not in {1, 2}:
                raise ValueError()
            if value["version"] == 2:
                fields.add("affected_profiles")
            if set(value) != fields:
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
            if (
                not isinstance(value["local_slots"], dict)
                or type(value["routing_complete"]) is not bool
            ):
                raise ValueError()
            if not isinstance(value["history_sources"], list) or any(
                not isinstance(path, str) or not Path(path).is_absolute()
                for path in value["history_sources"]
            ):
                raise ValueError()
            if type(value["history_snapshotted"]) is not bool:
                raise ValueError()
            known = self._capture()
            if value["version"] == 2:
                affected = value["affected_profiles"]
                known_profiles = set(known.profiles) | {
                    self.switcher._session_dir(number, record["email"])
                    for number, record in value["local_slots"].items()
                }
                if (
                    not isinstance(affected, list)
                    or len(affected) > 1000
                    or any(not isinstance(path, str) for path in affected)
                    or len(set(affected)) != len(affected)
                    or not set(map(Path, affected)).issubset(known_profiles)
                ):
                    raise ValueError()
                required = self._profiles_for_sources(
                    known,
                    [CredentialSource(**item["source"]) for item in value["copies"]],
                    value["local_slots"],
                )
                if not required.issubset(set(map(Path, affected))):
                    if (self.switcher._get_sequence_data() or {}) != value["roster"]:
                        raise SessionError(
                            "The account roster changed; reconcile the saved writer scope before recovery."
                        )
                    raise ValueError()
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

    def _profiles_for_sources(self, inventory, sources, local_slots):
        """Every native home or saved-slot home that can serve these copies."""
        source_ids = {source.id for source in sources}
        profiles = {
            profile
            for source_id, profile in inventory.profile_sources.items()
            if source_id in source_ids
        }
        if profiles:
            # A native credential home may be an arbitrary secure-storage
            # override for a process whose session records live elsewhere.
            # The migration shell does not prove that process's environment.
            # Without a durable/process-level association, fence every known
            # native home. Only saved-backup-only grants permit narrower scope.
            profiles.update(inventory.profiles)
        roster = self.switcher._get_sequence_data() or {}
        for number, record in roster.get("accounts", {}).items():
            if record.get("source") == "vision":
                continue
            email = record["email"]
            slot_sources = []
            for suffix in ("", ".prev"):
                slot_sources.extend(
                    [
                        CredentialSource(
                            "file",
                            str(self.switcher._store._backup_enc_path(number, email))
                            + suffix,
                            encoding="base64",
                        ),
                        CredentialSource(
                            "keychain",
                            SECURITY_SERVICE,
                            f"account-{number}-{email}{suffix}",
                        ),
                    ]
                )
            if number in local_slots or any(
                source.id in source_ids for source in slot_sources
            ):
                profiles.add(self.switcher._session_dir(number, email))
        # Preserve the old slot home even after committed routing removes its row.
        for number, record in local_slots.items():
            profiles.add(self.switcher._session_dir(number, record["email"]))
        return profiles

    def _selected_profiles(self, inventory, source_id):
        selected = next(
            (item for item in inventory.copies if item.source.id == source_id), None
        )
        if selected is None or selected.credential is None:
            raise SessionError(
                "Select a credential from the current migration inventory."
            )
        refresh_token = selected.credential["refreshToken"]
        matching = [
            item.source
            for item in inventory.copies
            if item.credential is not None
            and item.credential["refreshToken"] == refresh_token
        ]
        return self._profiles_for_sources(
            inventory, matching, self._local_slots(inventory, refresh_token)
        )

    @staticmethod
    def _journal_profiles(journal, inventory):
        # Old journals did not persist their writer scope. Never infer a smaller
        # scope from remaining files after an interrupted deletion.
        if journal["version"] == 1:
            return set(inventory.profiles)
        return set(map(Path, journal["affected_profiles"]))

    @staticmethod
    def _require_quiescent(inventory, affected_profiles):
        if set(inventory.live_profiles) & affected_profiles:
            raise SessionError(
                "Exit native sessions using the selected login before transferring it."
            )
        # A malformed process record cannot prove which grant the process holds.
        # Continue to fail closed even when its profile appears unrelated.
        for profile in set(inventory.profiles) | affected_profiles:
            sessions, unreadable = scan_live_sessions(profile)
            if unreadable:
                raise SessionError(
                    "A native session record is unreadable; repair it before transferring a login."
                )
            if profile in affected_profiles and sessions:
                raise SessionError(
                    "Exit native sessions using the selected login before transferring it."
                )

    @contextmanager
    def _lease(self, source_id=None):
        if self.root.is_symlink():
            raise SessionError("Migration storage cannot be a symbolic link.")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with FileLock(self.root / ".migration.lock"), ExitStack() as locks:
            # Capture names before acquiring consume locks, then revalidate once
            # all locks are held. No provider HTTP runs under the roster lock.
            initial = self._capture()
            journal = self._read()
            affected = (
                self._selected_profiles(initial, source_id)
                if journal is None
                else self._journal_profiles(journal, initial)
            )
            roster = self.switcher._get_sequence_data() or {}
            slots = sorted(roster.get("accounts", {}))
            for number in slots:
                locks.enter_context(
                    FileLock(self.switcher.credentials_dir / f".consume-{number}.lock")
                )
            for profile in sorted(affected):
                if profile.parent == self.switcher.backup_dir / "vision-logins":
                    locks.enter_context(FileLock(profile / ".vision-auth.lock"))
                if profile.is_dir():
                    locks.enter_context(claude_credentials_lock(config_home=profile))
            current = self._capture()
            self._require_quiescent(current, affected)
            if (
                _fingerprint(initial) != _fingerprint(current)
                or initial.profiles != current.profiles
            ):
                raise SessionError(
                    "Credential stores changed while acquiring migration locks; retry."
                )
            yield current, affected

    def _local_slots(self, inventory, refresh_token):
        """Bind only slots whose currently served backup is the selected grant."""
        copies = {item.source.id: item for item in inventory.copies}
        result = {}
        roster = self.switcher._get_sequence_data() or {}
        for number, record in roster.get("accounts", {}).items():
            if record.get("source") == "vision":
                continue
            email = record["email"]
            file_source = CredentialSource(
                "file",
                str(self.switcher._store._backup_enc_path(number, email)),
                encoding="base64",
            )
            keychain_source = CredentialSource(
                "keychain", SECURITY_SERVICE, f"account-{number}-{email}"
            )
            file_copy = copies.get(file_source.id)
            keychain_copy = copies.get(keychain_source.id)
            served = (
                file_copy
                if file_copy is not None and file_copy.raw is not None
                else keychain_copy
            )
            if (
                served is not None
                and served.credential is not None
                and served.credential["refreshToken"] == refresh_token
            ):
                result[number] = record
        return result

    def route_committed(self):
        """Convert migrated slots to one verified remote row, preserving aliases."""
        with self._lease():
            journal = self._read()
            if (
                journal is None
                or journal["receipt"] is None
                or journal["receipt"]["state"] != "committed"
            ):
                raise SessionError(
                    "Central ownership must commit before changing local account routes."
                )
            if journal["routing_complete"]:
                if self.history_path.exists():
                    shutil.rmtree(self.history_path)
                return journal["receipt"]
            receipt = journal["receipt"]
            items = self.registry.client.discover()
            target = next(
                (
                    item
                    for item in items
                    if item["login_id"] == receipt["login_id"]
                    and item["account_id"] == receipt["account_id"]
                ),
                None,
            )
            if target is not None and (
                target["email"] != receipt["email"]
                or target["organization_id"] != receipt["organization_id"]
            ):
                raise SessionError(
                    "The committed login identity changed; refresh registry discovery before routing."
                )
            if target is None:
                raise SessionError(
                    "The committed login is not currently authorized; recover routing after access is restored."
                )
            with FileLock(self.switcher.lock_file):
                current = self.switcher._get_sequence_data() or {}
                accounts = current.get("accounts", {})
                for number, original in journal["local_slots"].items():
                    row = accounts.get(number)
                    if row is not None and row != original:
                        raise SessionError(
                            "A migrated slot was reassigned; reconcile its local route before retrying."
                        )
                prepared = dict(current)
                prepared["visionLastSlot"] = max(
                    current.get("visionLastSlot", 0),
                    *(int(number) for number in accounts),
                    0,
                )
                prepared["accounts"] = {
                    number: row
                    for number, row in accounts.items()
                    if number not in journal["local_slots"]
                }
                merged = merge_accounts(prepared, items, self.registry.client.url)
                remote = next(
                    row
                    for row in merged["accounts"].values()
                    if row.get("visionLoginId") == receipt["login_id"]
                    and row.get("visionUrl") == self.registry.client.url
                )
                other_names = {
                    name
                    for row in merged["accounts"].values()
                    if row is not remote
                    for name in self.switcher._account_aliases(row)
                }
                previous_remote = next(
                    (
                        row
                        for row in accounts.values()
                        if row.get("visionLoginId") == receipt["login_id"]
                        and row.get("visionUrl") == self.registry.client.url
                    ),
                    None,
                )
                aliases = set(remote.get("visionMigratedAliases", []))
                if previous_remote is not None:
                    aliases.update(self.switcher._account_aliases(previous_remote))
                for row in journal["local_slots"].values():
                    alias = row.get("alias")
                    if alias:
                        if alias in other_names:
                            raise SessionError(
                                "A migrated alias is now used by another account; reconcile it before retrying."
                            )
                        aliases.add(alias)
                if journal["local_slots"]:
                    remote["disabled"] = remote.get("disabled", False) or all(
                        row.get("disabled", False)
                        for row in journal["local_slots"].values()
                    )
                remote["visionMigratedAliases"] = sorted(aliases)
                if previous_remote is not None and previous_remote.get("alias"):
                    remote["alias"] = previous_remote["alias"]
                elif aliases:
                    remote["alias"] = min(aliases)
                self.switcher._write_json(self.switcher.sequence_file, merged)
            journal["routing_complete"] = True
            self._save(journal)
            if self.history_path.exists():
                shutil.rmtree(self.history_path)
            return receipt

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

    @staticmethod
    def _require_known_copies(journal, inventory):
        known = {CredentialSource(**item["source"]).id for item in journal["copies"]}
        refresh_token = journal["credential"]["refreshToken"]
        if any(
            item.credential is not None
            and item.credential["refreshToken"] == refresh_token
            and item.source.id not in known
            for item in inventory.copies
        ):
            raise SessionError(
                "Another refresh copy appeared; cancel and repeat inventory."
            )

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
        current = self._capture()
        self._require_quiescent(current, self._journal_profiles(journal, current))
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
        with self._lease(source_id) as (inventory, affected):
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
                local_slots = self._local_slots(
                    inventory, selected.credential["refreshToken"]
                )
                journal = {
                    "version": 2,
                    "affected_profiles": sorted(map(str, affected)),
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
                    "local_slots": local_slots,
                    "routing_complete": False,
                    # Retained for recovery of version-1 journals; native
                    # sessions stay in their original home and are never copied.
                    "history_sources": [],
                    "history_snapshotted": False,
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
                    self._require_quiescent(
                        current, self._journal_profiles(journal, current)
                    )
                    if current.profiles != inventory.profiles:
                        raise SessionError(
                            "Known native profiles changed during migration; repeat inventory."
                        )
                    self._require_known_copies(journal, current)
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
                    after_delete = self._capture()
                    if after_delete.profiles != inventory.profiles:
                        raise SessionError(
                            "Known native profiles changed during migration; repeat inventory."
                        )
                    self._require_known_copies(journal, after_delete)
                    self._require_quiescent(
                        after_delete, self._journal_profiles(journal, after_delete)
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
