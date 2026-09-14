"""Durable ownership transfer for dedicated swap-managed Claude login profiles.

A caller holds this lease across native login and registration. Existing default
profiles and portable backups require separate inventory of their refresh copies;
they cannot be adopted through this dedicated-profile API.
"""

from __future__ import annotations

import json
import os
import secrets
import stat
import tempfile
import uuid
from pathlib import Path

from claude_swap import macos_keychain
from claude_swap.exceptions import SessionError
from claude_swap.locking import FileLock
from claude_swap.models import Platform
from claude_swap.session import keychain_service_name, profile_is_quiescent
from claude_swap.vision_registration import (
    RegistrationClient,
    registration_proof,
    registration_receipt,
)

MAX_BYTES = 1_048_576


def _sync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_private(path: Path) -> str | None:
    if path.is_symlink():
        raise SessionError("A handoff file cannot be a symbolic link.")
    try:
        fd = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        )
    except FileNotFoundError:
        return None
    with os.fdopen(fd, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_BYTES:
            raise SessionError("Handoff storage must be a bounded regular file.")
        if os.name != "nt" and (
            metadata.st_uid != os.getuid() or metadata.st_mode & 0o077
        ):
            raise SessionError("Handoff storage must be private and owned by you.")
        value = stream.read(MAX_BYTES + 1)
        if len(value) > MAX_BYTES:
            raise SessionError("Handoff storage is too large.")
    try:
        return value.decode("utf-8")
    except UnicodeError:
        raise SessionError("Handoff storage needs repair.") from None


def _write_private(path: Path, value: str) -> None:
    raw = value.encode("utf-8")
    if len(raw) > MAX_BYTES or path.is_symlink():
        raise SessionError("Invalid handoff storage destination or size.")
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".handoff-")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _credential(material: str) -> dict[str, str]:
    try:
        value = json.loads(material)
        oauth = value["claudeAiOauth"]
        result = {name: oauth[name] for name in ("accessToken", "refreshToken")}
        if any(
            not isinstance(v, str)
            or not v
            or len(v) > 65536
            or any(c in v for c in "\r\n\0")
            for v in result.values()
        ):
            raise ValueError()
        return result
    except (ValueError, TypeError, KeyError):
        raise SessionError(
            "The managed profile does not contain a valid Claude login grant."
        ) from None


class ManagedLoginHandoff:
    def __init__(
        self, backup_dir: Path, profile_id: str, registry: RegistrationClient | None
    ):
        try:
            if str(uuid.UUID(profile_id)) != profile_id:
                raise ValueError()
        except (ValueError, TypeError, AttributeError):
            raise SessionError("Invalid managed login profile ID.") from None
        self.root = backup_dir / "vision-logins"
        self.profile = self.root / profile_id
        self.registry = registry
        self.journal_path = self.profile / ".vision-handoff.json"
        self.auth_path = self.profile / ".credentials.json"
        self.lock = FileLock(self.profile / ".vision-auth.lock")
        self.held = False

    def __enter__(self):
        for path in (self.root, self.profile):
            if path.is_symlink() or (path.exists() and not path.is_dir()):
                raise SessionError("Managed login storage must use real directories.")
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
            if os.name != "nt" and path.stat().st_mode & 0o077:
                raise SessionError("Managed login storage must be private.")
        self.lock.__enter__()
        self.held = True
        return self

    def __exit__(self, *args):
        self.held = False
        self.lock.__exit__(*args)

    def _require_quiescent(self):
        if not self.held:
            raise SessionError("Managed login handoff requires its profile lock.")
        if not profile_is_quiescent(self.profile):
            raise SessionError(
                "Exit the native Claude session before handing off its login."
            )

    def _stores(self) -> dict[str, str | None]:
        keychain = None
        if Platform.detect() == Platform.MACOS:
            keychain = macos_keychain.get_password(
                keychain_service_name(self.profile),
                macos_keychain.keychain_account_name(),
            )
            if keychain is not None and len(keychain.encode()) > MAX_BYTES:
                raise SessionError("Managed Keychain credential is too large.")
        return {"keychain": keychain, "file": _read_private(self.auth_path)}

    def _journal(self):
        raw = _read_private(self.journal_path)
        if raw is None:
            return None
        try:
            value = json.loads(raw)
            if (
                set(value)
                != {"version", "url", "request_id", "proof", "stores", "receipt"}
                or type(value["version"]) is not int
                or value["version"] != 1
                or not isinstance(value["url"], str)
                or (
                    self.registry is not None
                    and value["url"] != self.registry.client.url
                )
            ):
                raise ValueError()
            registration_proof(value["request_id"], value["proof"])
            stores = value["stores"]
            if (
                not isinstance(stores, dict)
                or set(stores) != {"keychain", "file"}
                or any(
                    v is not None and not isinstance(v, str) for v in stores.values()
                )
            ):
                raise ValueError()
            if value["receipt"] is not None:
                registration_receipt(value["receipt"], value["request_id"])
            return value
        except (ValueError, TypeError, KeyError):
            raise SessionError(
                "The handoff journal needs reconciliation at its original registry."
            ) from None

    def _save(self, journal):
        if journal["receipt"] is not None:
            registration_receipt(journal["receipt"], journal["request_id"])
        _write_private(self.journal_path, json.dumps(journal, allow_nan=False))

    def _fence(self, journal):
        self._require_quiescent()
        current = self._stores()
        for name, material in current.items():
            if material is not None and material != journal["stores"][name]:
                raise SessionError(
                    "The native login changed; reconcile the pending handoff first."
                )
        # The journal already durably holds both backend copies. Missing copies
        # here mean an interrupted fence, not permission to regenerate a grant.
        if current["keychain"] is not None:
            macos_keychain.delete_password(
                keychain_service_name(self.profile),
                macos_keychain.keychain_account_name(),
            )
        self.auth_path.unlink(missing_ok=True)
        _sync_directory(self.profile)
        self._require_quiescent()
        if any(value is not None for value in self._stores().values()):
            raise SessionError("Native credential storage changed during handoff.")

    def _finish(self, journal):
        receipt = journal["receipt"]
        if receipt["state"] == "committed":
            self._require_quiescent()
            current = self._stores()
            old = journal["stores"]
            # A lost confirm reply may be recovered after another login. Retire
            # only the old grant's exact copies; never delete the newer login.
            if (
                current["keychain"] is not None
                and current["keychain"] == old["keychain"]
            ):
                macos_keychain.delete_password(
                    keychain_service_name(self.profile),
                    macos_keychain.keychain_account_name(),
                )
            if current["file"] is not None and current["file"] == old["file"]:
                self.auth_path.unlink()
                _sync_directory(self.profile)
            self._require_quiescent()
            journal["stores"] = {"keychain": None, "file": None}
            self._save(journal)
        elif receipt["state"] in {"cancelled", "expired"}:
            self._require_quiescent()
            current = self._stores()
            # A newer local login must never be overwritten by an old rollback.
            if any(
                material is not None and material != journal["stores"][name]
                for name, material in current.items()
            ):
                return
            old = journal["stores"]
            if old["keychain"] is not None and current["keychain"] is None:
                macos_keychain.set_password(
                    keychain_service_name(self.profile),
                    macos_keychain.keychain_account_name(),
                    old["keychain"],
                )
            if old["file"] is not None and current["file"] is None:
                _write_private(self.auth_path, old["file"])

    def prepare_login(self):
        """Do not overwrite an unresolved ownership transaction with a new login."""
        self._require_quiescent()
        journal = self._journal()
        if journal is None:
            return
        receipt = journal["receipt"]
        if receipt is None or receipt["state"] not in {
            "committed",
            "cancelled",
            "expired",
        }:
            raise SessionError(
                "Recover or cancel the pending upload before logging in again."
            )
        self._finish(journal)

    def upload(self):
        if self.registry is None:
            raise SessionError("Sign in to Vision before uploading this login.")
        self._require_quiescent()
        journal = self._journal()
        if journal is not None and journal["receipt"] is not None:
            state = journal["receipt"]["state"]
            if state == "committed":
                if any(value is not None for value in journal["stores"].values()):
                    self._finish(journal)
                if not any(value is not None for value in self._stores().values()):
                    return journal["receipt"]
                # A new native login is a new identity-bearing request, even if
                # the local profile name was reused.
                journal = None
            elif state in {"cancelled", "expired"}:
                self._finish(journal)
                journal = None
        if journal is None:
            stores = self._stores()
            material = stores["keychain"] or stores["file"]
            if material is None:
                raise SessionError("No managed Claude login is available to upload.")
            _credential(material)
            journal = {
                "version": 1,
                "url": self.registry.client.url,
                "request_id": str(uuid.uuid4()),
                "proof": secrets.token_urlsafe(32),
                "stores": stores,
                "receipt": None,
            }
            self._save(journal)
        request, proof = journal["request_id"], journal["proof"]
        receipt = journal["receipt"]
        if receipt is None or receipt["state"] == "preparing":
            if self._stores() != journal["stores"]:
                raise SessionError("The login changed before registration completed.")
            material = journal["stores"]["keychain"] or journal["stores"]["file"]
            receipt = self.registry.prepare_registration(
                request, proof, _credential(material)
            )
        else:
            receipt = self.registry.registration_status(request, proof)
        journal["receipt"] = receipt
        self._save(journal)
        if receipt["state"] == "pending_handoff":
            self._fence(journal)
            journal["receipt"] = self.registry.confirm_registration(
                request, proof, local_refreshers_stopped=True
            )
            self._save(journal)
        self._finish(journal)
        return journal["receipt"]

    def cancel(self):
        if self.registry is None:
            raise SessionError("Sign in to Vision before cancelling this upload.")
        self._require_quiescent()
        journal = self._journal()
        if journal is None:
            raise SessionError("No managed login handoff is pending.")
        journal["receipt"] = self.registry.cancel_registration(
            journal["request_id"], journal["proof"]
        )
        self._save(journal)
        self._finish(journal)
        return journal["receipt"]
