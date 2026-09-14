"""Read-only inventory of known local Claude refresh-credential copies.

This is a migration snapshot, not proof of exclusive ownership. A transfer must
revalidate it under the local consume/native locks and checkpoint live sessions.
Explicit profile paths extend discovery; arbitrary exports outside these stores
are not silently searched or adopted.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from claude_swap import macos_keychain
from claude_swap.credentials import CLAUDE_CODE_KEYCHAIN_SERVICE, SECURITY_SERVICE
from claude_swap.exceptions import SessionError
from claude_swap.models import Platform
from claude_swap.paths import get_default_claude_config_home
from claude_swap.session import keychain_service_name, profile_is_quiescent
from claude_swap.vision_handoff import MAX_BYTES, _credential, _read_private

MAX_SOURCES = 1000


@dataclass(frozen=True)
class CredentialSource:
    backend: str
    location: str
    account: str = ""
    encoding: str = "plain"

    @property
    def id(self):
        return hashlib.sha256(
            json.dumps([self.backend, self.location, self.account]).encode()
        ).hexdigest()

    def read(self):
        if self.backend == "file":
            raw = _read_private(Path(self.location))
        else:
            try:
                raw = macos_keychain.get_password(self.location, self.account)
            except macos_keychain.KEYCHAIN_ERRORS:
                raise SessionError(
                    "A migration Keychain source is unreadable."
                ) from None
        if raw is not None and len(raw.encode()) > MAX_BYTES:
            raise SessionError("A migration credential source is too large.")
        return raw


@dataclass(frozen=True)
class CredentialCopy:
    source: CredentialSource
    raw: str | None = field(repr=False)
    credential: dict[str, str] | None = field(repr=False)


@dataclass(frozen=True)
class Inventory:
    copies: tuple[CredentialCopy, ...] = field(repr=False)
    profiles: tuple[Path, ...]
    live_profiles: tuple[Path, ...]

    def public(self):
        candidates = []
        for item in self.copies:
            if item.credential is None:
                continue
            matching = [
                other.source.id
                for other in self.copies
                if other.credential is not None
                and other.credential["refreshToken"] == item.credential["refreshToken"]
            ]
            candidates.append(
                {
                    "source_id": item.source.id,
                    "backend": item.source.backend,
                    "location": item.source.location,
                    "account": item.source.account,
                    "matching_sources": matching,
                }
            )
        return {
            "state": "inventory_only",
            "sources_checked": len(self.copies),
            "candidates": candidates,
            "live_or_unreadable_profiles": [str(path) for path in self.live_profiles],
        }


def _directory_entries(path):
    # glob/exists can suppress permission errors. Absence is the only condition
    # under which migration is allowed to conclude that a store has no entries.
    if path.is_symlink():
        raise SessionError("Migration directories cannot be symbolic links.")
    try:
        return sorted(path.iterdir())
    except FileNotFoundError:
        return []
    except OSError:
        raise SessionError("A migration directory is unreadable.") from None


def _decode(source, raw):
    if raw is None:
        return None
    try:
        material = raw
        if source.encoding == "base64":
            material = base64.b64decode(raw.strip(), validate=True).decode("utf-8")
        # API keys do not participate in subscription refresh ownership.
        if material.startswith("sk-ant-api"):
            return None
        value = json.loads(material)
        if not isinstance(value, dict):
            raise TypeError()
        oauth = value.get("claudeAiOauth")
        if oauth is None:
            return None
        if not isinstance(oauth, dict):
            raise TypeError()
        refresh = oauth.get("refreshToken")
        if refresh is None or refresh == "":
            return None
        return _credential(material)
    except (ValueError, UnicodeError, TypeError):
        raise SessionError("A migration credential source needs repair.") from None


def capture_inventory(switcher, extra_profiles=()):
    sources = {}
    profiles = set()
    is_macos = Platform.detect() == Platform.MACOS
    username = macos_keychain.keychain_account_name() if is_macos else ""

    def add(source):
        sources[source.id] = source
        if len(sources) > MAX_SOURCES:
            raise SessionError("Too many credential sources for one migration.")

    def profile(path):
        # Native Keychain service names hash the supplied path spelling. Keep
        # that spelling even though file reads use an absolute path.
        native_path = str(path)
        path = Path(path).expanduser().absolute()
        _directory_entries(path)
        journal = _read_private(path / ".vision-handoff.json")
        if journal is not None:
            try:
                transaction = json.loads(journal)
                stores = transaction["stores"]
                if not isinstance(stores, dict) or set(stores) != {"keychain", "file"}:
                    raise ValueError()
                if any(value is not None for value in stores.values()):
                    raise SessionError(
                        "Reconcile the managed handoff before inventorying existing logins."
                    )
            except (ValueError, TypeError, KeyError):
                raise SessionError("A managed handoff journal needs repair.") from None
        profiles.add(path)
        add(CredentialSource("file", str(path / ".credentials.json")))
        if is_macos:
            add(
                CredentialSource(
                    "keychain", keychain_service_name(native_path), username
                )
            )

    default = get_default_claude_config_home()
    profile(default)
    profile(os.environ.get("CLAUDE_CONFIG_DIR") or default)
    secure = os.environ.get("CLAUDE_SECURESTORAGE_CONFIG_DIR")
    if secure:
        profile(secure)
    for path in extra_profiles:
        profile(path)
    if is_macos:
        add(CredentialSource("keychain", CLAUDE_CODE_KEYCHAIN_SERVICE, username))

    # Include orphaned files as well as current roster slots. Previous and
    # stashed generations can still be restored by the legacy recovery code.
    for path in _directory_entries(switcher.credentials_dir):
        if (
            (
                path.name.startswith(".creds-")
                and path.name.endswith((".enc", ".enc.prev"))
            )
            or (path.name.startswith(".unclaimed-") and path.name.endswith(".enc"))
            or path.name.endswith(".tmp")
        ):
            add(CredentialSource("file", str(path), encoding="base64"))

    roster = switcher._get_sequence_data() or {}
    accounts = roster.get("accounts", {})
    if not isinstance(accounts, dict) or any(
        not isinstance(row, dict) for row in accounts.values()
    ):
        raise SessionError("The migration roster needs repair.")
    for number, record in accounts.items():
        if record.get("source") == "vision":
            continue
        email = record.get("email")
        if not isinstance(email, str) or not email or any(c in email for c in "/\\\0"):
            raise SessionError("A migration roster email needs repair.")
        if not str(number).isascii() or not str(number).isdecimal():
            raise SessionError("A migration roster slot needs repair.")
        profile(switcher._session_dir(str(number), email))
        if is_macos:
            for suffix in ("", ".prev"):
                add(
                    CredentialSource(
                        "keychain",
                        SECURITY_SERVICE,
                        f"account-{number}-{email}{suffix}",
                    )
                )

    # Directory scans also find detached native profiles no longer in the roster.
    for name in ("sessions", "vision-logins"):
        for path in _directory_entries(switcher.backup_dir / name):
            if path.is_symlink():
                raise SessionError("A migration profile cannot be a symbolic link.")
            if path.is_dir():
                profile(path)
    for scope in _directory_entries(switcher.backup_dir / "vision-sessions"):
        if scope.is_symlink():
            raise SessionError("A migration scope cannot be a symbolic link.")
        if not scope.is_dir():
            continue
        for path in _directory_entries(scope):
            profile(path)

    copies = []
    for source in sorted(sources.values(), key=lambda source: source.id):
        try:
            raw = source.read()
        except OSError:
            raise SessionError("A migration credential source is unreadable.") from None
        copies.append(CredentialCopy(source, raw, _decode(source, raw)))
    ordered_profiles = tuple(sorted(profiles))
    return Inventory(
        tuple(copies),
        ordered_profiles,
        tuple(path for path in ordered_profiles if not profile_is_quiescent(path)),
    )
