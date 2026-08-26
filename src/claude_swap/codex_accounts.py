"""Account slot store for the Codex CLI (``cswap codex ...``).

Codex keeps its whole login in one JSON file — ``$CODEX_HOME/auth.json``
(default ``~/.codex``) — so the account lifecycle is file swapping: no
Keychain, no OAuth state to merge. Ported from the ``multi-tool-runner``
branch's ``tool_switcher.py``, Codex parts only (the Kimi tool and the
``persistent`` runner were deliberately not brought over).

The store gets its own namespace under the cswap backup root
(``<backup_root>/codex/``) so it can never collide with the Claude account
data in the root itself::

    accounts.json            {"activeSlot": 1, "accounts": {"1": {...}}}
    credentials/<slot>.json  verbatim snapshot of the live auth.json

The Claude switcher's machinery (locks, OAuth merging, usage store) is
deliberately not reused: Codex has no lock protocol to honour and its
credential file *is* the whole login. What IS kept is the same transactional
discipline — atomic temp-file + rename writes (0600), and a switch never
overwrites a live login that hasn't been snapshotted first.

**Swapping the file does not rotate a LIVE Codex TUI.** Measured 2026-08-26
(codex 0.147): codex reads ``auth.json`` once at startup and caches the auth
in memory — a running TUI completes turns on the cached credential even after
the on-disk file is replaced (verified with a server-rejected credential on
disk), and only a restart re-reads the file. ``cswap codex switch`` therefore
takes effect on the next codex start; rotating a live session is what
``cswap codex handoff`` (``codex_handoff.py``) exists for.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from claude_swap.exceptions import (
    AccountNotFoundError,
    CredentialReadError,
    CredentialWriteError,
    ValidationError,
)
from claude_swap.fsutil import replace_with_retry
from claude_swap.models import get_timestamp
from claude_swap.paths import get_backup_root
from claude_swap.printer import dimmed

CODEX_BACKUP_SUBDIR = "codex"
CODEX_CREDENTIAL_FILENAME = "auth.json"


def default_codex_home() -> Path:
    """Codex's live home: ``$CODEX_HOME`` if set, else ``~/.codex``.

    Mirrors codex's own resolution. The ``codex.home`` setting overrides both
    — callers that honour settings resolve through
    :func:`resolve_codex_home` and pass the result in explicitly.
    """
    env = os.environ.get("CODEX_HOME")
    if env:
        return Path(env)
    return Path.home() / ".codex"


def resolve_codex_home(configured: str | None) -> Path:
    """The effective Codex home: the ``codex.home`` setting when set,
    else :func:`default_codex_home` (``$CODEX_HOME`` / ``~/.codex``)."""
    if configured:
        return Path(configured).expanduser()
    return default_codex_home()


def _decode_jwt_payload(token: str) -> dict:
    """Decode a JWT's payload without verifying the signature (display only).

    Returns {} for anything that is not a well-formed three-part JWT — the
    payload is used purely to label accounts, never for auth decisions.
    """
    parts = token.split(".")
    if len(parts) != 3:
        return {}
    try:
        padded = parts[1] + "=" * (-len(parts[1]) % 4)
        return json.loads(base64.urlsafe_b64decode(padded))
    except (ValueError, json.JSONDecodeError):
        return {}


def _identity_from_credential(credential_text: str) -> str:
    """Best-effort human label for an auth.json snapshot ("" if unknown)."""
    try:
        data = json.loads(credential_text)
    except json.JSONDecodeError:
        return ""
    if not isinstance(data, dict):
        return ""
    id_token = (data.get("tokens") or {}).get("id_token") or ""
    claims = _decode_jwt_payload(id_token)
    profile = claims.get("https://api.openai.com/profile") or {}
    return profile.get("email") or claims.get("email") or ""


def _fingerprint(text: str) -> str:
    """Content hash used to match the live file against stored slots."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CodexAccount:
    """One stored Codex slot."""

    number: int
    label: str
    added: str
    active: bool


class CodexAccountStore:
    """Slot store + switch logic for Codex CLI logins."""

    def __init__(
        self, backup_root: Path | None = None, home: Path | None = None
    ) -> None:
        root = backup_root if backup_root is not None else get_backup_root()
        self.home = home if home is not None else default_codex_home()
        self.root = root / CODEX_BACKUP_SUBDIR
        self.credentials_dir = self.root / "credentials"
        self.accounts_path = self.root / "accounts.json"

    @property
    def live_credential_path(self) -> Path:
        return self.home / CODEX_CREDENTIAL_FILENAME

    # -- metadata ----------------------------------------------------------

    def _load(self) -> dict:
        try:
            data = json.loads(self.accounts_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {"activeSlot": None, "accounts": {}}
        if not isinstance(data, dict):
            return {"activeSlot": None, "accounts": {}}
        data.setdefault("activeSlot", None)
        data.setdefault("accounts", {})
        return data

    def _save(self, data: dict) -> None:
        data["lastUpdated"] = get_timestamp()
        self.root.mkdir(parents=True, exist_ok=True)
        temp = self.accounts_path.with_suffix(f".{os.getpid()}.tmp")
        temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        replace_with_retry(temp, self.accounts_path)

    def _slot_path(self, slot: int) -> Path:
        return self.credentials_dir / f"{slot}.json"

    def _write_slot(self, slot: int, content: str) -> None:
        self.credentials_dir.mkdir(parents=True, exist_ok=True)
        temp = self._slot_path(slot).with_suffix(f".{os.getpid()}.tmp")
        temp.write_text(content, encoding="utf-8")
        replace_with_retry(temp, self._slot_path(slot))
        if sys.platform != "win32":
            os.chmod(self._slot_path(slot), 0o600)

    # -- live credential I/O -----------------------------------------------

    def read_live(self) -> str:
        """Read the live auth.json, or raise CredentialReadError."""
        path = self.live_credential_path
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise CredentialReadError(
                f"No Codex CLI login found at {path} — "
                "log in with 'codex' first."
            ) from None
        except OSError as e:
            raise CredentialReadError(f"Could not read {path}: {e}") from e

    def _write_live(self, content: str) -> None:
        path = self.live_credential_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            temp.write_text(content, encoding="utf-8")
            replace_with_retry(temp, path)
            if sys.platform != "win32":
                os.chmod(path, 0o600)
        except OSError as e:
            raise CredentialWriteError(f"Could not write {path}: {e}") from e

    # -- identity / resolution ----------------------------------------------

    def _live_slot(self, data: dict | None = None) -> int | None:
        """Which slot the live auth.json currently matches by content, if any.

        Content-hash matching only: codex rotates tokens in place on refresh,
        after which the live file matches no snapshot even though it is still
        the same account. Callers that need "which account is this session
        on" through refreshes use ``active_slot`` (the recorded metadata)
        rather than this.
        """
        try:
            live = self.read_live()
        except CredentialReadError:
            return None
        live_fp = _fingerprint(live)
        if data is None:
            data = self._load()
        for num in data["accounts"]:
            try:
                stored = self._slot_path(int(num)).read_text(encoding="utf-8")
            except OSError:
                continue
            if _fingerprint(stored) == live_fp:
                return int(num)
        return None

    def active_slot(self) -> int | None:
        """The recorded active slot (survives codex's in-place token
        refreshes, unlike content matching), or None if never recorded."""
        slot = self._load().get("activeSlot")
        return int(slot) if isinstance(slot, int) else None

    def _refreshed_active_slot(self, data: dict) -> int | None:
        """The recorded active slot, iff the live file is that same account
        with rotated tokens.

        Codex refreshes tokens in place, so a session's live file soon
        matches no snapshot by content. Falling back to the recorded
        ``activeSlot`` alone would misattribute a genuinely different login
        (the user ran ``codex login`` by hand) to the old account — so the
        fallback additionally requires the live credential's IDENTITY (the
        id_token's email) to match the stored snapshot's. Returns None when
        either identity is unavailable: an unverifiable match is no match.
        """
        recorded = data.get("activeSlot")
        if not isinstance(recorded, int) or str(recorded) not in data["accounts"]:
            return None
        try:
            live_identity = _identity_from_credential(self.read_live())
            stored_identity = _identity_from_credential(
                self._slot_path(recorded).read_text(encoding="utf-8")
            )
        except (CredentialReadError, OSError):
            return None
        if live_identity and live_identity == stored_identity:
            return recorded
        return None

    def resolve(self, target: str) -> int:
        """Resolve a slot number or label to a slot number."""
        data = self._load()
        accounts = data["accounts"]
        if target.isdigit():
            slot = int(target)
            if str(slot) not in accounts:
                raise AccountNotFoundError(
                    f"No codex account in slot {slot}. "
                    "Use 'cswap codex list' to see stored accounts."
                )
            return slot
        matches = [
            int(num)
            for num, info in accounts.items()
            if info.get("label", "").lower() == target.lower()
        ]
        if not matches:
            raise AccountNotFoundError(
                f"No codex account labelled '{target}'. "
                "Use 'cswap codex list' to see stored accounts."
            )
        if len(matches) > 1:
            raise ValidationError(
                f"Label '{target}' matches slots "
                f"{', '.join(map(str, sorted(matches)))} — use the slot number."
            )
        return matches[0]

    # -- commands ------------------------------------------------------------

    def add(self, slot: int | None = None, label: str | None = None) -> tuple[int, str]:
        """Snapshot the live login into a slot. Returns (slot, label)."""
        live = self.read_live()
        data = self._load()
        accounts = data["accounts"]

        if slot is None:
            # Re-adding the same login refreshes its slot instead of
            # duplicating it (mirrors `cswap add`).
            existing = self._live_slot(data)
            if existing is not None:
                slot = existing
            else:
                used = {int(n) for n in accounts}
                slot = 1
                while slot in used:
                    slot += 1
        elif slot < 1:
            raise ValidationError("slot must be a positive integer")

        if str(slot) not in accounts:
            identity = label or _identity_from_credential(live)
            label = identity or f"codex-account-{slot}"
            accounts[str(slot)] = {"label": label, "added": get_timestamp()}
        elif label:
            accounts[str(slot)]["label"] = label
        else:
            label = accounts[str(slot)].get("label", "")

        self._write_slot(slot, live)
        data["activeSlot"] = slot
        self._save(data)
        return slot, label

    def snapshot_slot(self, slot: int) -> None:
        """Refresh ``slot``'s snapshot from the live auth.json, keeping its
        label and WITHOUT re-deriving the slot by content match.

        The handoff path needs this: codex rotates tokens in place, so at
        swap time the live file matches no stored fingerprint and ``add()``
        would file the outgoing login into a brand-new slot. The active slot
        is known from metadata; write the live content back to exactly it.
        """
        data = self._load()
        if str(slot) not in data["accounts"]:
            raise AccountNotFoundError(
                f"No codex account in slot {slot} to snapshot into."
            )
        self._write_slot(slot, self.read_live())
        self._save(data)

    def remove(self, target: str) -> tuple[int, str]:
        """Remove a slot. The live login file is never touched."""
        slot = self.resolve(target)
        data = self._load()
        label = data["accounts"].pop(str(slot), {}).get("label", "")
        self._slot_path(slot).unlink(missing_ok=True)
        if data.get("activeSlot") == slot:
            data["activeSlot"] = self._live_slot(data)
        self._save(data)
        return slot, label

    def list_accounts(self) -> tuple[int | None, list[CodexAccount]]:
        """(live slot, all accounts). Active is content-matched against the
        live file, falling back to the recorded active slot when codex's
        in-place token refresh has moved the file past every snapshot."""
        data = self._load()
        live_slot = self._live_slot(data)
        if live_slot is None:
            live_slot = self._refreshed_active_slot(data)
        rows = [
            CodexAccount(
                number=int(num),
                label=info.get("label", ""),
                added=info.get("added", ""),
                active=int(num) == live_slot,
            )
            for num, info in data["accounts"].items()
        ]
        rows.sort(key=lambda a: a.number)
        return live_slot, rows

    def status(self) -> tuple[int | None, str]:
        """(slot, label) of the account the live login belongs to."""
        live_slot, rows = self.list_accounts()
        if live_slot is None:
            path = self.live_credential_path
            if not path.exists():
                raise CredentialReadError(f"No Codex CLI login found at {path}.")
            return None, "unmanaged login (not in any cswap slot)"
        label = next((r.label for r in rows if r.number == live_slot), "")
        return live_slot, label

    def switch(self, target: str | None = None) -> tuple[int, str]:
        """Restore a slot into the live auth.json. Returns (slot, label).

        Bare ``switch`` rotates to the next slot after the live one. If the
        live login is unknown (never snapshotted) it is first auto-added to a
        free slot, so a switch never silently discards an unbacked-up login —
        the same guarantee the Claude switcher makes before it overwrites the
        active credential.
        """
        data = self._load()
        accounts = data["accounts"]
        if not accounts:
            raise AccountNotFoundError(
                "No codex accounts stored. Log in with 'codex', then run "
                "'cswap codex add'."
            )

        live_slot = self._live_slot(data)
        if live_slot is None and self.live_credential_path.exists():
            # Codex refreshes tokens in place, so an untouched session's live
            # file soon matches no snapshot by content. When the live file is
            # verifiably the recorded active account with rotated tokens,
            # re-snapshot it into that slot; otherwise it is an unknown login
            # and gets its own slot rather than overwriting anyone's.
            refreshed = self._refreshed_active_slot(data)
            if refreshed is not None:
                live_slot = refreshed
                self._write_slot(live_slot, self.read_live())
            else:
                live_slot, live_label = self.add()
                print(
                    dimmed(
                        "Current Codex CLI login was not stored — "
                        f"snapshotted to slot {live_slot} ({live_label})"
                    )
                )
                data = self._load()
                accounts = data["accounts"]

        if target is not None:
            slot = self.resolve(target)
        else:
            ordered = sorted(int(n) for n in accounts)
            if live_slot in ordered:
                slot = ordered[(ordered.index(live_slot) + 1) % len(ordered)]
            else:
                slot = ordered[0]

        if slot == live_slot and target is None:
            # Rotating with a single account: nothing to do.
            return slot, accounts[str(slot)].get("label", "")

        try:
            content = self._slot_path(slot).read_text(encoding="utf-8")
        except OSError as e:
            raise CredentialReadError(
                f"Could not read stored credential for slot {slot}: {e}"
            ) from e
        self._write_live(content)
        data["activeSlot"] = slot
        self._save(data)
        return slot, accounts[str(slot)].get("label", "")
