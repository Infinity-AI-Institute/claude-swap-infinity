"""Account slot store and switcher for the non-Claude tools (Codex, Kimi).

Both tools keep their login in a single JSON file under the user's home
directory, so the whole account lifecycle is file swapping — no Keychain, no
OAuth state to merge:

- Codex CLI:   ``$CODEX_HOME/auth.json``        (default ``~/.codex``)
- Kimi Code:   ``$KIMI_CODE_HOME/credentials/kimi-code.json`` (default
  ``~/.kimi-code``)

Each tool gets its own namespace under the cswap backup root
(``<backup_root>/codex/``, ``<backup_root>/kimi/``) so it can never collide
with the Claude account data in the root itself. Per tool the layout is::

    accounts.json           {"activeSlot": 1, "accounts": {"1": {...}}}
    credentials/<slot>.json  verbatim snapshot of the live credential file
    sessions/<slot>/         session-mode home for `cswap <tool> run`

The Claude switcher's machinery (locks, OAuth merging, usage store) is
deliberately not reused: these tools have no lock protocol to honour and
their credential file *is* the whole login.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from claude_swap.exceptions import (
    AccountNotFoundError,
    CredentialReadError,
    CredentialWriteError,
    SessionError,
    ValidationError,
)
from claude_swap.fsutil import replace_with_retry
from claude_swap.models import get_timestamp
from claude_swap.paths import get_backup_root
from claude_swap.printer import accent, dimmed, muted


@dataclass(frozen=True)
class ToolSpec:
    """Static description of one managed CLI tool."""

    key: str  # command namespace + backup subdir: "codex" | "kimi"
    display: str  # human name: "Codex CLI"
    binary: str  # executable resolved on PATH for `run`
    env_home_var: str  # env var that relocates the tool's home (session mode)
    default_home: str  # home-relative default dir: ".codex"
    credential_relpath: tuple[str, ...]  # path of the login file under home
    # Extra files mirrored into a `run` session home so the session inherits
    # the user's settings (not credentials — those come from the slot).
    shared_config_files: tuple[str, ...]


TOOLS: dict[str, ToolSpec] = {
    "codex": ToolSpec(
        key="codex",
        display="Codex CLI",
        binary="codex",
        env_home_var="CODEX_HOME",
        default_home=".codex",
        credential_relpath=("auth.json",),
        shared_config_files=("config.toml",),
    ),
    "kimi": ToolSpec(
        key="kimi",
        display="Kimi Code CLI",
        binary="kimi",
        env_home_var="KIMI_CODE_HOME",
        default_home=".kimi-code",
        credential_relpath=("credentials", "kimi-code.json"),
        shared_config_files=("config.toml", "tui.toml"),
    ),
}


def live_home(spec: ToolSpec) -> Path:
    """The tool's live home directory (env override, else the default)."""
    env = os.environ.get(spec.env_home_var)
    if env:
        return Path(env)
    return Path.home() / spec.default_home


def live_credential_path(spec: ToolSpec) -> Path:
    """The tool's live credential file."""
    return live_home(spec).joinpath(*spec.credential_relpath)


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


def _identity_from_credential(spec: ToolSpec, credential_text: str) -> str:
    """Best-effort human label for a credential snapshot ("" if unknown)."""
    try:
        data = json.loads(credential_text)
    except json.JSONDecodeError:
        return ""
    if spec.key == "codex":
        id_token = (data.get("tokens") or {}).get("id_token") or ""
        claims = _decode_jwt_payload(id_token)
        profile = claims.get("https://api.openai.com/profile") or {}
        return profile.get("email") or claims.get("email") or ""
    if spec.key == "kimi":
        claims = _decode_jwt_payload(data.get("access_token") or "")
        return claims.get("email") or claims.get("sub") or ""
    return ""


def _fingerprint(text: str) -> str:
    """Content hash used to match the live file against stored slots."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ToolAccount:
    """One stored slot for a tool."""

    number: int
    label: str
    added: str
    active: bool


class ToolAccountStore:
    """Slot store + switch logic for one non-Claude tool."""

    def __init__(self, tool: str, backup_root: Path | None = None) -> None:
        if tool not in TOOLS:
            raise ValidationError(f"unknown tool '{tool}' (expected: {', '.join(TOOLS)})")
        self.spec = TOOLS[tool]
        root = backup_root if backup_root is not None else get_backup_root()
        self.root = root / self.spec.key
        self.credentials_dir = self.root / "credentials"
        self.sessions_dir = self.root / "sessions"
        self.accounts_path = self.root / "accounts.json"

    # -- metadata ----------------------------------------------------------

    def _load(self) -> dict:
        try:
            data = json.loads(self.accounts_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"activeSlot": None, "accounts": {}}
        except json.JSONDecodeError:
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

    # -- live credential I/O -----------------------------------------------

    def read_live(self) -> str:
        """Read the tool's live credential file, or raise CredentialReadError."""
        path = live_credential_path(self.spec)
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise CredentialReadError(
                f"No {self.spec.display} login found at {path} — "
                f"log in with '{self.spec.binary}' first."
            ) from None
        except OSError as e:
            raise CredentialReadError(f"Could not read {path}: {e}") from e

    def _write_live(self, content: str) -> None:
        path = live_credential_path(self.spec)
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
        """Which slot the live credential file currently matches, if any."""
        try:
            live = self.read_live()
        except CredentialReadError:
            return None
        live_fp = _fingerprint(live)
        if data is None:
            data = self._load()
        for num in data["accounts"]:
            try:
                if _fingerprint(self._slot_path(int(num)).read_text(encoding="utf-8")) == live_fp:
                    return int(num)
            except OSError:
                continue
        return None

    def resolve(self, target: str) -> int:
        """Resolve a slot number or label to a slot number."""
        data = self._load()
        accounts = data["accounts"]
        if target.isdigit():
            slot = int(target)
            if str(slot) not in accounts:
                raise AccountNotFoundError(
                    f"No {self.spec.key} account in slot {slot}. "
                    f"Use 'cswap {self.spec.key} list' to see stored accounts."
                )
            return slot
        matches = [
            int(num)
            for num, info in accounts.items()
            if info.get("label", "").lower() == target.lower()
        ]
        if not matches:
            raise AccountNotFoundError(
                f"No {self.spec.key} account labelled '{target}'. "
                f"Use 'cswap {self.spec.key} list' to see stored accounts."
            )
        if len(matches) > 1:
            raise ValidationError(
                f"Label '{target}' matches slots {', '.join(map(str, sorted(matches)))} "
                "— use the slot number."
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
            identity = label or _identity_from_credential(self.spec, live)
            label = identity or f"{self.spec.key}-account-{slot}"
            accounts[str(slot)] = {"label": label, "added": get_timestamp()}
        elif label:
            accounts[str(slot)]["label"] = label
        else:
            label = accounts[str(slot)].get("label", "")

        self.credentials_dir.mkdir(parents=True, exist_ok=True)
        temp = self._slot_path(slot).with_suffix(f".{os.getpid()}.tmp")
        temp.write_text(live, encoding="utf-8")
        replace_with_retry(temp, self._slot_path(slot))
        if sys.platform != "win32":
            os.chmod(self._slot_path(slot), 0o600)

        data["activeSlot"] = slot
        self._save(data)
        return slot, label

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

    def list_accounts(self) -> tuple[int | None, list[ToolAccount]]:
        """(live slot, all accounts) — active is detected from the live file."""
        data = self._load()
        live_slot = self._live_slot(data)
        rows = [
            ToolAccount(
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
        data = self._load()
        live_slot = self._live_slot(data)
        if live_slot is None:
            path = live_credential_path(self.spec)
            if not path.exists():
                raise CredentialReadError(
                    f"No {self.spec.display} login found at {path}."
                )
            return None, "unmanaged login (not in any cswap slot)"
        return live_slot, data["accounts"].get(str(live_slot), {}).get("label", "")

    def switch(self, target: str | None = None) -> tuple[int, str]:
        """Restore a slot into the live credential path. Returns (slot, label).

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
                f"No {self.spec.key} accounts stored. Log in with "
                f"'{self.spec.binary}', then run 'cswap {self.spec.key} add'."
            )

        live_slot = self._live_slot(data)
        live_path = live_credential_path(self.spec)
        if live_slot is None and live_path.exists():
            live_slot, live_label = self.add()
            print(
                dimmed(
                    f"Current {self.spec.display} login was not stored — "
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

        content = self._slot_path(slot).read_text(encoding="utf-8")
        self._write_live(content)
        data["activeSlot"] = slot
        self._save(data)
        return slot, accounts[str(slot)].get("label", "")

    # -- session mode ---------------------------------------------------------

    def run_session(self, target: str, tool_args: list[str]) -> NoReturn:
        """Launch the tool as the given account, this terminal only.

        The tool's own home-relocation env var (CODEX_HOME / KIMI_CODE_HOME)
        points at a per-slot session dir holding a copy of the slot's
        credential plus the user's shared config files; the real home is
        never touched.
        """
        binary = shutil.which(self.spec.binary)
        if not binary:
            raise SessionError(
                f"'{self.spec.binary}' was not found on PATH. "
                f"Install {self.spec.display} first."
            )
        slot = self.resolve(target)
        data = self._load()
        label = data["accounts"].get(str(slot), {}).get("label", "")

        session_dir = self.sessions_dir / str(slot)
        session_dir.mkdir(parents=True, exist_ok=True)
        dest = session_dir.joinpath(*self.spec.credential_relpath)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._slot_path(slot), dest)
        # Mirror shared (non-credential) config from the live home so the
        # session behaves like the user's normal setup.
        for name in self.spec.shared_config_files:
            src = live_home(self.spec) / name
            if src.exists():
                try:
                    shutil.copyfile(src, session_dir / name)
                except OSError:
                    pass  # cosmetic; never block a launch on config copying

        print(
            f"{accent('Launching')} {self.spec.display} as "
            f"{muted(f'account {slot} ({label})')} {dimmed('[session mode]')}"
        )
        env = dict(os.environ)
        env[self.spec.env_home_var] = str(session_dir)
        argv = [binary, *tool_args]
        if sys.platform == "win32":
            # os.exec* detaches from the console confusingly on Windows; stay
            # resident as a thin wrapper and mirror the child's exit code
            # (same approach as session.py for claude).
            try:
                rc = subprocess.run(argv, env=env).returncode
            except KeyboardInterrupt:
                rc = 130
            sys.exit(rc)
        os.execvpe(binary, argv, env)
        raise AssertionError("unreachable")  # pragma: no cover
