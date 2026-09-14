"""Launch preparation for Claude's centrally owned, access-only credentials."""

from __future__ import annotations

import hashlib
import os
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path

from claude_swap import macos_keychain
from claude_swap.exceptions import SessionError
from claude_swap.models import Platform
from claude_swap.vision import VisionClient, VisionError, registry_id

# A selected Vision credential must not inherit another provider route or auth
# source from the terminal that invoked the launcher.
ROUTE_OVERRIDES = {
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_CUSTOM_HEADERS",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_SECURESTORAGE_CONFIG_DIR",
    "NODE_TLS_REJECT_UNAUTHORIZED",
}


@dataclass(frozen=True)
class RemoteLaunch:
    directory: Path
    generation: int
    expires_at: str | None
    env: dict[str, str] = field(repr=False)


def acquire_credential(client: VisionClient, record: dict, *, wait_seconds=90):
    """Ask the central owner to recover an unavailable login, never refresh locally."""
    account = registry_id(record.get("visionAccountId"), "aia_")
    login = registry_id(record.get("visionLoginId"), "ail_")
    try:
        return client.credential(account, login)
    except VisionError as error:
        if error.code != "credential_unavailable":
            raise
    generation = record.get("visionGeneration")
    if type(generation) is not int or generation < 1:
        raise VisionError("credential_unavailable")
    receipt = client.refresh(account, login, generation)
    if receipt["state"] == "reauth_required":
        raise SessionError("The Vision login needs reauthentication before launch.")
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            return client.credential(account, login)
        except VisionError as error:
            if error.code != "credential_unavailable":
                raise
            if time.monotonic() >= deadline:
                raise SessionError(
                    "Vision credential recovery is pending; retry the launch."
                ) from None
        time.sleep(min(1, max(0, deadline - time.monotonic())))


def prepare_launch(
    manager,
    record: dict,
    client: VisionClient | None,
    *,
    share: bool,
    share_history: bool,
) -> RemoteLaunch:
    from claude_swap.session import (
        AUTH_OVERRIDE_ENV_VARS,
        _mkdir_private,
        keychain_service_name,
    )

    if client is None or record.get("visionUrl") != client.url:
        raise SessionError(
            "Sign in to the account's Vision registry before launching it."
        )
    if record.get("disabled"):
        raise SessionError("This Vision account is disabled in the local roster.")
    credential = acquire_credential(client, record)
    if credential["email"] != record.get("email") or credential[
        "organization_id"
    ] != record.get("organizationUuid"):
        raise SessionError(
            "Vision account identity changed; synchronize the roster and retry."
        )

    # Identity, rather than alias or transient roster slot, keeps native resume
    # data stable when permissions or local display preferences change.
    scope = hashlib.sha256(client.url.encode()).hexdigest()
    root = manager.switcher.backup_dir / "vision-sessions"
    directory = root / scope / credential["login_id"]
    for path in (root, root / scope, directory):
        if path.is_symlink() or (
            path.exists() and not stat.S_ISDIR(path.stat().st_mode)
        ):
            raise SessionError("The Vision session path must be a real directory.")
    _mkdir_private(directory)
    # Never overwrite or silently adopt credentials created by a native login.
    # That grant needs the explicit ownership handoff before central use resumes.
    auth_path = directory / ".credentials.json"
    if auth_path.exists() or auth_path.is_symlink():
        raise SessionError(
            "This Vision session contains a local login that needs handoff."
        )
    if Platform.detect() == Platform.MACOS:
        try:
            native_login = macos_keychain.get_password(
                keychain_service_name(directory),
                macos_keychain.keychain_account_name(),
            )
        except macos_keychain.KEYCHAIN_ERRORS:
            raise SessionError(
                "The Vision session Keychain entry is unavailable; unlock it and retry."
            ) from None
        if native_login is not None:
            raise SessionError(
                "This Vision session contains a local Keychain login that needs handoff."
            )
    manager._sync_sharing(directory, share, share_history)
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in set(AUTH_OVERRIDE_ENV_VARS) | ROUTE_OVERRIDES
        and key not in {"VISION_API_KEY", "VISION_API_URL"}
    }
    env["CLAUDE_CONFIG_DIR"] = str(directory)
    env["CLAUDE_SECURESTORAGE_CONFIG_DIR"] = str(directory)
    env["CLAUDE_CODE_OAUTH_TOKEN"] = credential["accessToken"]
    return RemoteLaunch(
        directory, credential["generation"], credential["expires_at"], env
    )
