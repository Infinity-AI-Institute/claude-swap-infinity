"""``CredentialStore``: the active-credential read must stay on one profile.

The identity read honors ``CLAUDE_CONFIG_DIR`` (``paths.get_claude_config_home``)
while the Keychain read used a hardcoded service name that does not. Pairing one
profile's identity with another profile's credential is silent, and every
consumer of the active read inherits it.

The fix resolves the Keychain item the way claude does for the same environment
(``session.keychain_service_name``, the same derivation ``delete``, the
session read and capture already use) rather than skipping the Keychain under a
custom profile. Skipping would trade a wrong answer for a missing one: claude
writes rotations keychain-only on macOS, so a custom profile frequently has no
plaintext file at all and would render as "no credentials" while logged in.
"""

from __future__ import annotations

import hashlib
import json
import logging
import unicodedata
from pathlib import Path

import pytest

from claude_swap import macos_keychain
from claude_swap.credentials import (
    CLAUDE_CODE_KEYCHAIN_SERVICE,
    CLAUDE_CODE_MANAGED_KEYCHAIN_SERVICE,
    CredentialStore,
)
from claude_swap.models import Platform
from claude_swap.session import keychain_service_name


class _Host:
    """Minimal ``_StoreHost``: data only, read at call time."""

    def __init__(self, credentials_dir: Path):
        self.platform = Platform.MACOS
        self.credentials_dir = credentials_dir
        self._logger = logging.getLogger("test")


DEFAULT_PROFILE_CREDS = json.dumps({
    "claudeAiOauth": {
        "accessToken": "sk-default-profile",
        "refreshToken": "rt-default-profile",
        "expiresAt": 9999999999000,
    }
})

CUSTOM_PROFILE_CREDS = json.dumps({
    "claudeAiOauth": {
        "accessToken": "sk-custom-profile",
        "refreshToken": "rt-custom-profile",
        "expiresAt": 9999999999000,
    }
})

SECURE_PROFILE_CREDS = json.dumps({
    "claudeAiOauth": {
        "accessToken": "sk-secure-profile",
        "refreshToken": "rt-secure-profile",
        "expiresAt": 9999999999000,
    }
})


def _keychain(mapping: dict[str, str], seen: list[str]):
    """A fake Keychain: only the listed services exist, and record every probe.

    Anything unlisted returns ``None``, which is claude's rc-44 "absent item"
    signal — the case that legitimately falls through to the plaintext file.
    """

    def fake_get_password(service: str, account: str):
        seen.append(service)
        return mapping.get(service)

    return fake_get_password


class TestActiveReadStaysOnOneProfile:
    def test_custom_config_dir_does_not_return_the_default_keychain_item(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The hardcoded ``Claude Code-credentials`` item belongs to the DEFAULT
        profile. Under a custom ``CLAUDE_CONFIG_DIR`` it must never answer, or
        the store hands back one account's token against another's identity."""
        custom = tmp_path / "custom-profile"
        custom.mkdir()
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(custom))
        monkeypatch.delenv("CLAUDE_SECURESTORAGE_CONFIG_DIR", raising=False)

        seen: list[str] = []
        monkeypatch.setattr(
            "claude_swap.macos_keychain.get_password",
            _keychain(
                {
                    CLAUDE_CODE_KEYCHAIN_SERVICE: DEFAULT_PROFILE_CREDS,
                    CLAUDE_CODE_MANAGED_KEYCHAIN_SERVICE: "sk-ant-api-default",
                },
                seen,
            ),
        )

        result = CredentialStore(_Host(tmp_path / "backups"))._read_active_credentials()

        assert CLAUDE_CODE_KEYCHAIN_SERVICE not in seen, (
            f"read the default profile's OAuth item: {seen}"
        )
        assert CLAUDE_CODE_MANAGED_KEYCHAIN_SERVICE not in seen, (
            f"read the default profile's managed-key item: {seen}"
        )
        assert result.value != DEFAULT_PROFILE_CREDS
        assert not result.value

    def test_custom_config_dir_reads_its_own_hashed_keychain_item(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The regression a plain skip would introduce.

        On macOS claude writes rotations keychain-only, so a live custom profile
        commonly has a hashed item and NO plaintext file. Skipping the Keychain
        would report "no credentials" for a profile that is logged in; the
        redirect returns the profile's real credential."""
        custom = tmp_path / "custom-profile"
        custom.mkdir()  # deliberately no .credentials.json
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(custom))
        monkeypatch.delenv("CLAUDE_SECURESTORAGE_CONFIG_DIR", raising=False)

        seen: list[str] = []
        monkeypatch.setattr(
            "claude_swap.macos_keychain.get_password",
            _keychain(
                {
                    keychain_service_name(str(custom)): CUSTOM_PROFILE_CREDS,
                    CLAUDE_CODE_KEYCHAIN_SERVICE: DEFAULT_PROFILE_CREDS,
                },
                seen,
            ),
        )

        result = CredentialStore(_Host(tmp_path / "backups"))._read_active_credentials()

        assert result.value == CUSTOM_PROFILE_CREDS
        assert seen == [keychain_service_name(str(custom))]

    def test_custom_config_dir_falls_back_to_its_own_credentials_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """An ABSENT hashed item (rc 44) is claude's own signal to read the
        plaintext seed — and that file is unambiguously this profile's."""
        custom = tmp_path / "custom-profile"
        custom.mkdir()
        (custom / ".credentials.json").write_text(
            CUSTOM_PROFILE_CREDS, encoding="utf-8"
        )
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(custom))
        monkeypatch.delenv("CLAUDE_SECURESTORAGE_CONFIG_DIR", raising=False)

        seen: list[str] = []
        monkeypatch.setattr(
            "claude_swap.macos_keychain.get_password",
            _keychain({CLAUDE_CODE_KEYCHAIN_SERVICE: DEFAULT_PROFILE_CREDS}, seen),
        )

        store = CredentialStore(_Host(tmp_path / "backups"))
        assert store._read_active_credentials().value == CUSTOM_PROFILE_CREDS
        assert CLAUDE_CODE_KEYCHAIN_SERVICE not in seen

    def test_default_profile_still_reads_the_unsuffixed_item(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The common case is unchanged. With no ``CLAUDE_CONFIG_DIR`` the
        unsuffixed item IS this profile's credential."""
        monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
        monkeypatch.delenv("CLAUDE_SECURESTORAGE_CONFIG_DIR", raising=False)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

        seen: list[str] = []
        monkeypatch.setattr(
            "claude_swap.macos_keychain.get_password",
            _keychain({CLAUDE_CODE_KEYCHAIN_SERVICE: DEFAULT_PROFILE_CREDS}, seen),
        )

        store = CredentialStore(_Host(tmp_path / "backups"))
        assert store._read_active_credentials().value == DEFAULT_PROFILE_CREDS
        assert seen == [CLAUDE_CODE_KEYCHAIN_SERVICE]

    def test_config_dir_equal_to_the_default_falls_back_to_the_unsuffixed_item(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Setting ``CLAUDE_CONFIG_DIR`` to the default profile explicitly is not
        a custom profile. Claude hashes the exported string, so the hashed name
        is tried first, but a user who has always used the default profile may
        only have the unsuffixed item — so that fallback must remain."""
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        default = tmp_path / ".claude"
        default.mkdir()
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(default))
        monkeypatch.delenv("CLAUDE_SECURESTORAGE_CONFIG_DIR", raising=False)

        seen: list[str] = []
        monkeypatch.setattr(
            "claude_swap.macos_keychain.get_password",
            _keychain({CLAUDE_CODE_KEYCHAIN_SERVICE: DEFAULT_PROFILE_CREDS}, seen),
        )

        store = CredentialStore(_Host(tmp_path / "backups"))
        assert store._read_active_credentials().value == DEFAULT_PROFILE_CREDS
        assert seen == [
            keychain_service_name(str(default)),
            CLAUDE_CODE_KEYCHAIN_SERVICE,
        ]


class TestSecureStorageOverride:
    """``CLAUDE_SECURESTORAGE_CONFIG_DIR`` takes precedence when *defined*.

    Claude 2.1.220+ resolves secure storage from it and only falls back to
    ``CLAUDE_CONFIG_DIR`` when it is undefined. ``_read_capture_credentials``
    already reads it this way; the active read has to agree, or the two disagree
    about which profile they are looking at.
    """

    def test_defined_and_empty_selects_the_default_store(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Defined-but-empty means the DEFAULT secure store, even with a custom
        ``CLAUDE_CONFIG_DIR``. Here the unsuffixed item is the correct read, so
        a guard keyed only on ``CLAUDE_CONFIG_DIR`` would wrongly return
        nothing."""
        custom = tmp_path / "custom-profile"
        custom.mkdir()
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(custom))
        monkeypatch.setenv("CLAUDE_SECURESTORAGE_CONFIG_DIR", "")

        seen: list[str] = []
        monkeypatch.setattr(
            "claude_swap.macos_keychain.get_password",
            _keychain({CLAUDE_CODE_KEYCHAIN_SERVICE: DEFAULT_PROFILE_CREDS}, seen),
        )

        store = CredentialStore(_Host(tmp_path / "backups"))
        assert store._read_active_credentials().value == DEFAULT_PROFILE_CREDS
        assert seen == [CLAUDE_CODE_KEYCHAIN_SERVICE]

    def test_defined_and_set_selects_that_stores_hashed_item(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A defined, non-empty value names the only store claude will read."""
        custom = tmp_path / "custom-profile"
        custom.mkdir()
        secure = tmp_path / "secure-profile"
        secure.mkdir()
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(custom))
        monkeypatch.setenv("CLAUDE_SECURESTORAGE_CONFIG_DIR", str(secure))

        seen: list[str] = []
        monkeypatch.setattr(
            "claude_swap.macos_keychain.get_password",
            _keychain(
                {
                    keychain_service_name(str(secure)): SECURE_PROFILE_CREDS,
                    keychain_service_name(str(custom)): CUSTOM_PROFILE_CREDS,
                    CLAUDE_CODE_KEYCHAIN_SERVICE: DEFAULT_PROFILE_CREDS,
                },
                seen,
            ),
        )

        store = CredentialStore(_Host(tmp_path / "backups"))
        assert store._read_active_credentials().value == SECURE_PROFILE_CREDS
        assert seen == [keychain_service_name(str(secure))]


CLAUDE_OAUTH_SERVICE_SUFFIX = "-credentials"
CLAUDE_MANAGED_KEY_SERVICE_SUFFIX = ""


def _claude_service_name(service_suffix: str, secure_store_dir: str | None) -> str:
    """Claude Code's Keychain service name, restated independently of cswap.

    ``getMacOsKeychainStorageServiceName(serviceSuffix)`` (Claude Code 2.1.283
    bundle): ``"Claude Code" + serviceSuffix``, plus ``"-" +
    sha256(dir)[:8]`` unless the environment selects the default secure store
    (``secure_store_dir=None`` here). The OAuth item passes ``"-credentials"``;
    the managed ("/login" API key) item passes nothing. Deliberately not built
    from cswap's own helpers, so these tests pin cswap to Claude's contract
    rather than to itself.
    """
    name = "Claude Code" + service_suffix
    if secure_store_dir is None:
        return name
    digest = hashlib.sha256(
        unicodedata.normalize("NFC", secure_store_dir).encode("utf-8")
    ).hexdigest()
    return f"{name}-{digest[:8]}"


@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("mode", ["oauth", "managed", "fallback"])
def test_active_writes_and_clears_stay_on_selected_profile(tmp_path, monkeypatch, custom, mode):
    """Switching one profile must never overwrite or delete another login."""
    config_home = tmp_path / "infinity" if custom else Path.home() / ".claude"
    config_home.mkdir(parents=True, exist_ok=True)
    if custom:
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_home))
    else:
        monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("CLAUDE_SECURESTORAGE_CONFIG_DIR", raising=False)
    selected_store = str(config_home) if custom else None
    oauth_service = _claude_service_name(CLAUDE_OAUTH_SERVICE_SUFFIX, selected_store)
    managed_service = _claude_service_name(CLAUDE_MANAGED_KEY_SERVICE_SUFFIX, selected_store)
    untouched = {
        CLAUDE_CODE_KEYCHAIN_SERVICE: DEFAULT_PROFILE_CREDS,
        CLAUDE_CODE_MANAGED_KEYCHAIN_SERVICE: "sk-ant-api03-default-fixture",
    }
    mapping = dict(untouched)
    mapping[oauth_service] = CUSTOM_PROFILE_CREDS
    mapping[managed_service] = "sk-ant-api03-old-selected-fixture"

    def write(service, account, value):
        if mode == "fallback":
            raise OSError("Fixture Keychain write unavailable")
        mapping[service] = value

    monkeypatch.setattr("claude_swap.macos_keychain.get_password", lambda service, account: mapping.get(service))
    monkeypatch.setattr("claude_swap.macos_keychain.set_password", write)
    monkeypatch.setattr("claude_swap.macos_keychain.delete_password", lambda service, account: mapping.pop(service, None))
    store = CredentialStore(_Host(tmp_path / "backups"))
    value = "sk-ant-api03-new-selected-fixture" if mode == "managed" else SECURE_PROFILE_CREDS
    store._write_credentials(value)
    if custom:
        assert {key: mapping.get(key) for key in untouched} == untouched
    if mode == "fallback":
        assert oauth_service not in mapping
        assert (config_home / ".credentials.json").read_text() == value
    else:
        assert mapping[managed_service if mode == "managed" else oauth_service] == value
    # Activating one auth axis clears the other — on the selected profile only.
    if mode == "managed":
        assert oauth_service not in mapping
    else:
        assert managed_service not in mapping
    assert store._read_active_credentials().value == value


def _select_secure_store(
    environment: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[str | None, list[str | None]]:
    """Export one profile environment; return (selected store, foreign stores).

    The selected store is the one Claude reads for the environment (``None`` =
    the default store's unsuffixed items). Foreign stores belong to profiles
    the environment does NOT select; a switch must leave their items alone.
    """
    custom = tmp_path / "custom-profile"
    custom.mkdir()
    unrelated = str(tmp_path / "unrelated-profile")
    if environment == "explicit_default":
        # CLAUDE_CONFIG_DIR naming the default profile: Claude hashes the
        # exported string. The unsuffixed items are the same profile, so they
        # are neither selected nor foreign.
        default = Path.home() / ".claude"
        default.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(default))
        monkeypatch.delenv("CLAUDE_SECURESTORAGE_CONFIG_DIR", raising=False)
        return str(default), [unrelated]
    if environment == "secure_storage":
        secure = tmp_path / "secure-profile"
        secure.mkdir()
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(custom))
        monkeypatch.setenv("CLAUDE_SECURESTORAGE_CONFIG_DIR", str(secure))
        return str(secure), [unrelated, str(custom), None]
    if environment == "secure_storage_empty":
        # Defined-but-empty selects the DEFAULT secure store even under a
        # custom CLAUDE_CONFIG_DIR, so the custom profile's items are foreign.
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(custom))
        monkeypatch.setenv("CLAUDE_SECURESTORAGE_CONFIG_DIR", "")
        return None, [unrelated, str(custom)]
    raise AssertionError(f"unknown environment {environment!r}")


@pytest.mark.parametrize("axis", ["oauth", "managed"])
@pytest.mark.parametrize(
    "environment", ["explicit_default", "secure_storage", "secure_storage_empty"]
)
def test_switch_writes_the_item_claude_reads_for_the_environment(
    tmp_path, monkeypatch, block_real_keychain, environment, axis
):
    """The write lands in the item Claude reads, clears the other axis there,
    and leaves every other profile's items byte-for-byte alone."""
    selected_store, foreign_stores = _select_secure_store(
        environment, tmp_path, monkeypatch
    )
    account = macos_keychain.keychain_account_name()
    selected_oauth = _claude_service_name(CLAUDE_OAUTH_SERVICE_SUFFIX, selected_store)
    selected_managed = _claude_service_name(
        CLAUDE_MANAGED_KEY_SERVICE_SUFFIX, selected_store
    )
    foreign_items = {
        _claude_service_name(suffix, store_dir): f"foreign{suffix}-{store_dir}"
        for store_dir in foreign_stores
        for suffix in (CLAUDE_OAUTH_SERVICE_SUFFIX, CLAUDE_MANAGED_KEY_SERVICE_SUFFIX)
    }
    for service, value in foreign_items.items():
        block_real_keychain.set_password(service, account, value)

    if axis == "oauth":
        new_credential = SECURE_PROFILE_CREDS
        written_service, other_axis_service = selected_oauth, selected_managed
        block_real_keychain.set_password(
            other_axis_service, account, "sk-ant-api03-old-selected-fixture"
        )
    else:
        new_credential = "sk-ant-api03-new-selected-fixture"
        written_service, other_axis_service = selected_managed, selected_oauth
        block_real_keychain.set_password(other_axis_service, account, CUSTOM_PROFILE_CREDS)

    store = CredentialStore(_Host(tmp_path / "backups"))
    store._write_credentials(new_credential)

    assert block_real_keychain.get_password(written_service, account) == new_credential
    assert block_real_keychain.get_password(other_axis_service, account) is None
    assert {
        service: block_real_keychain.get_password(service, account)
        for service in foreign_items
    } == foreign_items
    assert store._read_active_credentials().value == new_credential
