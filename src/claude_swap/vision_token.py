"""Manual Vision API key shared with codex-swap, independent of native homes."""

from __future__ import annotations

import getpass
import json
import os
import re
import stat
import sys
import warnings
from pathlib import Path

from claude_swap.exceptions import ClaudeSwitchError, SessionError
from claude_swap.vision import VisionClient, origin
from claude_swap.vision_handoff import _read_private, _write_private


def token_path() -> Path:
    configured = os.environ.get("XDG_CONFIG_HOME")
    root = Path(configured) if configured else Path.home() / ".config"
    if not root.is_absolute():
        raise SessionError("XDG_CONFIG_HOME must be an absolute path.")
    return root / "vision" / "credentials.json"


def _check_directory(path: Path, *, create: bool = False) -> bool:
    directory = path.parent
    if create:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        metadata = directory.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISDIR(metadata.st_mode) or (
        os.name != "nt"
        and (metadata.st_uid != os.getuid() or metadata.st_mode & 0o077)
    ):
        raise SessionError("Vision token storage must be a private directory owned by you.")
    return True


def _validate_token(token: str) -> str:
    if not isinstance(token, str) or not re.fullmatch(r"vsk_[0-9a-f]{40}", token):
        raise SessionError("Enter a Vision API key from Vision Settings → API keys.")
    return token


def saved_token_client() -> VisionClient | None:
    path = token_path()
    if not _check_directory(path):
        return None
    raw = _read_private(path)
    if raw is None:
        return None
    try:
        saved = json.loads(raw)
        if (
            not isinstance(saved, dict)
            or set(saved) != {"version", "url", "api_key"}
            or type(saved["version"]) is not int
            or saved["version"] != 1
        ):
            raise ValueError()
        key = _validate_token(saved["api_key"])
        destination = origin(saved["url"])
    except (ValueError, TypeError, KeyError, ClaudeSwitchError):
        raise SessionError("Saved Vision token needs repair; run --set-vision-token again.") from None
    override = os.environ.get("VISION_API_URL")
    if override and origin(override) != destination:
        raise SessionError("Saved Vision token belongs to another origin; configure a key for that origin.")
    return VisionClient(destination, key)


def save_token(token: str) -> Path:
    key = _validate_token(token)
    destination = origin(os.environ.get("VISION_API_URL", "https://vision.infinity.inc"))
    path = token_path()
    _check_directory(path, create=True)
    _write_private(path, json.dumps({"version": 1, "url": destination, "api_key": key}))
    return path


def setup_command(argv: list[str]) -> None:
    """Set only wrapper configuration; never construct a native account switcher."""
    try:
        if len(argv) > 1:
            raise SessionError("Use --set-vision-token [TOKEN] by itself.")
        if argv:
            token = argv[0]
        else:
            if not sys.stdin.isatty():
                raise SessionError("Run --set-vision-token in a terminal to enter the key privately.")
            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                token = getpass.getpass("Vision API key: ")
        path = save_token(token)
        print(f"Vision API key saved for cswap and codex-swap in {path}.")
        if os.environ.get("VISION_API_KEY"):
            print("VISION_API_KEY is set and takes precedence over the saved key.")
    except (EOFError, KeyboardInterrupt):
        raise SystemExit(130) from None
    except (ClaudeSwitchError, OSError, getpass.GetPassWarning):
        # Filesystem errors can include user-controlled content; keep diagnostics secret-free.
        print("Could not save Vision API key. Use a valid key and private config directory.", file=sys.stderr)
        raise SystemExit(1) from None
