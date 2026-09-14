"""Private durable browser proofs and API keys, separate from user preferences."""

import json
import os
import stat
from pathlib import Path

from claude_swap.exceptions import SessionError
from claude_swap.locking import FileLock
from claude_swap.session import _mkdir_private
from claude_swap.vision_handoff import _read_private, _sync_directory, _write_private

MAX_STATE_BYTES = 16384


class VisionState:
    def __init__(self, root: Path):
        self.directory = root / "vision-auth"

    def _check_directory(self, create=False):
        if create:
            _mkdir_private(self.directory)
        try:
            metadata = self.directory.lstat()
        except FileNotFoundError:
            return False
        if not stat.S_ISDIR(metadata.st_mode) or (
            os.name != "nt"
            and (metadata.st_uid != os.getuid() or metadata.st_mode & 0o077)
        ):
            raise SessionError(
                "Vision sign-in storage must be a private directory owned by you."
            )
        return True

    def _path(self, name):
        if name not in {"key", "pending"}:
            raise SessionError("Invalid Vision sign-in state name.")
        return self.directory / (name + ".json")

    def read(self, name):
        path = self._path(name)
        if not self._check_directory():
            return None
        raw = _read_private(path)
        if raw is None:
            return None
        try:
            if len(raw.encode()) > MAX_STATE_BYTES:
                raise ValueError()
            return json.loads(raw)
        except ValueError:
            raise SessionError("Vision sign-in state needs repair.") from None

    def write(self, name, value):
        path = self._path(name)
        self._check_directory(create=True)
        raw = json.dumps(value, allow_nan=False)
        if len(raw.encode()) > MAX_STATE_BYTES:
            raise SessionError("Vision sign-in state is too large.")
        _write_private(path, raw)

    def remove(self, name):
        path = self._path(name)
        if self._check_directory():
            path.unlink(missing_ok=True)
            _sync_directory(self.directory)

    def lock(self):
        self._check_directory(create=True)
        return FileLock(self.directory / "sign-in.lock")
