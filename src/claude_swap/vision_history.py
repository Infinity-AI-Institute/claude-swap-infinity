"""Copy native conversation history without importing account or auth settings."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from pathlib import Path

from claude_swap.exceptions import SessionError
from claude_swap.locking import FileLock
from claude_swap.vision_handoff import _read_private, _sync_directory, _write_private

HISTORY_MARKER = ".vision-history-import.json"
CHUNK = 1024 * 1024


def _directory(path):
    if path.is_symlink():
        raise SessionError("History destinations cannot be symbolic links.")
    if not path.exists():
        _directory(path.parent)
        path.mkdir(mode=0o700)
    elif not path.is_dir():
        raise SessionError("A history directory conflicts with an existing file.")


def _open_file(path):
    fd = os.open(
        path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    )
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise SessionError("Native history must contain regular files.")
    return os.fdopen(fd, "rb")


def _copy_file(source, destination):
    if source.is_symlink() or destination.is_symlink():
        raise SessionError("Nested history files cannot be symbolic links.")
    _directory(destination.parent)
    with _open_file(source) as incoming:
        source_size = os.fstat(incoming.fileno()).st_size
        try:
            existing = _open_file(destination)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            with existing:
                destination_size = os.fstat(existing.fileno()).st_size
                remaining = min(source_size, destination_size)
                while remaining:
                    count = min(CHUNK, remaining)
                    if incoming.read(count) != existing.read(count):
                        raise SessionError(
                            "Native history versions conflict; reconcile them before migration."
                        )
                    remaining -= count
            if destination_size >= source_size:
                return
            incoming.seek(0)
        fd, temporary = tempfile.mkstemp(dir=destination.parent, prefix=".history-")
        try:
            with os.fdopen(fd, "wb") as output:
                shutil.copyfileobj(incoming, output, CHUNK)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
            _sync_directory(destination.parent)
        finally:
            Path(temporary).unlink(missing_ok=True)


def _copy_tree(source, destination):
    if source.is_symlink():
        raise SessionError("Nested history directories cannot be symbolic links.")
    _directory(destination)
    for child in sorted(source.iterdir()):
        target = destination / child.name
        if child.is_symlink():
            raise SessionError("Nested history entries cannot be symbolic links.")
        if child.is_dir():
            _copy_tree(child, target)
        else:
            _copy_file(child, target)


def _merge_index(source, destination):
    """Deduplicate history index lines; conversation transcripts stay byte-exact."""
    _directory(destination.parent)
    if source.is_symlink() or destination.is_symlink():
        raise SessionError("History index files cannot be symbolic links.")
    fd, temporary = tempfile.mkstemp(dir=destination.parent, prefix=".history-index-")
    try:
        seen = set()
        with os.fdopen(fd, "wb") as output:
            for path in (destination, source):
                try:
                    incoming = _open_file(path)
                except FileNotFoundError:
                    continue
                with incoming:
                    for line in incoming:
                        normalized = line.rstrip(b"\r\n") + b"\n"
                        digest = hashlib.sha256(normalized).digest()
                        if digest not in seen:
                            output.write(normalized)
                            seen.add(digest)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        _sync_directory(destination.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def copy_history(profiles, destination, *, known_profiles=None):
    """Snapshot only supported history roots, including known shared-history links."""
    allowed = set(profiles if known_profiles is None else known_profiles)
    _directory(destination)
    for profile in profiles:
        for name in ("projects", "history.jsonl"):
            source = profile / name
            if source.is_symlink():
                resolved = source.resolve(strict=True)
                if resolved not in {
                    (path / name).resolve()
                    for path in allowed
                    if not (path / name).is_symlink()
                }:
                    raise SessionError(
                        "A shared history link points outside the known native profiles."
                    )
                source = resolved
            try:
                metadata = source.stat()
            except FileNotFoundError:
                continue
            if name == "projects":
                if not stat.S_ISDIR(metadata.st_mode):
                    raise SessionError("Native projects history must be a directory.")
                _copy_tree(source, destination / name)
            else:
                _merge_index(source, destination / name)


def install_history(snapshot, destination, request_id):
    """Block new central launches until a recoverable history import completes."""
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise SessionError("The migration history snapshot is missing or invalid.")
    _directory(destination)
    with FileLock(destination / ".vision-history.lock"):
        marker = destination / HISTORY_MARKER
        existing = _read_private(marker)
        expected = json.dumps({"request_id": request_id})
        if existing is not None and existing != expected:
            raise SessionError(
                "Another history import needs recovery before this migration."
            )
        _write_private(marker, expected)
        copy_history([snapshot], destination)
        marker.unlink()
        _sync_directory(destination)
