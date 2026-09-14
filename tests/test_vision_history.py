"""History migration copies conversation state and preserves conflicts for recovery."""

import json
from pathlib import Path

import pytest

from claude_swap.exceptions import SessionError
from claude_swap.vision_history import HISTORY_MARKER, copy_history, install_history


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_snapshot_copies_history_but_not_auth_or_settings(tmp_path):
    source, snapshot = tmp_path / "source", tmp_path / "snapshot"
    put(source / "projects" / "project" / "session.jsonl", b"conversation\n")
    put(source / "history.jsonl", b'{"display":"hello"}\n')
    put(source / ".credentials.json", b"synthetic-refresh")
    put(source / "settings.json", b"synthetic-api-key")
    copy_history([source], snapshot)
    assert (
        snapshot / "projects" / "project" / "session.jsonl"
    ).read_bytes() == b"conversation\n"
    assert {p.name for p in snapshot.iterdir()} == {"projects", "history.jsonl"}


def test_repeat_import_preserves_extended_transcript_and_deduplicates_index(tmp_path):
    source, snapshot, target = (
        tmp_path / name for name in ("source", "snapshot", "target")
    )
    transcript = Path("projects/project/session.jsonl")
    put(source / transcript, b"first\n")
    put(source / "history.jsonl", b'{"display":"hello"}\n')
    copy_history([source], snapshot)
    install_history(snapshot, target, "request")
    put(target / transcript, b"first\nsecond\n")
    install_history(snapshot, target, "request")
    assert (target / transcript).read_bytes() == b"first\nsecond\n"
    assert (target / "history.jsonl").read_bytes() == b'{"display":"hello"}\n'
    assert not (target / HISTORY_MARKER).exists()


def test_conflict_preserves_both_versions_and_leaves_recovery_marker(tmp_path):
    snapshot, target = tmp_path / "snapshot", tmp_path / "target"
    relative = Path("projects/project/session.jsonl")
    put(snapshot / relative, b"source version")
    put(target / relative, b"different version")
    with pytest.raises(SessionError, match="conflict"):
        install_history(snapshot, target, "request")
    assert (snapshot / relative).read_bytes() == b"source version"
    assert (target / relative).read_bytes() == b"different version"
    assert json.loads((target / HISTORY_MARKER).read_text())["request_id"] == "request"


def test_known_shared_history_root_is_supported_but_unknown_link_is_rejected(tmp_path):
    profile, shared, destination = (
        tmp_path / name for name in ("profile", "shared", "destination")
    )
    profile.mkdir()
    put(shared / "projects" / "session.jsonl", b"conversation")
    (profile / "projects").symlink_to(shared / "projects", target_is_directory=True)
    with pytest.raises(SessionError, match="outside"):
        copy_history([profile], destination)
    copy_history([profile], destination, known_profiles=[profile, shared])
    assert (destination / "projects" / "session.jsonl").read_bytes() == b"conversation"


def test_nested_symlink_is_not_followed(tmp_path):
    profile = tmp_path / "profile"
    put(profile / "projects" / "safe.jsonl", b"conversation")
    secret = tmp_path / "credential"
    secret.write_text("synthetic-refresh")
    (profile / "projects" / "linked.jsonl").symlink_to(secret)
    with pytest.raises(SessionError, match="symbolic"):
        copy_history([profile], tmp_path / "destination")


def test_missing_snapshot_cannot_silently_complete_import(tmp_path):
    with pytest.raises(SessionError, match="missing"):
        install_history(tmp_path / "missing", tmp_path / "target", "request")


def test_pending_import_cannot_be_replaced_by_another_request(tmp_path):
    snapshot, target = tmp_path / "snapshot", tmp_path / "target"
    snapshot.mkdir()
    target.mkdir()
    marker = target / HISTORY_MARKER
    marker.write_text(json.dumps({"request_id": "original"}))
    marker.chmod(0o600)
    with pytest.raises(SessionError, match="Another"):
        install_history(snapshot, target, "different")
    assert json.loads(marker.read_text())["request_id"] == "original"
