"""Feature flags of Vision-proxied sessions are refreshed out of band (#40).

Claude Code fetches its GrowthBook flags from api.anthropic.com directly, with
CLAUDE_CODE_OAUTH_TOKEN as the bearer. Under the Vision adapter that variable
holds the adapter's capability, so the fetch is refused and the flags cached in
``.claude.json`` stop moving. These tests drive the refresher with a fake
native that behaves like Claude Code's ``doctor``: it writes a fresh flag
payload into the ``.claude.json`` of the CLAUDE_CONFIG_DIR it was given.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import textwrap
import time
from pathlib import Path

import pytest

from claude_swap.flag_refresh import (
    FLAG_KEYS,
    REFRESH_AFTER_SECONDS,
    FlagRefresher,
)
from claude_swap.paths import global_config_path_for

# The fake native stamps its payload with the real clock, so NOW is real too.
NOW = time.time()
TWELVE_DAYS_AGO_MS = int((NOW - 12 * 86400) * 1000)
FRESH_FLAGS = {"tengu_usage_overage_included_models": ["Fable 5.2"]}

FAKE_NATIVE = textwrap.dedent(
    """
    import json, os, sys, time
    from pathlib import Path

    mode = Path(sys.argv[0]).with_suffix(".mode").read_text()
    config_dir = Path(os.environ["CLAUDE_CONFIG_DIR"])
    report = {
        "argv": sys.argv[1:],
        "config_dir": str(config_dir),
        "token": os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"),
        "base_url": os.environ.get("ANTHROPIC_BASE_URL"),
        "secure_storage": os.environ.get("CLAUDE_SECURESTORAGE_CONFIG_DIR"),
        "seed": json.loads((config_dir / ".claude.json").read_text())
        if (config_dir / ".claude.json").exists() else None,
    }
    Path(sys.argv[0]).with_suffix(".report").write_text(json.dumps(report))
    if mode == "hang":
        time.sleep(60)
    if mode == "flags":
        seed = report["seed"] or {}
        seed.update({
            "cachedGrowthBookFeatures": %r,
            "cachedGrowthBookFeaturesAt": int(time.time() * 1000),
            "cachedExperimentFeatures": [],
            "cachedExperimentData": {},
            "oauthAccount": {"emailAddress": "pool@example.invalid"},
        })
        (config_dir / ".claude.json").write_text(json.dumps(seed))
    print("doctor output that must never reach the terminal")
    """
    % (FRESH_FLAGS,)
)


@pytest.fixture
def native(tmp_path):
    """A fake native; ``mode`` picks what its ``doctor`` does."""
    script = tmp_path / "bin" / "fake_claude.py"
    script.parent.mkdir()
    script.write_text(FAKE_NATIVE)

    class Native:
        command = [sys.executable, str(script)]

        def mode(self, value):
            script.with_suffix(".mode").write_text(value)

        def report(self):
            path = script.with_suffix(".report")
            return json.loads(path.read_text()) if path.exists() else None

    fake = Native()
    fake.mode("flags")
    return fake


@pytest.fixture
def profile(tmp_path):
    """The session's own config home, holding 12-day-old cached flags."""
    home = tmp_path / "profile"
    home.mkdir()
    config = home / ".claude.json"
    config.write_text(
        json.dumps(
            {
                "userID": "device-id-of-this-machine",
                "oauthAccount": {"emailAddress": "person@example.invalid"},
                "projects": {"/work": {"allowedTools": []}},
                "cachedGrowthBookFeatures": {
                    "tengu_usage_overage_included_models": ["Fable 5.1"]
                },
                "cachedGrowthBookFeaturesAt": TWELVE_DAYS_AGO_MS,
                "cachedExperimentFeatures": ["old"],
                "cachedExperimentData": {"old": {}},
            }
        )
    )
    if sys.platform != "win32":
        os.chmod(config, 0o640)
    return config


def make_refresher(native, config, tmp_path, *, token="real-provider-token", **kw):
    tokens = []

    def token_source():
        tokens.append(1)
        if isinstance(token, Exception):
            raise token
        return token

    refresher = FlagRefresher(
        native.command,
        config,
        {
            "PATH": os.environ.get("PATH", ""),
            "ANTHROPIC_BASE_URL": "http://127.0.0.1:9",
            "CLAUDE_CODE_OAUTH_TOKEN": "adapter-capability",
            "CLAUDE_CONFIG_DIR": str(config.parent),
            "CLAUDE_SECURESTORAGE_CONFIG_DIR": str(tmp_path / "secure"),
        },
        token_source,
        marker_dir=tmp_path / "markers",
        clock=kw.pop("clock", lambda: NOW),
        **kw,
    )
    return refresher, tokens


def read(config):
    return json.loads(config.read_text())


def test_stale_flags_are_refreshed_from_a_real_token_run(native, profile, tmp_path):
    before = read(profile)
    refresher, tokens = make_refresher(native, profile, tmp_path)

    assert refresher.refresh_if_stale() == "refreshed"

    after = read(profile)
    assert after["cachedGrowthBookFeatures"] == FRESH_FLAGS
    assert after["cachedGrowthBookFeaturesAt"] > TWELVE_DAYS_AGO_MS
    assert after["cachedExperimentFeatures"] == []
    assert after["cachedExperimentData"] == {}
    # Only the flag keys move: the identity native wrote for the pool login,
    # and everything else in the person's config, stay as they were.
    for key in set(before) - set(FLAG_KEYS):
        assert after[key] == before[key]
    assert set(after) == set(before)
    assert len(tokens) == 1
    if sys.platform != "win32":
        assert stat.S_IMODE(profile.stat().st_mode) == 0o640
    assert not (profile.parent / ".claude.json.lock").exists()


def test_the_refresh_runs_isolated_from_the_session(native, profile, tmp_path):
    refresher, _ = make_refresher(native, profile, tmp_path)
    refresher.refresh_if_stale()

    report = native.report()
    assert report["argv"] == ["doctor"]
    # The real token goes to the throwaway run only, never through the
    # adapter's URL, and native writes into a throwaway profile rather than
    # the person's, seeded with this machine's id so rollouts match.
    assert report["token"] == "real-provider-token"
    assert report["base_url"] is None
    assert Path(report["config_dir"]) != profile.parent
    assert report["secure_storage"] not in {None, str(tmp_path / "secure")}
    assert report["seed"] == {"userID": "device-id-of-this-machine"}
    assert not Path(report["config_dir"]).exists()


def test_fresh_flags_are_left_alone_without_issuing_a_token(native, profile, tmp_path):
    data = read(profile)
    data["cachedGrowthBookFeaturesAt"] = int((NOW - 60) * 1000)
    profile.write_text(json.dumps(data))
    refresher, tokens = make_refresher(native, profile, tmp_path)

    assert refresher.refresh_if_stale() == "fresh"
    assert tokens == []
    assert native.report() is None


def test_flags_never_cached_count_as_stale(native, profile, tmp_path):
    data = read(profile)
    for key in FLAG_KEYS:
        data.pop(key)
    profile.write_text(json.dumps(data))
    refresher, _ = make_refresher(native, profile, tmp_path)

    assert refresher.refresh_if_stale() == "refreshed"
    assert read(profile)["cachedGrowthBookFeatures"] == FRESH_FLAGS


def test_a_run_that_writes_no_flags_changes_nothing_and_backs_off(
    native, profile, tmp_path
):
    native.mode("noflags")
    before = profile.read_bytes()
    refresher, tokens = make_refresher(native, profile, tmp_path)

    assert refresher.refresh_if_stale() == "no_flags"
    assert profile.read_bytes() == before
    # A failed attempt is not retried by every launch in the next hour.
    again, more_tokens = make_refresher(native, profile, tmp_path)
    assert again.refresh_if_stale() == "backoff"
    assert more_tokens == []


def test_backoff_expires(native, profile, tmp_path):
    native.mode("noflags")
    refresher, _ = make_refresher(native, profile, tmp_path)
    refresher.refresh_if_stale()
    native.mode("flags")
    later, _ = make_refresher(
        native, profile, tmp_path, clock=lambda: NOW + REFRESH_AFTER_SECONDS
    )

    assert later.refresh_if_stale() == "refreshed"


def test_a_hung_run_is_stopped_and_changes_nothing(native, profile, tmp_path):
    native.mode("hang")
    before = profile.read_bytes()
    refresher, _ = make_refresher(native, profile, tmp_path, timeout_seconds=1)

    started = time.monotonic()
    assert refresher.refresh_if_stale() == "failed"
    assert time.monotonic() - started < 30
    assert profile.read_bytes() == before


def test_newer_flags_written_meanwhile_are_kept(native, profile, tmp_path):
    refresher, _ = make_refresher(native, profile, tmp_path)
    newer = {"from": "another session"}
    original = refresher._run_native

    def run_then_other_writer(token):
        result = original(token)
        data = read(profile)
        data["cachedGrowthBookFeatures"] = newer
        data["cachedGrowthBookFeaturesAt"] = int(time.time() * 1000) + 60_000
        profile.write_text(json.dumps(data))
        return result

    refresher._run_native = run_then_other_writer
    assert refresher.refresh_if_stale() == "superseded"
    assert read(profile)["cachedGrowthBookFeatures"] == newer


def test_a_token_failure_is_soft(native, profile, tmp_path):
    before = profile.read_bytes()
    refresher, _ = make_refresher(
        native, profile, tmp_path, token=RuntimeError("Vision is down")
    )

    assert refresher.refresh_if_stale() == "failed"
    assert profile.read_bytes() == before
    assert native.report() is None


@pytest.mark.parametrize("content", [None, "not json", "[]"])
def test_a_missing_or_unreadable_config_is_never_written(
    native, profile, tmp_path, content
):
    if content is None:
        profile.unlink()
    else:
        profile.write_text(content)
    refresher, tokens = make_refresher(native, profile, tmp_path)

    assert refresher.refresh_if_stale() == "no_config"
    assert tokens == []
    if content is None:
        assert not profile.exists()
    else:
        assert profile.read_text() == content


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges")
def test_a_symlinked_config_is_not_replaced(native, profile, tmp_path):
    real = tmp_path / "dotfiles.json"
    profile.rename(real)
    profile.symlink_to(real)
    refresher, tokens = make_refresher(native, profile, tmp_path)

    assert refresher.refresh_if_stale() == "no_config"
    assert profile.is_symlink()
    assert tokens == []


def test_background_thread_refreshes_and_stops(native, profile, tmp_path):
    refresher, _ = make_refresher(native, profile, tmp_path)
    refresher.start()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if read(profile)["cachedGrowthBookFeatures"] == FRESH_FLAGS:
            break
        time.sleep(0.05)
    refresher.stop(timeout=5)
    assert read(profile)["cachedGrowthBookFeatures"] == FRESH_FLAGS
    assert not refresher.thread.is_alive()


def test_stopping_ends_a_run_in_progress(native, profile, tmp_path):
    native.mode("hang")
    refresher, _ = make_refresher(native, profile, tmp_path)
    refresher.start()
    deadline = time.monotonic() + 30
    while native.report() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    started = time.monotonic()
    refresher.stop(timeout=10)
    assert time.monotonic() - started < 10
    assert not refresher.thread.is_alive()


def test_config_path_follows_the_native_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    custom = tmp_path / "custom"
    assert global_config_path_for({"CLAUDE_CONFIG_DIR": str(custom)}) == (
        custom / ".claude.json"
    )
    assert global_config_path_for({}) == Path.home() / ".claude.json"
    legacy = custom / ".config.json"
    custom.mkdir()
    legacy.write_text("{}")
    assert global_config_path_for({"CLAUDE_CONFIG_DIR": str(custom)}) == legacy
