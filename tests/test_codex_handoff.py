"""Tests for the codex session handoff (codex_handoff.py).

Everything external — tmux, the process table, the usage API, the clock —
is injected: the FakeRunner records every command and serves scripted
outputs, so these tests assert the EXACT command sequences (paste
discipline, C-c, relaunch, update-prompt answer) without a real tmux or
network anywhere.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from claude_swap.codex_handoff import (
    BOOT_POLL_ATTEMPTS,
    DEFAULT_WRAPUP_MESSAGE,
    STOP_POLL_ATTEMPTS,
    CodexHandoff,
    HandoffBlocked,
    HandoffConfig,
    HandoffError,
    TmuxDriver,
    build_handoff_config,
    codex_pids_in_pane,
    newest_session_id,
    run_watch,
)
from claude_swap.codex_usage import CodexUsage, UsageWindow
from claude_swap.exceptions import ClaudeSwitchError, ConfigError
from claude_swap.settings import set_setting

from tests.test_codex_accounts import codex_auth, make_store, write_live

TARGET = "agent:0.0"
SESSION_ID = "0198a2b4-1111-2222-3333-444455556666"


# -- fakes ---------------------------------------------------------------------


def ps_rows(*rows: tuple[int, int, str]) -> str:
    return "\n".join(f"{pid:>5} {ppid:>5} {cmd}" for pid, ppid, cmd in rows) + "\n"


PS_WITH_CODEX = ps_rows((100, 1, "-zsh"), (200, 100, "codex"))
PS_NO_CODEX = ps_rows((100, 1, "-zsh"))

WORKING_CAPTURE = "▌ Working (12s • Esc to interrupt)\n"
IDLE_CAPTURE = "› done. anything else?\n"
UPDATE_CAPTURE = "Update available! 0.148\n1) update now\n2) remind me\n3) skip\n"


class FakeRunner:
    """Records every command; serves scripted outputs for the read calls.

    ``capture_queue`` / ``ps_queue`` pop one entry per call and repeat the
    last one — so a scenario scripts the sequence of pane states and process
    tables the handoff will observe.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], str | None]] = []
        self.capture_queue: list[str] = [IDLE_CAPTURE]
        self.ps_queue: list[str] = [PS_WITH_CODEX]
        self.pane_pid_value = "100"
        self.pane_dead_value = "0"
        self.target_exists_rc = 0

    def _pop(self, queue: list[str]) -> str:
        return queue.pop(0) if len(queue) > 1 else queue[0]

    def __call__(self, args, input_text=None) -> subprocess.CompletedProcess:
        self.calls.append((tuple(args), input_text))
        if args[0] == "ps":
            return subprocess.CompletedProcess(
                args, 0, stdout=self._pop(self.ps_queue), stderr=""
            )
        assert args[0] == "tmux", f"unexpected program {args[0]}"
        sub = args[1]
        if sub == "capture-pane":
            return subprocess.CompletedProcess(
                args, 0, stdout=self._pop(self.capture_queue), stderr=""
            )
        if sub == "display-message":
            fmt = args[-1]
            if fmt == "#{pane_id}":
                return subprocess.CompletedProcess(
                    args, self.target_exists_rc, stdout="%1\n", stderr=""
                )
            if fmt == "#{pane_pid}":
                return subprocess.CompletedProcess(
                    args, 0, stdout=f"{self.pane_pid_value}\n", stderr=""
                )
            if fmt == "#{pane_dead}":
                return subprocess.CompletedProcess(
                    args, 0, stdout=f"{self.pane_dead_value}\n", stderr=""
                )
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    @property
    def tmux_calls(self) -> list[tuple[str, ...]]:
        return [c[0] for c in self.calls if c[0][0] == "tmux"]

    @property
    def mutating_calls(self) -> list[tuple[str, ...]]:
        """Every call that could change external state (anything beyond the
        read-only capture/display/ps calls)."""
        read_only = {"capture-pane", "display-message"}
        out = []
        for args, _ in self.calls:
            if args[0] == "ps":
                continue
            if args[0] == "tmux" and args[1] in read_only:
                continue
            out.append(args)
        return out


class FakeClock:
    """Injectable monotonic + sleep pair; sleep advances the clock."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


# -- scenario helpers ----------------------------------------------------------

USAGE_FRESH = CodexUsage(
    plan="plus",
    windows=(UsageWindow("5h", 10.0, None), UsageWindow("7d", 20.0, None)),
)
USAGE_NEARLY = CodexUsage(
    windows=(UsageWindow("5h", 50.0, None), UsageWindow("7d", 60.0, None)),
)
USAGE_EXHAUSTED = CodexUsage(
    windows=(UsageWindow("5h", 99.0, None), UsageWindow("7d", 40.0, None)),
)


def make_fetcher(usage_by_token: dict[str, object]):
    """A fetch_usage seam keyed by the credential's refresh token."""

    def fetch(credential_text: str):
        token = json.loads(credential_text)["tokens"]["refresh_token"]
        result = usage_by_token[token]
        if isinstance(result, Exception):
            raise result
        return result

    return fetch


def make_sessions(codex_home: Path, *entries: tuple[str, int]) -> None:
    """Create rollout files with explicit mtimes under sessions/2026/08/26."""
    day = codex_home / "sessions" / "2026" / "08" / "26"
    day.mkdir(parents=True, exist_ok=True)
    for name, mtime in entries:
        path = day / name
        path.write_text("{}\n", encoding="utf-8")
        os.utime(path, (mtime, mtime))


def make_config(codex_home: Path, **overrides) -> HandoffConfig:
    values = dict(
        tmux_target=TARGET,
        codex_home=codex_home,
        wrapup_message="Please wrap up now.",
        wrapup_grace_s=10,
        resume_args=(),
        threshold=90.0,
        window_thresholds={},
        poll_interval_s=300,
    )
    values.update(overrides)
    return HandoffConfig(**values)


def seeded_handoff(
    tmp_path: Path,
    *,
    usage_by_token: dict[str, object] | None = None,
    config_overrides: dict | None = None,
    runner: FakeRunner | None = None,
    clock: FakeClock | None = None,
):
    """A three-slot store (slot 1 active/exhausted, slot 2 fresh, slot 3
    nearly exhausted), a session tree, and a CodexHandoff over fakes."""
    store = make_store(tmp_path)
    write_live(store, codex_auth("two@example.com", token="b"))
    store.add(slot=2)
    write_live(store, codex_auth("three@example.com", token="c"))
    store.add(slot=3)
    write_live(store, codex_auth("one@example.com", token="a"))
    store.add(slot=1)  # live login; activeSlot == 1

    make_sessions(
        store.home,
        ("rollout-2026-08-26T10-00-00-99999999-aaaa-bbbb-cccc-dddddddddddd.jsonl", 1000),
        (f"rollout-2026-08-26T12-00-00-{SESSION_ID}.jsonl", 2000),
    )

    usage = usage_by_token or {
        "refresh-a": USAGE_EXHAUSTED,
        "refresh-b": USAGE_FRESH,
        "refresh-c": USAGE_NEARLY,
    }
    runner = runner if runner is not None else FakeRunner()
    clock = clock if clock is not None else FakeClock()
    emitted: list[str] = []
    handoff = CodexHandoff(
        store,
        make_config(store.home, **(config_overrides or {})),
        runner=runner,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        fetch_usage=make_fetcher(usage),
        emit=emitted.append,
    )
    return handoff, store, runner, clock, emitted


# -- unit pieces ---------------------------------------------------------------


class TestCodexPidsInPane:
    def test_finds_codex_descendants(self):
        ps = ps_rows(
            (100, 1, "-zsh"),
            (200, 100, "codex resume abc"),
            (300, 200, "/usr/bin/some-helper"),
            (400, 1, "codex"),  # unrelated pane — not under 100
        )
        assert codex_pids_in_pane(100, ps) == (200,)

    def test_pane_root_itself_can_be_codex(self):
        ps = ps_rows((100, 1, "/opt/bin/codex resume abc"))
        assert codex_pids_in_pane(100, ps) == (100,)

    def test_codex_prefixed_commands_do_not_match(self):
        ps = ps_rows((100, 1, "-zsh"), (200, 100, "codex-helper daemon"))
        assert codex_pids_in_pane(100, ps) == ()

    def test_garbage_lines_are_ignored(self):
        ps = "not a row\n  x y z\n" + ps_rows((100, 1, "codex"))
        assert codex_pids_in_pane(100, ps) == (100,)


class TestNewestSessionId:
    def test_picks_newest_rollout_by_mtime(self, tmp_path):
        make_sessions(
            tmp_path,
            (f"rollout-2026-08-26T12-00-00-{SESSION_ID}.jsonl", 2000),
            ("rollout-2026-08-26T10-00-00-99999999-aaaa-bbbb-cccc-dddddddddddd.jsonl", 1000),
        )
        assert newest_session_id(tmp_path) == SESSION_ID

    def test_ignores_files_without_a_uuid(self, tmp_path):
        make_sessions(
            tmp_path,
            (f"rollout-2026-08-26T10-00-00-{SESSION_ID}.jsonl", 1000),
            ("rollout-not-a-session.jsonl", 2000),
        )
        assert newest_session_id(tmp_path) == SESSION_ID

    def test_no_sessions_raises(self, tmp_path):
        with pytest.raises(HandoffError, match="rollout"):
            newest_session_id(tmp_path)

    def test_uuid_is_lowercased(self, tmp_path):
        make_sessions(
            tmp_path,
            (f"rollout-2026-08-26T10-00-00-{SESSION_ID.upper()}.jsonl", 1000),
        )
        assert newest_session_id(tmp_path) == SESSION_ID


class TestPickTarget:
    def test_picks_slot_with_most_margin_skipping_active(self, tmp_path):
        handoff, *_ = seeded_handoff(tmp_path)
        slot, label, margin = handoff.pick_target()
        assert (slot, label) == (2, "two@example.com")
        assert margin == 70.0  # 90 - max(10, 20)

    def test_skips_active_even_when_it_has_the_most_margin(self, tmp_path):
        handoff, *_ = seeded_handoff(
            tmp_path,
            usage_by_token={
                "refresh-a": USAGE_FRESH,  # active, best margin — still skipped
                "refresh-b": USAGE_NEARLY,
                "refresh-c": USAGE_EXHAUSTED,
            },
        )
        slot, _, _ = handoff.pick_target()
        assert slot == 2

    def test_refuses_when_every_other_slot_is_past_a_wall(self, tmp_path):
        handoff, *_ = seeded_handoff(
            tmp_path,
            usage_by_token={
                "refresh-a": USAGE_FRESH,
                "refresh-b": USAGE_EXHAUSTED,
                "refresh-c": USAGE_EXHAUSTED,
            },
        )
        with pytest.raises(HandoffBlocked, match="positive margin"):
            handoff.pick_target()

    def test_refuses_when_no_other_slot_exists(self, tmp_path):
        store = make_store(tmp_path)
        write_live(store, codex_auth("one@example.com", token="a"))
        store.add(slot=1)
        handoff = CodexHandoff(
            store,
            make_config(store.home),
            runner=FakeRunner(),
            fetch_usage=make_fetcher({"refresh-a": USAGE_FRESH}),
            emit=lambda _msg: None,
        )
        with pytest.raises(HandoffBlocked):
            handoff.pick_target()

    def test_unfetchable_slot_is_skipped_not_chosen(self, tmp_path):
        from claude_swap.codex_usage import CodexUsageError

        handoff, *_ = seeded_handoff(
            tmp_path,
            usage_by_token={
                "refresh-a": USAGE_EXHAUSTED,
                "refresh-b": CodexUsageError("http-401"),
                "refresh-c": USAGE_NEARLY,
            },
        )
        slot, _, _ = handoff.pick_target()
        assert slot == 3

    def test_per_window_walls_shape_the_choice(self, tmp_path):
        # Slot 2's weekly window is past a tight 7d wall even though its
        # global margin looks fine; slot 3 must win.
        handoff, *_ = seeded_handoff(
            tmp_path,
            usage_by_token={
                "refresh-a": USAGE_EXHAUSTED,
                "refresh-b": CodexUsage(
                    windows=(UsageWindow("5h", 10.0, None), UsageWindow("7d", 97.0, None))
                ),
                "refresh-c": USAGE_NEARLY,
            },
            config_overrides={"window_thresholds": {"7d": 96.0}},
        )
        slot, _, _ = handoff.pick_target()
        assert slot == 3


# -- the full sequence ---------------------------------------------------------


def run_happy_path(tmp_path, **kwargs):
    runner = kwargs.pop("runner", None) or FakeRunner()
    runner.capture_queue = [WORKING_CAPTURE, IDLE_CAPTURE]
    runner.ps_queue = [PS_WITH_CODEX, PS_NO_CODEX, PS_WITH_CODEX]
    handoff, store, runner, clock, emitted = seeded_handoff(
        tmp_path, runner=runner, **kwargs
    )
    rc = handoff.execute()
    return rc, handoff, store, runner, clock, emitted


class TestExecuteHappyPath:
    def test_returns_zero_and_swaps_to_the_best_slot(self, tmp_path):
        rc, handoff, store, runner, clock, emitted = run_happy_path(tmp_path)
        assert rc == 0
        live = json.loads(store.live_credential_path.read_text())
        assert live["tokens"]["refresh_token"] == "refresh-b"  # slot 2
        assert store._load()["activeSlot"] == 2

    def test_outgoing_credential_is_snapshotted_before_the_swap(self, tmp_path):
        # Simulate codex's in-place token rotation: the live file no longer
        # matches slot 1's snapshot at handoff time. The rotated content must
        # end up back in slot 1, not be lost.
        runner = FakeRunner()
        runner.capture_queue = [WORKING_CAPTURE, IDLE_CAPTURE]
        runner.ps_queue = [PS_WITH_CODEX, PS_NO_CODEX, PS_WITH_CODEX]
        handoff, store, runner, clock, emitted = seeded_handoff(
            tmp_path, runner=runner
        )
        rotated = codex_auth("one@example.com", token="a-rotated")
        write_live(store, rotated)
        handoff.execute()
        assert store._slot_path(1).read_text() == rotated

    def test_paste_discipline_load_then_bracketed_paste_then_enter(self, tmp_path):
        rc, handoff, store, runner, clock, emitted = run_happy_path(tmp_path)
        calls = runner.calls
        load_idx = next(
            i for i, (args, _) in enumerate(calls)
            if args[:2] == ("tmux", "load-buffer")
        )
        load_args, load_input = calls[load_idx]
        assert load_args == ("tmux", "load-buffer", "-b", "cswap-codex-handoff", "-")
        assert load_input == "Please wrap up now."
        paste_args, _ = calls[load_idx + 1]
        assert paste_args == (
            "tmux", "paste-buffer", "-p", "-d", "-b", "cswap-codex-handoff",
            "-t", TARGET,
        )
        enter_args, _ = calls[load_idx + 2]
        assert enter_args == ("tmux", "send-keys", "-t", TARGET, "Enter")

    def test_stop_sends_two_interrupts_before_polling(self, tmp_path):
        rc, handoff, store, runner, clock, emitted = run_happy_path(tmp_path)
        interrupts = [
            i for i, args in enumerate(runner.tmux_calls)
            if args == ("tmux", "send-keys", "-t", TARGET, "C-c")
        ]
        assert len(interrupts) == 2
        assert interrupts[1] == interrupts[0] + 1  # consecutive C-c pair

    def test_relaunch_is_a_literal_resume_command_then_enter(self, tmp_path):
        rc, handoff, store, runner, clock, emitted = run_happy_path(tmp_path)
        calls = runner.tmux_calls
        literal_idx = next(
            i for i, args in enumerate(calls)
            if args[:4] == ("tmux", "send-keys", "-t", TARGET) and "-l" in args
        )
        assert calls[literal_idx] == (
            "tmux", "send-keys", "-t", TARGET, "-l", f"codex resume {SESSION_ID}",
        )
        assert calls[literal_idx + 1] == ("tmux", "send-keys", "-t", TARGET, "Enter")

    def test_resume_args_are_appended_verbatim(self, tmp_path):
        rc, handoff, store, runner, clock, emitted = run_happy_path(
            tmp_path,
            config_overrides={"resume_args": ("--model", "gpt-5.6-sol")},
        )
        assert any(
            args == (
                "tmux", "send-keys", "-t", TARGET, "-l",
                f"codex resume {SESSION_ID} --model gpt-5.6-sol",
            )
            for args in runner.tmux_calls
        )

    def test_waits_out_the_working_indicator_before_stopping(self, tmp_path):
        rc, handoff, store, runner, clock, emitted = run_happy_path(tmp_path)
        # First capture showed the working indicator, so at least one
        # quiescence sleep happened before the C-c pair.
        assert clock.sleeps  # slept at least once
        assert any("quiescent" in line for line in emitted)

    def test_report_names_both_accounts_and_the_session(self, tmp_path):
        rc, handoff, store, runner, clock, emitted = run_happy_path(tmp_path)
        report = emitted[-1]
        assert "one@example.com" in report
        assert "two@example.com" in report
        assert SESSION_ID in report

    def test_dead_pane_is_respawned_with_the_resume_command(self, tmp_path):
        runner = FakeRunner()
        runner.capture_queue = [IDLE_CAPTURE]
        runner.ps_queue = [PS_WITH_CODEX, PS_NO_CODEX, PS_WITH_CODEX]
        runner.pane_dead_value = "1"
        handoff, store, runner, clock, emitted = seeded_handoff(
            tmp_path, runner=runner
        )
        handoff.execute()
        assert any(
            args[:2] == ("tmux", "respawn-pane")
            and args[2:] == ("-k", "-t", TARGET, f"codex resume {SESSION_ID}")
            for args in runner.tmux_calls
        )


class TestExecuteGraceAndUpdatePrompt:
    def test_grace_expiry_proceeds_anyway(self, tmp_path):
        runner = FakeRunner()
        runner.capture_queue = [WORKING_CAPTURE]  # never goes idle
        runner.ps_queue = [PS_WITH_CODEX, PS_NO_CODEX, PS_WITH_CODEX]
        handoff, store, runner, clock, emitted = seeded_handoff(
            tmp_path, runner=runner, config_overrides={"wrapup_grace_s": 6}
        )
        rc = handoff.execute()
        assert rc == 0
        assert any("expired" in line for line in emitted)
        # The swap still happened.
        live = json.loads(store.live_credential_path.read_text())
        assert live["tokens"]["refresh_token"] == "refresh-b"

    def test_update_prompt_is_answered_with_3_then_enter(self, tmp_path):
        runner = FakeRunner()
        runner.capture_queue = [IDLE_CAPTURE, UPDATE_CAPTURE, IDLE_CAPTURE]
        runner.ps_queue = [PS_WITH_CODEX, PS_NO_CODEX, PS_WITH_CODEX]
        handoff, store, runner, clock, emitted = seeded_handoff(
            tmp_path, runner=runner
        )
        rc = handoff.execute()
        assert rc == 0
        calls = runner.tmux_calls
        answer_idx = next(
            i for i, args in enumerate(calls)
            if args == ("tmux", "send-keys", "-t", TARGET, "-l", "3")
        )
        assert calls[answer_idx + 1] == ("tmux", "send-keys", "-t", TARGET, "Enter")
        # Answered exactly once, never with a bare Enter first.
        assert (
            sum(
                1 for args in calls
                if args == ("tmux", "send-keys", "-t", TARGET, "-l", "3")
            )
            == 1
        )


class TestExecuteFailClosed:
    def test_refuses_to_swap_while_codex_lives(self, tmp_path):
        runner = FakeRunner()
        runner.capture_queue = [IDLE_CAPTURE]
        runner.ps_queue = [PS_WITH_CODEX]  # codex never exits
        handoff, store, runner, clock, emitted = seeded_handoff(
            tmp_path, runner=runner
        )
        live_before = store.live_credential_path.read_text()
        with pytest.raises(HandoffError, match="still running"):
            handoff.execute()
        # The credential swap never happened.
        assert store.live_credential_path.read_text() == live_before
        assert store._load()["activeSlot"] == 1
        # And it did not give up early: the full poll budget was spent.
        ps_calls = [c for c in runner.calls if c[0][0] == "ps"]
        assert len(ps_calls) >= STOP_POLL_ATTEMPTS

    def test_refuses_when_the_pane_does_not_exist(self, tmp_path):
        runner = FakeRunner()
        runner.target_exists_rc = 1
        handoff, store, runner, clock, emitted = seeded_handoff(
            tmp_path, runner=runner
        )
        with pytest.raises(HandoffError, match="does not exist"):
            handoff.execute()
        assert runner.mutating_calls == []

    def test_refuses_when_no_codex_runs_in_the_pane(self, tmp_path):
        runner = FakeRunner()
        runner.ps_queue = [PS_NO_CODEX]
        handoff, store, runner, clock, emitted = seeded_handoff(
            tmp_path, runner=runner
        )
        with pytest.raises(HandoffError, match="no codex process"):
            handoff.execute()
        assert runner.mutating_calls == []

    def test_boot_timeout_raises_with_manual_recovery_hint(self, tmp_path):
        runner = FakeRunner()
        runner.capture_queue = [IDLE_CAPTURE]
        runner.ps_queue = [PS_WITH_CODEX, PS_NO_CODEX]  # never comes back
        handoff, store, runner, clock, emitted = seeded_handoff(
            tmp_path, runner=runner
        )
        with pytest.raises(HandoffError, match="codex resume"):
            handoff.execute()
        # Bounded: the boot loop polled its budget, not forever.
        boot_sleeps = [s for s in clock.sleeps if s > 0]
        assert len(boot_sleeps) < STOP_POLL_ATTEMPTS + BOOT_POLL_ATTEMPTS + 10

    def test_if_needed_fetch_failure_is_an_error_not_a_guess(self, tmp_path):
        from claude_swap.codex_usage import CodexUsageError

        handoff, store, runner, clock, emitted = seeded_handoff(
            tmp_path,
            usage_by_token={
                "refresh-a": CodexUsageError("timeout"),
                "refresh-b": USAGE_FRESH,
                "refresh-c": USAGE_FRESH,
            },
        )
        with pytest.raises(HandoffError, match="refusing to decide"):
            handoff.execute(if_needed=True)
        assert runner.calls == []


class TestIfNeeded:
    def test_below_every_wall_is_a_clean_noop(self, tmp_path):
        handoff, store, runner, clock, emitted = seeded_handoff(
            tmp_path,
            usage_by_token={
                "refresh-a": USAGE_NEARLY,  # margin +30
                "refresh-b": USAGE_FRESH,
                "refresh-c": USAGE_FRESH,
            },
        )
        rc = handoff.execute(if_needed=True)
        assert rc == 0
        assert any("no handoff needed" in line for line in emitted)
        assert runner.calls == []  # idempotent for cron: nothing touched

    def test_past_a_wall_triggers_the_handoff(self, tmp_path):
        runner = FakeRunner()
        runner.capture_queue = [IDLE_CAPTURE]
        runner.ps_queue = [PS_WITH_CODEX, PS_NO_CODEX, PS_WITH_CODEX]
        handoff, store, runner, clock, emitted = seeded_handoff(
            tmp_path, runner=runner
        )
        rc = handoff.execute(if_needed=True)
        assert rc == 0
        assert store._load()["activeSlot"] == 2

    def test_per_window_wall_triggers_before_the_global_threshold(self, tmp_path):
        runner = FakeRunner()
        runner.capture_queue = [IDLE_CAPTURE]
        runner.ps_queue = [PS_WITH_CODEX, PS_NO_CODEX, PS_WITH_CODEX]
        handoff, store, runner, clock, emitted = seeded_handoff(
            tmp_path,
            runner=runner,
            usage_by_token={
                # 7d at 97%: below the global 99 threshold, past the 7d=96 wall.
                "refresh-a": CodexUsage(
                    windows=(UsageWindow("5h", 10.0, None), UsageWindow("7d", 97.0, None))
                ),
                "refresh-b": USAGE_FRESH,
                "refresh-c": USAGE_FRESH,
            },
            config_overrides={
                "threshold": 99.0,
                "window_thresholds": {"7d": 96.0},
            },
        )
        rc = handoff.execute(if_needed=True)
        assert rc == 0
        assert store._load()["activeSlot"] == 2


class TestDryRun:
    def test_prints_the_plan_and_acts_on_nothing(self, tmp_path):
        handoff, store, runner, clock, emitted = seeded_handoff(tmp_path)
        live_before = store.live_credential_path.read_text()
        slots_before = {
            n: store._slot_path(n).read_text() for n in (1, 2, 3)
        }
        rc = handoff.execute(dry_run=True)
        assert rc == 0
        assert runner.calls == []  # zero tmux/ps calls, mutating or not
        assert store.live_credential_path.read_text() == live_before
        assert {
            n: store._slot_path(n).read_text() for n in (1, 2, 3)
        } == slots_before
        assert store._load()["activeSlot"] == 1
        plan = "\n".join(emitted)
        assert TARGET in plan
        assert SESSION_ID in plan
        assert "slot 2" in plan


class TestDefaultWrapupMessage:
    def test_covers_the_contract(self):
        for phrase in ("usage limit", "atomic step", "resume", "long-running"):
            assert phrase in DEFAULT_WRAPUP_MESSAGE


class TestBuildHandoffConfig:
    def test_refuses_without_tmux_target(self, tmp_path):
        with pytest.raises(ConfigError, match="codex.tmux_target"):
            build_handoff_config(tmp_path)

    def test_resolves_settings_and_defaults(self, tmp_path, temp_home):
        set_setting(tmp_path, "codex.tmux_target", TARGET)
        config = build_handoff_config(tmp_path)
        assert config.tmux_target == TARGET
        assert config.codex_home == temp_home / ".codex"
        assert config.wrapup_message == DEFAULT_WRAPUP_MESSAGE
        assert config.wrapup_grace_s == 120
        assert config.resume_args == ()
        assert config.threshold == 90.0
        assert config.window_thresholds == {}
        assert config.poll_interval_s == 300

    def test_explicit_settings_flow_through(self, tmp_path, temp_home):
        set_setting(tmp_path, "codex.tmux_target", TARGET)
        set_setting(tmp_path, "codex.home", str(temp_home / "ch"))
        set_setting(tmp_path, "codex.wrapup_message", "Custom wrap-up.")
        set_setting(tmp_path, "codex.wrapup_grace_s", "45")
        set_setting(tmp_path, "codex.resume_args", "--model gpt-5.6-sol")
        set_setting(tmp_path, "codex.poll_interval_s", "60")
        set_setting(tmp_path, "autoswitch.threshold", "80")
        set_setting(tmp_path, "autoswitch.thresholds", "5h=95,7d=98")
        config = build_handoff_config(tmp_path)
        assert config.codex_home == temp_home / "ch"
        assert config.wrapup_message == "Custom wrap-up."
        assert config.wrapup_grace_s == 45
        assert config.resume_args == ("--model", "gpt-5.6-sol")
        assert config.poll_interval_s == 60
        assert config.threshold == 80.0
        assert config.window_thresholds == {"5h": 95.0, "7d": 98.0}

    def test_malformed_thresholds_fail_loud(self, tmp_path):
        # A wall with a typo can enter the file by hand-editing; the handoff
        # must refuse rather than silently gate nothing.
        (tmp_path / "settings.json").write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "autoswitch": {"thresholds": "7d=oops"},
                    "codex": {"tmux_target": TARGET},
                }
            )
        )
        with pytest.raises(ConfigError, match="autoswitch.thresholds"):
            build_handoff_config(tmp_path)


class TestConfigValidationViaSetSetting:
    """`cswap config set` strictness for the codex keys."""

    def test_accepts_valid_values(self, tmp_path):
        assert set_setting(tmp_path, "codex.tmux_target", "a:0.0") == "a:0.0"
        assert set_setting(tmp_path, "codex.wrapup_grace_s", "0") == 0
        assert set_setting(tmp_path, "codex.wrapup_grace_s", "3600") == 3600
        assert set_setting(tmp_path, "codex.poll_interval_s", "15") == 15

    @pytest.mark.parametrize(
        "key,value",
        [
            ("codex.wrapup_grace_s", "-1"),
            ("codex.wrapup_grace_s", "3601"),
            ("codex.wrapup_grace_s", "ten"),
            ("codex.poll_interval_s", "5"),
            ("codex.tmux_target", ""),
        ],
    )
    def test_rejects_invalid_values(self, tmp_path, key, value):
        with pytest.raises(ConfigError):
            set_setting(tmp_path, key, value)

    def test_unknown_codex_key_is_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="unknown setting"):
            set_setting(tmp_path, "codex.pane", "a:0.0")


class TestTmuxDriverErrors:
    def test_nonzero_tmux_exit_raises_with_the_subcommand(self):
        def failing_runner(args, input_text=None):
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="boom")

        driver = TmuxDriver(TARGET, runner=failing_runner)
        with pytest.raises(HandoffError, match="capture-pane"):
            driver.capture()

    def test_unparseable_pane_pid_raises(self):
        def odd_runner(args, input_text=None):
            return subprocess.CompletedProcess(args, 0, stdout="nope\n", stderr="")

        driver = TmuxDriver(TARGET, runner=odd_runner)
        with pytest.raises(HandoffError, match="pane pid"):
            driver.pane_pid()


class TestRunWatch:
    class _StubHandoff:
        def __init__(self, config, outcomes):
            self.config = config
            self.outcomes = list(outcomes)
            self.calls: list[dict] = []
            self.logs: list[str] = []

        def execute(self, *, if_needed=False, dry_run=False):
            self.calls.append({"if_needed": if_needed, "dry_run": dry_run})
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        def _log(self, message):
            self.logs.append(message)

    def test_polls_if_needed_every_interval(self, tmp_path):
        config = make_config(tmp_path, poll_interval_s=77)
        stub = self._StubHandoff(config, [0, 0, 0])
        sleeps: list[float] = []
        run_watch(stub, sleep=sleeps.append, max_ticks=3)
        assert [c["if_needed"] for c in stub.calls] == [True, True, True]
        assert sleeps == [77, 77]  # no trailing sleep after the last tick

    def test_errors_are_logged_and_the_loop_keeps_polling(self, tmp_path):
        config = make_config(tmp_path, poll_interval_s=10)
        stub = self._StubHandoff(
            config, [ClaudeSwitchError("usage fetch blew up"), 0]
        )
        run_watch(stub, sleep=lambda _s: None, max_ticks=2)
        assert len(stub.calls) == 2
        assert any("usage fetch blew up" in line for line in stub.logs)
