"""Codex session handoff: rotate a LIVE codex TUI onto a fresh account.

Why this exists — and why it is not a hot swap. Measured 2026-08-26 (codex
0.147): codex reads ``auth.json`` ONCE at startup and caches the auth in
memory. A running TUI ignores an on-disk swap entirely — a turn succeeded
with a server-rejected credential sitting on disk — and only a restart
re-reads the file. So rotating a live codex session MUST be:

    warn the agent → stop the TUI → swap auth.json → ``codex resume <id>``

The critical simplifying fact: cswap swaps credentials INSIDE one
``CODEX_HOME``, so sessions, history, and codex's ``/goal`` store all
persist across the swap — ``codex resume`` restores the full working
context on the new account.

Every step is fail-closed:

1.  ``codex.tmux_target`` must be configured — a pane is never guessed.
2.  ``--if-needed`` evaluates the ACTIVE account against the same wall
    machinery the Claude side uses (``oauth.switch_margin`` with
    ``autoswitch.threshold`` / ``autoswitch.thresholds``; codex windows are
    "5h" and "7d"). Below every wall → exit 0, no action (cron-idempotent).
3.  The target is the registered slot with the MOST margin to its walls;
    if no other slot has positive margin the handoff refuses — it never
    lands on an exhausted account.
4.  The wrap-up message is pasted with bracketed paste (``load-buffer`` +
    ``paste-buffer -p``, then a separate Enter) so multiline text cannot
    submit early.
5.  Quiescence is awaited up to ``codex.wrapup_grace_s``; on expiry the
    handoff proceeds anyway — the resume carries the context.
6.  The TUI is stopped with C-c (twice, with pauses) and the pane's process
    tree is POLLED until the codex process is actually gone; the credential
    swap never runs while it lives (the dying TUI could write auth.json).
7.  The swap snapshots the outgoing credential back into its slot first,
    then restores the target slot — same discipline as the Claude switch.
8.  The relaunch answers codex's "Update available" prompt with option 3
    (skip), never a bare Enter — option 1 runs the updater and has killed
    real boots on our fleet.

All external effects (tmux, the process table, network, the clock) come
through injectable seams so the whole sequence is unit-testable with a fake
runner. The quiescence indicator and the update-prompt handling are
heuristics distilled from production incidents; see the README section for
operational notes.
"""

from __future__ import annotations

import re
import shlex
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from claude_swap import oauth
from claude_swap.codex_accounts import CodexAccountStore, resolve_codex_home
from claude_swap.codex_usage import (
    CodexUsage,
    CodexUsageError,
    as_usage_dict,
    fetch_usage_from_text,
)
from claude_swap.exceptions import ClaudeSwitchError, ConfigError
from claude_swap.settings import (
    load_codex_settings,
    load_settings,
    parse_window_thresholds,
)

DEFAULT_WRAPUP_MESSAGE = (
    "[cswap handoff] This account is nearing its usage limit, so this codex "
    "session will be stopped and resumed on a fresh account shortly. Please: "
    "finish the atomic step currently in flight, write any in-flight state "
    "to disk or to your issue/PR, and do not start new long-running work. "
    "The session will be resumed with full context after the account swap."
)

# The codex TUI renders "(... Esc to interrupt)" on its status line for the
# whole time a turn is running. Its absence from a pane capture is the
# quiescence heuristic — a plain-text match, so an agent that happens to
# print the same phrase produces a false "still working" (benign: the grace
# window expires and the handoff proceeds anyway).
WORKING_INDICATOR = "Esc to interrupt"

# codex's self-update prompt. Answer "3" (skip this version) — option 1 runs
# the updater in place of the session and has killed real boots on our fleet;
# a bare Enter selects it.
UPDATE_PROMPT_MARKER = "Update available"
UPDATE_PROMPT_ANSWER = "3"

_PASTE_BUFFER_NAME = "cswap-codex-handoff"

QUIESCE_POLL_S = 2.0
STOP_PAUSE_S = 1.0
STOP_POLL_ATTEMPTS = 15
STOP_POLL_S = 1.0
BOOT_POLL_ATTEMPTS = 30
BOOT_POLL_S = 2.0

_SESSION_ID_RE = re.compile(
    r"rollout-.*?([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
    r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\.jsonl$"
)


class HandoffError(ClaudeSwitchError):
    """A handoff step failed (or refused to proceed); nothing after it ran."""


class HandoffBlocked(HandoffError):
    """A handoff is wanted but no viable target slot exists."""


class Runner(Protocol):
    """Subprocess seam: run ``args``, optionally with text on stdin."""

    def __call__(
        self, args: list[str], input_text: str | None = None
    ) -> subprocess.CompletedProcess: ...


def _default_runner(
    args: list[str], input_text: str | None = None
) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603 — fixed argv, never shell=True
        args, input=input_text, capture_output=True, text=True
    )


class TmuxDriver:
    """The handoff's entire tmux surface, one injectable-runner call at a time.

    Every method shells out to ``tmux`` with a fixed argv (never a shell
    string) against the one configured target pane. A non-zero exit from any
    checked call raises :class:`HandoffError` — a vanished pane must stop
    the sequence, not be papered over.
    """

    def __init__(self, target: str, runner: Runner | None = None) -> None:
        self.target = target
        self._runner = runner if runner is not None else _default_runner

    def _tmux(
        self,
        *args: str,
        input_text: str | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        result = self._runner(["tmux", *args], input_text)
        if check and result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise HandoffError(
                f"tmux {args[0]} failed (rc {result.returncode}"
                f"{': ' + stderr if stderr else ''}) — target pane "
                f"'{self.target}'"
            )
        return result

    def target_exists(self) -> bool:
        result = self._tmux(
            "display-message", "-p", "-t", self.target, "#{pane_id}",
            check=False,
        )
        return result.returncode == 0

    def capture(self) -> str:
        result = self._tmux("capture-pane", "-p", "-t", self.target)
        return result.stdout or ""

    def paste_message(self, text: str) -> None:
        """Deliver multiline text as ONE literal paste, then submit.

        ``paste-buffer -p`` (bracketed paste) keeps embedded newlines from
        submitting the message line-by-line mid-composition; the Enter that
        actually submits is a separate keypress afterwards.
        """
        self._tmux("load-buffer", "-b", _PASTE_BUFFER_NAME, "-", input_text=text)
        self._tmux(
            "paste-buffer", "-p", "-d", "-b", _PASTE_BUFFER_NAME,
            "-t", self.target,
        )
        self._tmux("send-keys", "-t", self.target, "Enter")

    def send_interrupt(self) -> None:
        self._tmux("send-keys", "-t", self.target, "C-c")

    def send_literal(self, text: str) -> None:
        # ``-l``: literal keys, so a command line is not interpreted as
        # key names ("Enter", "C-c", ...).
        self._tmux("send-keys", "-t", self.target, "-l", text)

    def send_enter(self) -> None:
        self._tmux("send-keys", "-t", self.target, "Enter")

    def pane_pid(self) -> int:
        result = self._tmux(
            "display-message", "-p", "-t", self.target, "#{pane_pid}"
        )
        text = (result.stdout or "").strip()
        try:
            return int(text)
        except ValueError:
            raise HandoffError(
                f"could not read the pane pid for '{self.target}' "
                f"(tmux said {text!r})"
            ) from None

    def pane_dead(self) -> bool:
        result = self._tmux(
            "display-message", "-p", "-t", self.target, "#{pane_dead}"
        )
        return (result.stdout or "").strip() == "1"

    def respawn(self, command: list[str]) -> None:
        self._tmux(
            "respawn-pane", "-k", "-t", self.target, shlex.join(command)
        )

    def list_processes(self) -> str:
        """The process table (``pid ppid command`` rows) via the same runner
        seam, so tests can fake it. Not a tmux command, but the pane's
        process tree is part of the same external surface."""
        result = self._runner(["ps", "-ax", "-o", "pid=,ppid=,command="], None)
        if result.returncode != 0:
            raise HandoffError(
                f"ps failed (rc {result.returncode}); cannot verify the "
                "codex process state"
            )
        return result.stdout or ""


def _is_codex_command(command: str) -> bool:
    head = command.split(None, 1)[0] if command.split() else ""
    return head == "codex" or head.endswith("/codex")


def codex_pids_in_pane(pane_pid: int, ps_output: str) -> tuple[int, ...]:
    """Pids of codex processes in the pane's process tree (root included).

    ``ps_output`` is ``ps -ax -o pid=,ppid=,command=`` text. The pane pid is
    usually the shell with codex as a child, but a respawned pane can run
    codex as the root itself — both shapes are covered.
    """
    children: dict[int, list[int]] = defaultdict(list)
    command_by_pid: dict[int, str] = {}
    for line in ps_output.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid, ppid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        children[ppid].append(pid)
        command_by_pid[pid] = parts[2]

    hits: list[int] = []
    seen: set[int] = set()
    queue = [pane_pid]
    while queue:
        pid = queue.pop()
        if pid in seen:
            continue
        seen.add(pid)
        queue.extend(children.get(pid, []))
        if _is_codex_command(command_by_pid.get(pid, "")):
            hits.append(pid)
    return tuple(sorted(hits))


def newest_session_id(codex_home: Path) -> str:
    """The live session's id: the uuid in the NEWEST ``rollout-*.jsonl``
    under ``$CODEX_HOME/sessions/`` (codex appends to it continuously, so
    recency identifies the active session)."""
    sessions_dir = codex_home / "sessions"
    candidates = [
        path
        for path in sessions_dir.rglob("rollout-*.jsonl")
        if _SESSION_ID_RE.search(path.name)
    ]
    if not candidates:
        raise HandoffError(
            f"no rollout-*.jsonl session files under {sessions_dir} — "
            "is a codex session running with this codex.home?"
        )
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    match = _SESSION_ID_RE.search(newest.name)
    assert match is not None  # filtered above
    return match.group(1).lower()


def codex_switch_margin(
    usage: CodexUsage,
    threshold: float,
    window_thresholds: dict[str, float] | None,
) -> float | None:
    """Points until the nearest switch wall for a codex account; negative =
    past it. Pure reuse of ``oauth.switch_margin`` over the normalized shape
    ``as_usage_dict`` emits — ``models=("all",)`` so any unusual window the
    API reports (folded in as scoped) gates too."""
    return oauth.switch_margin(
        as_usage_dict(usage), ("all",), threshold, window_thresholds
    )


@dataclass(frozen=True)
class HandoffConfig:
    """Everything a handoff run needs, resolved once up front."""

    tmux_target: str
    codex_home: Path
    wrapup_message: str
    wrapup_grace_s: int
    resume_args: tuple[str, ...]
    threshold: float
    window_thresholds: dict[str, float]
    poll_interval_s: int


def build_handoff_config(backup_root: Path) -> HandoffConfig:
    """Resolve settings into a :class:`HandoffConfig`, refusing loudly on
    anything that would make the handoff guess."""
    codex = load_codex_settings(backup_root)
    autoswitch = load_settings(backup_root)
    if not codex.tmux_target:
        raise ConfigError(
            "codex.tmux_target is not set — the handoff never guesses which "
            "pane holds the live agent. Set it with: cswap config set "
            "codex.tmux_target '<session>:<window>.<pane>'"
        )
    try:
        window_thresholds = parse_window_thresholds(autoswitch.thresholds)
    except ValueError as e:
        # Strict, like the auto engine: a wall with a typo must stop the
        # handoff, not silently gate nothing.
        raise ConfigError(f"autoswitch.thresholds: {e}") from e
    return HandoffConfig(
        tmux_target=codex.tmux_target,
        codex_home=resolve_codex_home(codex.home),
        wrapup_message=codex.wrapup_message or DEFAULT_WRAPUP_MESSAGE,
        wrapup_grace_s=codex.wrapup_grace_s,
        resume_args=tuple(shlex.split(codex.resume_args or "")),
        threshold=autoswitch.threshold,
        window_thresholds=window_thresholds,
        poll_interval_s=codex.poll_interval_s,
    )


def _describe_windows(usage: CodexUsage) -> str:
    return ", ".join(f"{w.label} {w.pct:.0f}%" for w in usage.windows) or "no windows"


class CodexHandoff:
    """One handoff run (or one ``--if-needed`` evaluation) over a store, a
    tmux pane, and a usage fetcher — all injectable."""

    def __init__(
        self,
        store: CodexAccountStore,
        config: HandoffConfig,
        *,
        runner: Runner | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        fetch_usage: Callable[[str], CodexUsage] = fetch_usage_from_text,
        emit: Callable[[str], None] = print,
    ) -> None:
        self.store = store
        self.config = config
        self.tmux = TmuxDriver(config.tmux_target, runner=runner)
        self._sleep = sleep
        self._monotonic = monotonic
        self._fetch_usage = fetch_usage
        self._emit = emit

    # -- evaluation ---------------------------------------------------------

    def active_wall_state(self) -> tuple[float, CodexUsage]:
        """(margin, usage) for the ACTIVE account, from the live auth.json.

        Fail-closed by contract: an unfetchable or windowless usage raises
        rather than guessing in either direction — ``--if-needed`` must not
        trigger a disruptive handoff on no evidence, and must equally not
        report "no handoff needed" on no evidence.
        """
        live = self.store.read_live()
        try:
            usage = self._fetch_usage(live)
        except CodexUsageError as e:
            raise HandoffError(
                f"could not fetch the active codex account's usage ({e}); "
                "refusing to decide a handoff on unknown usage"
            ) from e
        margin = codex_switch_margin(
            usage, self.config.threshold, self.config.window_thresholds
        )
        if margin is None:
            raise HandoffError(
                "the usage API reported no rate-limit windows for the active "
                "account; refusing to decide a handoff on unknown usage"
            )
        return margin, usage

    def pick_target(self) -> tuple[int, str, float]:
        """The registered slot with the most margin to its walls, skipping
        the active account. Slots whose usage cannot be fetched are skipped
        (unknown margin is not positive margin). Raises
        :class:`HandoffBlocked` when nothing has positive margin."""
        live_slot, rows = self.store.list_accounts()
        best: tuple[int, str, float] | None = None
        reasons: list[str] = []
        for row in rows:
            if row.number == live_slot:
                continue
            try:
                content = self.store._slot_path(row.number).read_text(
                    encoding="utf-8"
                )
                usage = self._fetch_usage(content)
            except (OSError, CodexUsageError) as e:
                reason = e if isinstance(e, CodexUsageError) else "unreadable"
                reasons.append(f"slot {row.number} ({row.label}): skipped ({reason})")
                continue
            margin = codex_switch_margin(
                usage, self.config.threshold, self.config.window_thresholds
            )
            if margin is None:
                reasons.append(
                    f"slot {row.number} ({row.label}): skipped (no windows)"
                )
                continue
            reasons.append(
                f"slot {row.number} ({row.label}): margin {margin:+.1f} "
                f"[{_describe_windows(usage)}]"
            )
            if margin > 0 and (best is None or margin > best[2]):
                best = (row.number, row.label, margin)
        if best is None:
            detail = "; ".join(reasons) if reasons else "no other slots registered"
            raise HandoffBlocked(
                "no codex slot with positive margin to hand off to — "
                f"refusing to land on an exhausted account. {detail}. "
                "Register another account with 'cswap codex add'."
            )
        return best

    # -- steps --------------------------------------------------------------

    def _await_quiescence(self) -> str:
        """Poll the pane until codex's working indicator clears, bounded by
        ``wrapup_grace_s``. Returns a one-line outcome for the report."""
        grace = self.config.wrapup_grace_s
        deadline = self._monotonic() + grace
        while True:
            if WORKING_INDICATOR not in self.tmux.capture():
                return "agent quiescent before the grace expired"
            if self._monotonic() >= deadline:
                return (
                    f"grace of {grace}s expired with the agent still working; "
                    "proceeded anyway (resume carries the context)"
                )
            self._sleep(QUIESCE_POLL_S)

    def _stop_tui(self, pane_pid: int) -> None:
        """C-c twice with pauses, then poll the pane's process tree until the
        codex process is gone. Refuses (raises) if it will not die — the
        swap must never race a live TUI that could rewrite auth.json."""
        self.tmux.send_interrupt()
        self._sleep(STOP_PAUSE_S)
        self.tmux.send_interrupt()
        for attempt in range(STOP_POLL_ATTEMPTS):
            pids = codex_pids_in_pane(pane_pid, self.tmux.list_processes())
            if not pids:
                return
            if attempt in (4, 9):
                # Two more nudges midway through the budget; some TUI states
                # eat the first pair.
                self.tmux.send_interrupt()
                self._sleep(STOP_PAUSE_S)
                self.tmux.send_interrupt()
            self._sleep(STOP_POLL_S)
        raise HandoffError(
            f"codex (pid(s) {', '.join(map(str, pids))}) is still running in "
            f"pane '{self.config.tmux_target}' after C-c and "
            f"{STOP_POLL_ATTEMPTS} checks — refusing to swap credentials "
            "under a live TUI. Stop it manually and re-run."
        )

    def _swap_credentials(self, outgoing_slot: int | None, target_slot: int) -> None:
        if outgoing_slot is not None:
            # Snapshot the outgoing login back into its own slot first (codex
            # rotates tokens in place, so the stored snapshot is stale).
            self.store.snapshot_slot(outgoing_slot)
        self.store.switch(str(target_slot))

    def _relaunch(self, session_id: str) -> None:
        command = ["codex", "resume", session_id, *self.config.resume_args]
        if self.tmux.pane_dead():
            # The pane's root process WAS codex (no shell underneath): bring
            # the pane back running the resume directly.
            self.tmux.respawn(command)
        else:
            self.tmux.send_literal(shlex.join(command))
            self.tmux.send_enter()

    def _await_boot(self) -> None:
        """Wait for the resumed TUI, answering the update prompt with '3'
        (skip) — NEVER a bare Enter, which selects option 1 and runs the
        updater in place of the session (measured incident)."""
        answered_update_prompt = False
        for _ in range(BOOT_POLL_ATTEMPTS):
            captured = self.tmux.capture()
            if UPDATE_PROMPT_MARKER in captured and not answered_update_prompt:
                self._log("update prompt detected — answering "
                          f"'{UPDATE_PROMPT_ANSWER}' (skip)")
                self.tmux.send_literal(UPDATE_PROMPT_ANSWER)
                self.tmux.send_enter()
                answered_update_prompt = True
                self._sleep(BOOT_POLL_S)
                continue
            # The pane pid changes across a respawn; re-resolve every poll.
            if codex_pids_in_pane(self.tmux.pane_pid(), self.tmux.list_processes()):
                return
            self._sleep(BOOT_POLL_S)
        raise HandoffError(
            "the resumed codex TUI did not come up within "
            f"{int(BOOT_POLL_ATTEMPTS * BOOT_POLL_S)}s. Credentials are "
            "already swapped; resume manually with 'codex resume <id>' in "
            f"pane '{self.config.tmux_target}'."
        )

    # -- orchestration ------------------------------------------------------

    def _log(self, message: str) -> None:
        self._emit(message)

    def execute(self, *, if_needed: bool = False, dry_run: bool = False) -> int:
        """Run the handoff. Returns 0 on success or a clean no-op; raises
        :class:`HandoffError` / :class:`HandoffBlocked` otherwise."""
        if if_needed:
            margin, usage = self.active_wall_state()
            state = (
                f"active account: margin {margin:+.1f} to the nearest wall "
                f"[{_describe_windows(usage)}; threshold "
                f"{self.config.threshold:.0f}%"
                + (
                    f", walls {self.config.window_thresholds}"
                    if self.config.window_thresholds
                    else ""
                )
                + "]"
            )
            self._log(state)
            if margin > 0:
                self._log("no handoff needed")
                return 0
            self._log("wall reached — handing off")

        old_slot, old_label = self._resolve_outgoing()
        target_slot, target_label, target_margin = self.pick_target()
        session_id = newest_session_id(self.config.codex_home)

        if dry_run:
            self._log("dry run — no action taken. Plan:")
            self._log(f"  pane:        {self.config.tmux_target}")
            self._log(f"  session id:  {session_id}")
            self._log(
                f"  account:     slot {old_slot} ({old_label}) -> "
                f"slot {target_slot} ({target_label}, margin "
                f"{target_margin:+.1f})"
            )
            self._log(
                "  resume cmd:  "
                + shlex.join(
                    ["codex", "resume", session_id, *self.config.resume_args]
                )
            )
            return 0

        if not self.tmux.target_exists():
            raise HandoffError(
                f"tmux target '{self.config.tmux_target}' does not exist — "
                "check codex.tmux_target"
            )
        pane_pid = self.tmux.pane_pid()
        if not codex_pids_in_pane(pane_pid, self.tmux.list_processes()):
            raise HandoffError(
                f"no codex process found in pane '{self.config.tmux_target}' "
                "— is the TUI running there? Refusing a blind swap; use "
                "'cswap codex switch' for a cold rotation."
            )

        self._log(f"pasting wrap-up message to {self.config.tmux_target}")
        self.tmux.paste_message(self.config.wrapup_message)

        self._log(
            f"waiting up to {self.config.wrapup_grace_s}s for quiescence"
        )
        grace_outcome = self._await_quiescence()
        self._log(grace_outcome)

        self._log("stopping the codex TUI")
        self._stop_tui(pane_pid)

        self._log(
            f"swapping auth.json: slot {old_slot} ({old_label}) -> "
            f"slot {target_slot} ({target_label})"
        )
        self._swap_credentials(old_slot, target_slot)

        self._log(f"relaunching: codex resume {session_id}")
        self._relaunch(session_id)
        self._await_boot()

        self._log(
            f"handoff complete: {old_label or f'slot {old_slot}'} -> "
            f"{target_label} (slot {target_slot}), session {session_id}; "
            f"{grace_outcome}"
        )
        return 0

    def _resolve_outgoing(self) -> tuple[int | None, str]:
        live_slot, rows = self.store.list_accounts()
        if live_slot is None:
            return None, "unmanaged login"
        label = next((r.label for r in rows if r.number == live_slot), "")
        return live_slot, label


def run_watch(
    handoff: CodexHandoff,
    *,
    sleep: Callable[[float], None] = time.sleep,
    max_ticks: int | None = None,
) -> int:
    """The ``cswap codex watch`` loop: every ``codex.poll_interval_s``, run
    the ``--if-needed`` evaluation and hand off when a wall is reached.

    Deliberately NOT wired into the Claude ``cswap auto`` engine: the codex
    rotation is a stop-swap-resume sequence with its own failure modes, so
    the two loops stay separate and composable. Errors are logged and the
    loop keeps polling (a transient fetch failure must not kill an
    unattended watcher); ``max_ticks`` bounds the loop for tests.
    """
    interval = handoff.config.poll_interval_s
    ticks = 0
    while max_ticks is None or ticks < max_ticks:
        ticks += 1
        try:
            handoff.execute(if_needed=True)
        except ClaudeSwitchError as e:
            handoff._log(f"error: {e}")
        if max_ticks is not None and ticks >= max_ticks:
            break
        sleep(interval)
    return 0
