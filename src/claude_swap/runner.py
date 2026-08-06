"""Persistent autonomous mission runner with model-priority fallback.

``PersistentRunner`` is a headless supervisor that keeps working on a mission
until the operator stops it. It walks a configurable priority list of backends:

    Claude Code -> Codex CLI -> Kimi Code -> OpenRouter Qwen 3.8
    -> opencode free models -> local models (/rig ollama/llama-server)

When the active backend returns a quota, rate-limit, or auth error, the runner
switches to the next option. Subscription backends try every stored account
before falling back to the next backend tier.

State and logs live under ``<backup_root>/runner/`` so the daemon survives
process restarts and can be inspected with ``cswap persistent status/logs``.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

from claude_swap import paths
from claude_swap.exceptions import ClaudeSwitchError
from claude_swap.fsutil import replace_with_retry
from claude_swap.models import get_timestamp
from claude_swap.settings import (
    atomic_write_json,
    load_settings as load_autoswitch_settings,
)
from claude_swap.switcher import ClaudeAccountSwitcher
from claude_swap.tool_switcher import TOOLS, ToolAccountStore, live_credential_path as tool_live_credential_path

_logger = logging.getLogger("claude-swap")

RUNNER_DIRNAME = "runner"
STATE_FILENAME = "state.json"
MISSION_FILENAME = "mission.md"
LOG_FILENAME = "runner.log"
STOP_FILENAME = "stop"
PID_FILENAME = "daemon.pid"

DEFAULT_PRIORITY = [
    "claude",
    "codex",
    "kimi",
    "openrouter-qwen",
    "opencode-free",
    "local",
]

DEFAULT_FREE_MODELS = [
    "opencode/deepseek-v4-flash-free",
    "opencode/laguna-s-2.1-free",
    "opencode/ling-3.0-flash-free",
    "opencode/longcat-2.0-free",
    "opencode/nemotron-3-ultra-free",
    "opencode/north-mini-code-free",
]

DEFAULT_OPENROUTER_MODEL = "qwen/qwen3.8-max"
DEFAULT_LOCAL_ENDPOINTS = [
    {"url": "http://127.0.0.1:11434/v1", "type": "ollama"},
    {"url": "http://127.0.0.1:8080/v1", "type": "llama-server"},
]


class StepResult(enum.Enum):
    """Outcome of one backend step."""

    SUCCESS = "success"
    QUOTA_ERROR = "quota_error"
    RATE_LIMIT = "rate_limit"
    AUTH_ERROR = "auth_error"
    OTHER_ERROR = "other_error"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class RunnerSettings:
    """User-tunable knobs for the persistent runner."""

    priority: tuple[str, ...] = field(default_factory=lambda: tuple(DEFAULT_PRIORITY))
    retry_attempts: int = 2
    cooldown_seconds: float = 30.0
    auto_loop: bool = True
    max_history_turns: int = 20
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL
    free_models: tuple[str, ...] = field(default_factory=lambda: tuple(DEFAULT_FREE_MODELS))
    local_endpoints: tuple[dict, ...] = field(
        default_factory=lambda: tuple({"url": e["url"], "type": e["type"]} for e in DEFAULT_LOCAL_ENDPOINTS)
    )


@dataclass
class MissionState:
    """Serialized daemon state."""

    mission: str = ""
    backend_index: int = 0
    account_index: dict[str, int] = field(default_factory=dict)
    history: list[dict[str, str]] = field(default_factory=list)
    status: str = "idle"  # idle | running | paused | completed | failed
    last_error: str | None = None
    last_backend: str | None = None
    started_at: str | None = None
    updated_at: str | None = None
    total_steps: int = 0

    def to_dict(self) -> dict:
        return {
            "mission": self.mission,
            "backendIndex": self.backend_index,
            "accountIndex": self.account_index,
            "history": self.history,
            "status": self.status,
            "lastError": self.last_error,
            "lastBackend": self.last_backend,
            "startedAt": self.started_at,
            "updatedAt": self.updated_at,
            "totalSteps": self.total_steps,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MissionState":
        return cls(
            mission=data.get("mission", ""),
            backend_index=data.get("backendIndex", 0),
            account_index=data.get("accountIndex", {}),
            history=data.get("history", []),
            status=data.get("status", "idle"),
            last_error=data.get("lastError"),
            last_backend=data.get("lastBackend"),
            started_at=data.get("startedAt"),
            updated_at=data.get("updatedAt"),
            total_steps=data.get("totalSteps", 0),
        )


class RunnerStore:
    """Filesystem persistence for runner state, mission, and logs."""

    def __init__(self, backup_root: Path | None = None) -> None:
        root = backup_root if backup_root is not None else paths.get_backup_root()
        self.root = root / RUNNER_DIRNAME
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def state_path(self) -> Path:
        return self.root / STATE_FILENAME

    @property
    def mission_path(self) -> Path:
        return self.root / MISSION_FILENAME

    @property
    def log_path(self) -> Path:
        return self.root / LOG_FILENAME

    @property
    def stop_path(self) -> Path:
        return self.root / STOP_FILENAME

    @property
    def pid_path(self) -> Path:
        return self.root / PID_FILENAME

    def load_state(self) -> MissionState:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            return MissionState.from_dict(data)
        except (FileNotFoundError, json.JSONDecodeError):
            return MissionState()

    def save_state(self, state: MissionState) -> None:
        state.updated_at = datetime.now(timezone.utc).isoformat()
        atomic_write_json(self.state_path, state.to_dict())

    def write_mission(self, mission: str) -> None:
        self.mission_path.write_text(mission, encoding="utf-8")

    def read_mission(self) -> str:
        try:
            return self.mission_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def log(self, line: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{stamp}] {line}\n")

    def stop_requested(self) -> bool:
        return self.stop_path.exists()

    def request_stop(self) -> None:
        self.stop_path.touch()

    def clear_stop(self) -> None:
        self.stop_path.unlink(missing_ok=True)

    def write_pid(self) -> None:
        self.pid_path.write_text(str(os.getpid()), encoding="utf-8")

    def read_pid(self) -> int | None:
        try:
            return int(self.pid_path.read_text(encoding="utf-8").strip())
        except (FileNotFoundError, ValueError):
            return None

    def clear_pid(self) -> None:
        self.pid_path.unlink(missing_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_tool(argv: list[str], timeout: float = 600) -> subprocess.CompletedProcess[str]:
    """Run a CLI tool, working around Windows .cmd/.bat executables."""
    if sys.platform == "win32" and argv:
        resolved = shutil.which(argv[0])
        if resolved and resolved.lower().endswith((".cmd", ".bat")):
            argv = ["cmd", "/c", resolved, *argv[1:]]
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdin=subprocess.DEVNULL,
        timeout=timeout,
    )


def _load_runner_settings(backup_root: Path | None = None) -> RunnerSettings:
    """Load runner section from settings.json; missing/corrupt → defaults."""
    root = backup_root if backup_root is not None else paths.get_backup_root()
    try:
        raw = json.loads((root / "settings.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return RunnerSettings()
    section = raw.get("runner")
    if not isinstance(section, dict):
        return RunnerSettings()

    def _to_tuple(value, default):
        if isinstance(value, list):
            return tuple(value)
        return default

    return RunnerSettings(
        priority=_to_tuple(section.get("priority"), tuple(DEFAULT_PRIORITY)),
        retry_attempts=int(section.get("retryAttempts", 2)),
        cooldown_seconds=float(section.get("cooldownSeconds", 30.0)),
        auto_loop=bool(section.get("autoLoop", True)),
        max_history_turns=int(section.get("maxHistoryTurns", 20)),
        openrouter_model=str(section.get("openrouterModel", DEFAULT_OPENROUTER_MODEL)),
        free_models=_to_tuple(section.get("freeModels"), tuple(DEFAULT_FREE_MODELS)),
        local_endpoints=_to_tuple(
            section.get("localEndpoints"),
            tuple({"url": e["url"], "type": e["type"]} for e in DEFAULT_LOCAL_ENDPOINTS),
        ),
    )


def _save_runner_settings(settings: RunnerSettings, backup_root: Path | None = None) -> None:
    """Persist the runner section, preserving other settings."""
    root = backup_root if backup_root is not None else paths.get_backup_root()
    path = root / "settings.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        raw = {}
    raw["runner"] = {
        "priority": list(settings.priority),
        "retryAttempts": settings.retry_attempts,
        "cooldownSeconds": settings.cooldown_seconds,
        "autoLoop": settings.auto_loop,
        "maxHistoryTurns": settings.max_history_turns,
        "openrouterModel": settings.openrouter_model,
        "freeModels": list(settings.free_models),
        "localEndpoints": list(settings.local_endpoints),
    }
    raw["schemaVersion"] = raw.get("schemaVersion", 1)
    atomic_write_json(path, raw)


@dataclass(frozen=True)
class BackendOutput:
    """Result from a single backend execution step."""

    result: StepResult
    stdout: str
    stderr: str
    returncode: int
    backend: str
    account: str | None = None
    model: str | None = None


def _classify_output(backend: str, returncode: int, stdout: str, stderr: str) -> StepResult:
    """Detect quota/rate-limit/auth failures from tool output."""
    text = (stdout + "\n" + stderr).lower()
    patterns = {
        StepResult.QUOTA_ERROR: [
            "quota exceeded", "rate limit exceeded", "too many requests",
            "usage limit", "you've reached your", "insufficient quota",
            "quota limit", "out of credits", "credit limit",
        ],
        StepResult.RATE_LIMIT: [
            "rate limit", "429", "too many requests", "throttled",
            "slow down", "retry-after",
        ],
        StepResult.AUTH_ERROR: [
            "authentication failed", "unauthorized", "invalid token",
            "token expired", "credentials", "401", "403",
        ],
    }
    for result, phrases in patterns.items():
        if any(p in text for p in phrases):
            return result
    if returncode == 0:
        return StepResult.SUCCESS
    if "not found" in text or "command not found" in text or "was not found" in text:
        return StepResult.UNAVAILABLE
    return StepResult.OTHER_ERROR


def _build_prompt(mission: str, history: list[dict[str, str]], max_turns: int) -> str:
    """Compose the next-turn prompt from mission + recent history."""
    parts = [mission]
    recent = history[-max_turns:] if history else []
    if recent:
        parts.append("\n\n--- previous turns ---")
        for turn in recent:
            role = turn.get("role", "turn")
            content = turn.get("content", "")
            parts.append(f"\n[{role}]\n{content[:4000]}")
        parts.append("\n\nContinue from where you left off. Produce the next concrete step.")
    return "\n".join(parts)


class Backend:
    """Abstract backend that can execute one step of a mission."""

    name: ClassVar[str]

    def __init__(self, store: RunnerStore, settings: RunnerSettings) -> None:
        self.store = store
        self.settings = settings

    def prepare(self) -> tuple[bool, str]:
        """Return (ready, reason)."""
        return True, ""

    def rotate_account(self) -> str | None:
        """Subscription backends override this to cycle accounts."""
        return None

    def step(self, prompt: str) -> BackendOutput:
        raise NotImplementedError


class ClaudeBackend(Backend):
    name = "claude"

    def __init__(self, store: RunnerStore, settings: RunnerSettings) -> None:
        super().__init__(store, settings)
        self.switcher = ClaudeAccountSwitcher()
        self.binary: str | None = None
        self.account_label: str | None = None

    def _account_label(self) -> str:
        """Fetch account label without printing to stdout."""
        try:
            payload = self.switcher.status(json_output=True)
            active = payload.get("active") or {}
            email = active.get("email", "")
            org = active.get("organizationName", "")
            if org:
                return f"{email} [{org}]"
            return email or "unknown"
        except ClaudeSwitchError:
            return "unknown"

    def prepare(self) -> tuple[bool, str]:
        binary = shutil.which("claude")
        if not binary:
            return False, "claude not on PATH"
        self.binary = binary
        if not paths.get_credentials_path().exists():
            return False, "no Claude login found"
        self.account_label = self._account_label()
        return True, f"active: {self.account_label}"

    def rotate_account(self) -> str | None:
        """Switch to the next/best Claude account; return new label or None."""
        try:
            self.switcher.switch(strategy="best")
            self.account_label = self._account_label()
            return self.account_label
        except ClaudeSwitchError:
            return None

    def step(self, prompt: str) -> BackendOutput:
        argv = [self.binary or "claude", "-p", prompt]
        proc = _run_tool(argv)
        result = _classify_output(self.name, proc.returncode, proc.stdout, proc.stderr)
        return BackendOutput(
            result=result,
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
            backend=self.name,
            account=self.account_label,
        )


class CodexBackend(Backend):
    name = "codex"

    def __init__(self, store: RunnerStore, settings: RunnerSettings) -> None:
        super().__init__(store, settings)
        self.store_tool = ToolAccountStore("codex")
        self.binary: str | None = None
        self.account_label: str | None = None

    def _account_label(self) -> str:
        try:
            payload = self.store_tool.status(json_output=True)
            slot = payload.get("activeAccountNumber")
            label = payload.get("label", "")
            if slot is not None:
                return f"{slot}: {label}"
            return label or "unmanaged"
        except ClaudeSwitchError:
            return "unmanaged"

    def prepare(self) -> tuple[bool, str]:
        binary = shutil.which("codex")
        if not binary:
            return False, "codex not on PATH"
        self.binary = binary
        if not tool_live_credential_path(self.store_tool.spec).exists():
            return False, "no Codex login found"
        self.account_label = self._account_label()
        return True, self.account_label

    def rotate_account(self) -> str | None:
        """Switch to the next stored Codex account; return new label or None."""
        try:
            slot, label = self.store_tool.switch()
            self.account_label = f"{slot}: {label}"
            return self.account_label
        except ClaudeSwitchError:
            return None

    def step(self, prompt: str) -> BackendOutput:
        argv = [self.binary or "codex", prompt]
        proc = _run_tool(argv)
        result = _classify_output(self.name, proc.returncode, proc.stdout, proc.stderr)
        return BackendOutput(
            result=result,
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
            backend=self.name,
            account=self.account_label,
        )


class KimiBackend(Backend):
    name = "kimi"

    def __init__(self, store: RunnerStore, settings: RunnerSettings) -> None:
        super().__init__(store, settings)
        self.store_tool = ToolAccountStore("kimi")
        self.binary: str | None = None
        self.account_label: str | None = None

    def _account_label(self) -> str:
        try:
            payload = self.store_tool.status(json_output=True)
            slot = payload.get("activeAccountNumber")
            label = payload.get("label", "")
            if slot is not None:
                return f"{slot}: {label}"
            return label or "unmanaged"
        except ClaudeSwitchError:
            return "unmanaged"

    def prepare(self) -> tuple[bool, str]:
        binary = shutil.which("kimi")
        if not binary:
            return False, "kimi not on PATH"
        self.binary = binary
        if not tool_live_credential_path(self.store_tool.spec).exists():
            return False, "no Kimi login found"
        self.account_label = self._account_label()
        return True, self.account_label

    def rotate_account(self) -> str | None:
        """Switch to the next stored Kimi account; return new label or None."""
        try:
            slot, label = self.store_tool.switch()
            self.account_label = f"{slot}: {label}"
            return self.account_label
        except ClaudeSwitchError:
            return None

    def step(self, prompt: str) -> BackendOutput:
        argv = [self.binary or "kimi", "--prompt", prompt]
        proc = _run_tool(argv)
        result = _classify_output(self.name, proc.returncode, proc.stdout, proc.stderr)
        return BackendOutput(
            result=result,
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
            backend=self.name,
            account=self.account_label,
        )


class OpenRouterBackend(Backend):
    name = "openrouter-qwen"

    def __init__(self, store: RunnerStore, settings: RunnerSettings) -> None:
        super().__init__(store, settings)
        self.model = settings.openrouter_model

    def prepare(self) -> tuple[bool, str]:
        if not os.environ.get("OPENROUTER_API_KEY"):
            return False, "OPENROUTER_API_KEY not set"
        return True, f"model: {self.model}"

    def step(self, prompt: str) -> BackendOutput:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096,
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            return BackendOutput(
                result=StepResult.SUCCESS,
                stdout=content,
                stderr="",
                returncode=0,
                backend=self.name,
                model=self.model,
            )
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            result = _classify_output(self.name, e.code, "", detail)
            return BackendOutput(
                result=result,
                stdout="",
                stderr=detail,
                returncode=e.code,
                backend=self.name,
                model=self.model,
            )
        except Exception as e:
            return BackendOutput(
                result=StepResult.OTHER_ERROR,
                stdout="",
                stderr=str(e),
                returncode=1,
                backend=self.name,
                model=self.model,
            )


class OpencodeFreeBackend(Backend):
    name = "opencode-free"

    def __init__(self, store: RunnerStore, settings: RunnerSettings) -> None:
        super().__init__(store, settings)
        self.models = list(settings.free_models)
        self.binary: str | None = None
        self.model_index = 0

    def prepare(self) -> tuple[bool, str]:
        binary = shutil.which("opencode")
        if not binary:
            return False, "opencode not on PATH"
        self.binary = binary
        return True, f"free models: {len(self.models)}"

    def step(self, prompt: str) -> BackendOutput:
        model = self.models[self.model_index % len(self.models)]
        argv = [self.binary or "opencode", "run", "-m", model, prompt]
        proc = _run_tool(argv, timeout=300)
        result = _classify_output(self.name, proc.returncode, proc.stdout, proc.stderr)
        if result in (StepResult.QUOTA_ERROR, StepResult.RATE_LIMIT):
            self.model_index += 1
        return BackendOutput(
            result=result,
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
            backend=self.name,
            model=model,
        )


class LocalBackend(Backend):
    name = "local"

    def __init__(self, store: RunnerStore, settings: RunnerSettings) -> None:
        super().__init__(store, settings)
        self.endpoints = list(settings.local_endpoints)
        self.endpoint_index = 0
        self.current_model: str | None = None

    def _probe(self, url: str) -> tuple[list[str], str]:
        """Return (available model ids, error string)."""
        try:
            req = urllib.request.Request(
                f"{url}/models",
                headers={"Authorization": "Bearer no-key"} if "ollama" not in url else {},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            models = data.get("data", data.get("models", []))
            ids = [m.get("id", m.get("model", "")) for m in models]
            return [i for i in ids if i], ""
        except Exception as e:
            return [], str(e)

    def prepare(self) -> tuple[bool, str]:
        for offset, endpoint in enumerate(self.endpoints):
            url = endpoint["url"]
            ids, err = self._probe(url)
            if ids:
                self.endpoint_index = offset
                self.current_model = ids[0]
                return True, f"{endpoint['type']} @ {url} -> {ids[0]}"
        return False, "no local endpoint reachable"

    def step(self, prompt: str) -> BackendOutput:
        endpoint = self.endpoints[self.endpoint_index % len(self.endpoints)]
        url = endpoint["url"]
        model = self.current_model or "unknown"
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
            "stream": False,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{url}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            return BackendOutput(
                result=StepResult.SUCCESS,
                stdout=content,
                stderr="",
                returncode=0,
                backend=self.name,
                model=f"{endpoint['type']}/{model}",
            )
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            result = _classify_output(self.name, e.code, "", detail)
            return BackendOutput(
                result=result,
                stdout="",
                stderr=detail,
                returncode=e.code,
                backend=self.name,
                model=f"{endpoint['type']}/{model}",
            )
        except Exception as e:
            return BackendOutput(
                result=StepResult.OTHER_ERROR,
                stdout="",
                stderr=str(e),
                returncode=1,
                backend=self.name,
                model=f"{endpoint['type']}/{model}",
            )


BACKEND_REGISTRY: dict[str, type[Backend]] = {
    "claude": ClaudeBackend,
    "codex": CodexBackend,
    "kimi": KimiBackend,
    "openrouter-qwen": OpenRouterBackend,
    "opencode-free": OpencodeFreeBackend,
    "local": LocalBackend,
}


def _make_backend(name: str, store: RunnerStore, settings: RunnerSettings) -> Backend:
    cls = BACKEND_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"unknown backend '{name}'")
    return cls(store, settings)


class PersistentRunner:
    """Headless daemon that runs a mission across a priority list of backends."""

    def __init__(
        self,
        store: RunnerStore | None = None,
        settings: RunnerSettings | None = None,
        on_event: Callable[[str], None] | None = None,
    ) -> None:
        self.store = store or RunnerStore()
        self.settings = settings or _load_runner_settings()
        self.on_event = on_event or (lambda _: None)
        self._stop_event = threading.Event()

    def _emit(self, line: str) -> None:
        self.store.log(line)
        self.on_event(line)

    def _current_backend_name(self, state: MissionState) -> str | None:
        priority = list(self.settings.priority)
        if not priority or state.backend_index >= len(priority):
            return None
        return priority[state.backend_index]

    def _next_backend(self, state: MissionState) -> bool:
        """Advance to the next backend tier. Return True if one remains."""
        state.backend_index += 1
        state.account_index = {}
        return state.backend_index < len(self.settings.priority)

    def _cycle_account(self, backend: Backend, backend_name: str, state: MissionState) -> bool:
        """Try the next stored account for a subscription backend.

        Returns True if there might be another account to try.
        """
        if backend_name not in ("claude", "codex", "kimi"):
            return False
        idx = state.account_index.get(backend_name, 0) + 1
        state.account_index[backend_name] = idx
        # Ask the backend to rotate; if it returns a label we have a fresh account.
        new_label = backend.rotate_account()
        if new_label:
            self._emit(f"rotated {backend_name} -> {new_label}")
            return True
        # Heuristic: allow up to 10 accounts before giving up on this backend.
        return idx < 10

    def run_once(self) -> MissionState:
        """Single daemon tick: run one step and update state."""
        state = self.store.load_state()
        if not state.mission:
            # Allow starting from a mission file written by start_daemon().
            state.mission = self.store.read_mission()
        if not state.mission:
            state.status = "failed"
            state.last_error = "no mission loaded"
            self.store.save_state(state)
            return state

        backend_name = self._current_backend_name(state)
        if backend_name is None:
            state.status = "failed"
            state.last_error = "all backends exhausted"
            self.store.save_state(state)
            return state

        state.status = "running"
        state.last_backend = backend_name
        self.store.save_state(state)

        backend = _make_backend(backend_name, self.store, self.settings)
        ready, reason = backend.prepare()
        if not ready:
            self._emit(f"{backend_name} unavailable: {reason}")
            if not self._next_backend(state):
                state.status = "failed"
                state.last_error = f"{backend_name} unavailable and no backends left"
                self.store.save_state(state)
            else:
                state.last_error = f"{backend_name} unavailable; escalated"
                self.store.save_state(state)
            return state

        self._emit(f"using {backend_name} ({reason})")
        prompt = _build_prompt(state.mission, state.history, self.settings.max_history_turns)

        try:
            output = backend.step(prompt)
        except Exception as e:
            output = BackendOutput(
                result=StepResult.OTHER_ERROR,
                stdout="",
                stderr=str(e),
                returncode=1,
                backend=backend_name,
            )

        state.total_steps += 1

        if output.result == StepResult.SUCCESS:
            state.history.append({"role": output.backend, "content": output.stdout})
            state.last_error = None
            if self.settings.auto_loop:
                state.status = "running"
                self._emit(f"{backend_name} step succeeded ({len(output.stdout)} chars); looping")
            else:
                state.status = "completed"
                self._emit(f"{backend_name} step succeeded; mission completed")
            self.store.save_state(state)
            return state

        # Quota/rate-limit/auth: try another account for subscription backends,
        # otherwise fall through to next backend tier.
        self._emit(
            f"{backend_name} {output.result.value} "
            f"(rc={output.returncode}): {output.stderr[:200]}"
        )

        if backend_name in ("claude", "codex", "kimi"):
            if self._cycle_account(backend, backend_name, state):
                self._emit(f"trying next {backend_name} account")
                self.store.save_state(state)
                return state

        if not self._next_backend(state):
            state.status = "failed"
            state.last_error = (
                f"{backend_name} {output.result.value}; all backends exhausted"
            )
            self.store.save_state(state)
        else:
            next_name = self._current_backend_name(state)
            state.last_backend = next_name
            state.last_error = f"{backend_name} failed; escalating to {next_name}"
            self.store.save_state(state)
        return state

    def run_loop(self) -> int:
        """Run until stopped or all backends exhausted."""
        self.store.clear_stop()
        self.store.write_pid()
        try:
            state = self.store.load_state()
            if state.started_at is None:
                state.started_at = _now()
                self.store.save_state(state)
            self._emit("daemon started")

            while not self._stop_event.is_set() and not self.store.stop_requested():
                state = self.run_once()
                if state.status in ("completed", "failed"):
                    break
                self._stop_event.wait(self.settings.cooldown_seconds)

            final = self.store.load_state()
            if self.store.stop_requested():
                final.status = "paused"
                self.store.save_state(final)
                self._emit("daemon stopped by operator")
                return 0
            self._emit(f"daemon finished with status={final.status}")
            return 0 if final.status == "completed" else 1
        finally:
            self.store.clear_pid()

    def stop(self) -> None:
        self._stop_event.set()
        self.store.request_stop()


def start_daemon(mission: str, backup_root: Path | None = None) -> int:
    """Fork a daemon process and return immediately."""
    store = RunnerStore(backup_root)
    store.clear_stop()
    store.write_mission(mission)
    state = MissionState(
        mission=mission,
        status="idle",
        started_at=_now(),
    )
    store.save_state(state)

    # Spawn detached child. On Windows use creationflags; on POSIX double-fork.
    if sys.platform == "win32":
        creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        proc = subprocess.Popen(
            [sys.executable, "-m", "claude_swap.runner", "--daemon"],
            creationflags=creationflags,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
        store.write_pid()  # child will overwrite with its own pid
        print(f"runner daemon started (pid {proc.pid})")
        return proc.pid
    else:
        # First fork
        pid1 = os.fork()
        if pid1 > 0:
            print(f"runner daemon started (pid {pid1})")
            return pid1
        os.chdir("/")
        os.setsid()
        # Second fork
        pid2 = os.fork()
        if pid2 > 0:
            sys.exit(0)
        # Run daemon in this grandchild
        runner = PersistentRunner(store=store)
        sys.exit(runner.run_loop())


def stop_daemon(backup_root: Path | None = None) -> bool:
    """Signal the daemon to stop."""
    store = RunnerStore(backup_root)
    store.request_stop()
    pid = store.read_pid()
    if pid is not None:
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/PID", str(pid)],
                    capture_output=True,
                    check=False,
                )
            else:
                os.kill(pid, 15)
        except ProcessLookupError:
            pass
        except OSError:
            pass
    return True


def _process_exists(pid: int) -> bool:
    """Cross-platform check whether a process is still alive."""
    if sys.platform == "win32":
        try:
            proc = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True,
                check=False,
            )
            return str(pid) in proc.stdout
        except OSError:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def runner_status(backup_root: Path | None = None) -> dict:
    """Human-readable and machine-readable status."""
    store = RunnerStore(backup_root)
    state = store.load_state()
    pid = store.read_pid()
    running = False
    if pid is not None:
        running = _process_exists(pid)
    return {
        "running": running,
        "pid": pid,
        "status": state.status,
        "backend": state.last_backend,
        "backendIndex": state.backend_index,
        "totalSteps": state.total_steps,
        "lastError": state.last_error,
        "startedAt": state.started_at,
        "updatedAt": state.updated_at,
        "mission": state.mission[:200] + "..." if len(state.mission) > 200 else state.mission,
    }


def tail_logs(lines: int = 50, backup_root: Path | None = None) -> str:
    store = RunnerStore(backup_root)
    try:
        text = store.log_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    all_lines = text.splitlines()
    return "\n".join(all_lines[-lines:])


def set_priority(priority: list[str], backup_root: Path | None = None) -> None:
    unknown = [p for p in priority if p not in BACKEND_REGISTRY]
    if unknown:
        raise ValueError(f"unknown backends: {', '.join(unknown)}")
    settings = _load_runner_settings(backup_root)
    settings = dataclasses.replace(settings, priority=tuple(priority))
    _save_runner_settings(settings, backup_root)


if __name__ == "__main__":
    # Invoked as ``python -m claude_swap.runner --daemon`` by start_daemon().
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--daemon", action="store_true")
    args = parser.parse_args()
    if args.daemon:
        runner = PersistentRunner()
        sys.exit(runner.run_loop())
    parser.print_help()
    sys.exit(2)
