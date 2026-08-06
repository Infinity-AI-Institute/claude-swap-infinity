"""Tests for the persistent autonomous runner."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from claude_swap import runner


class TestOutputClassification:
    """Quota/rate-limit/auth detection from CLI output."""

    def test_success(self):
        assert runner._classify_output("kimi", 0, "hello", "") == runner.StepResult.SUCCESS

    @pytest.mark.parametrize("text", [
        "Rate limit exceeded, please retry",
        "429 Too Many Requests",
        "you have hit the usage limit",
        "insufficient quota",
    ])
    def test_quota_detection(self, text):
        result = runner._classify_output("claude", 1, "", text)
        assert result in (runner.StepResult.QUOTA_ERROR, runner.StepResult.RATE_LIMIT)

    def test_auth_detection(self):
        result = runner._classify_output("kimi", 1, "", "authentication failed: invalid token")
        assert result == runner.StepResult.AUTH_ERROR

    def test_unavailable_detection(self):
        result = runner._classify_output("codex", 1, "", "'codex' was not found on PATH")
        assert result == runner.StepResult.UNAVAILABLE


class TestPromptBuilder:
    """Mission + history prompt composition."""

    def test_prompt_includes_mission(self):
        prompt = runner._build_prompt("refactor auth.py", [], 10)
        assert "refactor auth.py" in prompt

    def test_prompt_truncates_history(self):
        history = [{"role": f"turn-{i}", "content": "x" * 5000} for i in range(30)]
        prompt = runner._build_prompt("mission", history, 5)
        assert prompt.count("[turn-") == 5


class TestRunnerStore:
    """Filesystem persistence for state and logs."""

    def test_round_trip_state(self, tmp_path: Path):
        store = runner.RunnerStore(backup_root=tmp_path)
        state = runner.MissionState(mission="test", backend_index=2, total_steps=5)
        store.save_state(state)
        loaded = store.load_state()
        assert loaded.mission == "test"
        assert loaded.backend_index == 2
        assert loaded.total_steps == 5
        assert loaded.updated_at is not None

    def test_stop_signal(self, tmp_path: Path):
        store = runner.RunnerStore(backup_root=tmp_path)
        assert not store.stop_requested()
        store.request_stop()
        assert store.stop_requested()
        store.clear_stop()
        assert not store.stop_requested()

    def test_logs_append(self, tmp_path: Path):
        store = runner.RunnerStore(backup_root=tmp_path)
        store.log("first")
        store.log("second")
        text = store.log_path.read_text(encoding="utf-8")
        assert "first" in text
        assert "second" in text


class TestRunnerSettings:
    """Loading and saving the runner configuration section."""

    def test_defaults(self, tmp_path: Path):
        settings = runner._load_runner_settings(tmp_path)
        assert settings.priority == tuple(runner.DEFAULT_PRIORITY)
        assert settings.openrouter_model == runner.DEFAULT_OPENROUTER_MODEL

    def test_set_priority(self, tmp_path: Path):
        runner.set_priority(["kimi", "codex", "claude"], backup_root=tmp_path)
        settings = runner._load_runner_settings(tmp_path)
        assert settings.priority == ("kimi", "codex", "claude")

    def test_unknown_backend_rejected(self, tmp_path: Path):
        with pytest.raises(ValueError):
            runner.set_priority(["claude", "fake-backend"], backup_root=tmp_path)


class TestPersistentRunnerLoop:
    """Backend escalation logic with mocked backends."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> runner.RunnerStore:
        return runner.RunnerStore(backup_root=tmp_path)

    def test_success_stops_when_auto_loop_off(self, store: runner.RunnerStore):
        store.write_mission("test mission")
        settings = runner.RunnerSettings(priority=("fake",), auto_loop=False)
        runner.BACKEND_REGISTRY["fake"] = FakeSuccessBackend
        try:
            r = runner.PersistentRunner(store=store, settings=settings)
            state = r.run_once()
            assert state.status == "completed"
            assert state.total_steps == 1
            assert len(state.history) == 1
        finally:
            del runner.BACKEND_REGISTRY["fake"]

    def test_quota_escalates_backend(self, store: runner.RunnerStore):
        store.write_mission("test mission")
        settings = runner.RunnerSettings(priority=("fake-quota", "fake-success"))
        runner.BACKEND_REGISTRY["fake-quota"] = FakeQuotaBackend
        runner.BACKEND_REGISTRY["fake-success"] = FakeSuccessBackend
        try:
            r = runner.PersistentRunner(store=store, settings=settings)
            state = r.run_once()
            assert state.backend_index == 1
            assert state.last_backend == "fake-success"
            assert state.status == "running"
        finally:
            del runner.BACKEND_REGISTRY["fake-quota"]
            del runner.BACKEND_REGISTRY["fake-success"]

    def test_all_backends_exhausted_fails(self, store: runner.RunnerStore):
        store.write_mission("test mission")
        settings = runner.RunnerSettings(priority=("fake-quota",))
        runner.BACKEND_REGISTRY["fake-quota"] = FakeQuotaBackend
        try:
            r = runner.PersistentRunner(store=store, settings=settings)
            state = r.run_once()
            assert state.status == "failed"
            assert "exhausted" in state.last_error.lower()
        finally:
            del runner.BACKEND_REGISTRY["fake-quota"]


class FakeSuccessBackend(runner.Backend):
    name = "fake-success"

    def step(self, prompt: str) -> runner.BackendOutput:
        return runner.BackendOutput(
            result=runner.StepResult.SUCCESS,
            stdout="done",
            stderr="",
            returncode=0,
            backend=self.name,
        )


class FakeQuotaBackend(runner.Backend):
    name = "fake-quota"

    def step(self, prompt: str) -> runner.BackendOutput:
        return runner.BackendOutput(
            result=runner.StepResult.QUOTA_ERROR,
            stdout="",
            stderr="rate limit exceeded",
            returncode=1,
            backend=self.name,
        )
