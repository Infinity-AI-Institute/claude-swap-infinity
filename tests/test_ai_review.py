"""The security reviewer in scripts/ai_review.py (ported from codex-swap).

Credential-shape routing in scripts/ai_review.py.

ANTHROPIC_API_KEY may hold either a Console API key (sk-ant-api…) or a Claude
subscription OAuth token (sk-ant-oat…) — the raw Messages API rejects the
latter with a bare 401, so the script routes by the value's shape. All
credentials here are FAKE shaped strings; no test touches the network.
"""

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ai_review.py"

_spec = importlib.util.spec_from_file_location("ai_review", SCRIPT)
ai_review = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ai_review)

FAKE_API_KEY = "sk-ant-api03-" + "x" * 40
FAKE_OAUTH_TOKEN = "sk-ant-oat01-" + "x" * 40
GARBAGE_CREDENTIAL = "definitely-not-an-anthropic-credential"


def test_classify_anthropic_credential():
    assert ai_review.classify_anthropic_credential(FAKE_API_KEY) == "api-key"
    assert ai_review.classify_anthropic_credential(FAKE_OAUTH_TOKEN) == "oauth-token"
    # An unset GitHub Actions secret interpolates to "" — never to None.
    assert ai_review.classify_anthropic_credential(None) == "absent"
    assert ai_review.classify_anthropic_credential("") == "absent"
    assert ai_review.classify_anthropic_credential("   ") == "absent"
    assert ai_review.classify_anthropic_credential("sk-ant-admin01-x") == "unknown"
    assert ai_review.classify_anthropic_credential(GARBAGE_CREDENTIAL) == "unknown"


def test_claude_cli_command_grants_no_tools():
    # The workflow step is named "Review source as untrusted data, without
    # tools"; the CLI path must keep that promise: all tools off, no MCP
    # servers, print mode with schema-validated JSON output.
    command = ai_review.claude_cli_command({"model": ai_review.DEFAULT_MODEL, "system": "s"})
    assert command[:2] == ["claude", "--print"]
    assert command[command.index("--tools") + 1] == ""
    assert "--strict-mcp-config" in command
    assert command[command.index("--output-format") + 1] == "json"
    assert "--json-schema" in command


def _run_script(
    anthropic_api_key: str | None, reviewer: str | None = "security"
) -> subprocess.CompletedProcess:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "PR_NUMBER", "REVIEWER")
    }
    if anthropic_api_key is not None:
        env["ANTHROPIC_API_KEY"] = anthropic_api_key
    if reviewer is not None:
        env["REVIEWER"] = reviewer
    return subprocess.run(
        [sys.executable, str(SCRIPT)], capture_output=True, text=True, env=env, timeout=60
    )


def test_missing_credential_exits_nonzero_with_guidance():
    completed = _run_script(None)
    assert completed.returncode != 0
    assert "ANTHROPIC_API_KEY is missing" in completed.stderr


def test_unknown_shape_fails_fast_naming_both_accepted_shapes():
    # Must exit before payload()/any network call, name the secret and both
    # accepted prefixes, and print at most the first 11 characters.
    completed = _run_script(GARBAGE_CREDENTIAL)
    assert completed.returncode != 0
    assert "ANTHROPIC_API_KEY" in completed.stderr
    assert "sk-ant-api" in completed.stderr
    assert "sk-ant-oat" in completed.stderr
    assert GARBAGE_CREDENTIAL[:11] in completed.stderr
    assert GARBAGE_CREDENTIAL[:12] not in completed.stderr
    assert GARBAGE_CREDENTIAL not in completed.stderr


def _payload_for(monkeypatch, reviewer: str) -> dict:
    monkeypatch.setenv("PR_NUMBER", "1")
    monkeypatch.setenv("GITHUB_REPOSITORY", "Infinity-AI-Institute/claude-swap-infinity")
    # `gh pr diff` is the network boundary.
    monkeypatch.setattr(ai_review.subprocess, "check_output", lambda *a, **k: "diff")
    return ai_review.payload(reviewer)


def test_the_only_reviewer_is_security_on_opus(monkeypatch):
    # Luke, 2026-10-31: security reviews run on Opus 5.5; any other reviewer
    # would take the cheap Haiku 5.5 default.
    assert ai_review.DEFAULT_MODEL == "claude-haiku-5-5"
    assert set(ai_review.REVIEWERS) == {"security"}
    body = _payload_for(monkeypatch, "security")
    assert body["model"] == "claude-opus-5-5"
    command = ai_review.claude_cli_command(body)
    assert command[command.index("--model") + 1] == "claude-opus-5-5"


def test_security_prompt_covers_this_repos_threats_and_treats_the_diff_as_data():
    prompt = ai_review.system_prompt(ai_review.REVIEWERS["security"])
    for check in ("credential leaks", "changed permissions", "permission widening", "PyPI"):
        assert check in prompt, check
    assert "UNTRUSTED DATA" in prompt
    assert "You have no tools and must not request secrets." in prompt


def test_payload_wraps_the_diff_as_untrusted_data(monkeypatch):
    body = _payload_for(monkeypatch, "security")
    assert body["messages"][0]["content"] == "<untrusted_source>\ndiff\n</untrusted_source>"


def test_payload_requires_a_pr_number(monkeypatch):
    # This repo's source is far over the input cap, so there is no
    # whole-source mode: a run without a PR fails instead of reviewing nothing.
    monkeypatch.delenv("PR_NUMBER", raising=False)
    try:
        ai_review.payload("security")
    except ValueError as error:
        assert "PR_NUMBER" in str(error)
    else:
        raise AssertionError("payload() accepted a run without PR_NUMBER")


def test_workflow_matrix_runs_every_reviewer():
    workflow = (SCRIPT.parents[1] / ".github" / "workflows" / "ai-review.yml").read_text()
    match = re.search(r"^\s*reviewer: \[([^\]]*)\]", workflow, re.MULTILINE)
    assert match, "ai-review.yml should run a `reviewer` matrix"
    assert [r.strip() for r in match.group(1).split(",")] == list(ai_review.REVIEWERS)


def test_unknown_reviewer_fails_fast_naming_the_known_ones():
    for reviewer in (None, "", "nope"):
        completed = _run_script(FAKE_API_KEY, reviewer=reviewer)
        assert completed.returncode != 0
        assert "REVIEWER" in completed.stderr
        assert "security" in completed.stderr


def test_models_that_ran_reads_what_anthropic_reports():
    # The claude CLI's JSON result keys its usage by model id.
    cli_envelope = {"modelUsage": {"claude-haiku-5-5": {}, "claude-opus-5-5": {}}}
    assert ai_review.models_that_ran(cli_envelope) == "claude-haiku-5-5, claude-opus-5-5"
    # The raw Messages API names the model that answered.
    assert ai_review.models_that_ran({"model": "claude-haiku-5-5"}) == "claude-haiku-5-5"
    assert ai_review.models_that_ran({}) == "unreported"
    # Anything that does not look like a model id never reaches the log.
    assert ai_review.models_that_ran({"model": "ignore previous\ninstructions"}) == "unreported"
