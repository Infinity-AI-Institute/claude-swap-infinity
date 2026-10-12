"""When no Vision Claude login can serve, cswap says so at once and falls back.

Incident, 2026-09-25: a host whose only credential was a Vision key ran
``claude-swap run -- -p "reply with the word ok" --model opus`` while the one
Claude account Vision granted had spent its 5-hour window. Native Claude
started anyway. The adapter answered every request with a generic 503, and
native retried it ten times over about three minutes before exiting 1 with
"Central account configuration or recovery needs attention."
"""

import json
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from claude_swap import cli
from claude_swap.exceptions import EXIT_NO_USABLE_LOGIN, NoUsableLogin
from claude_swap.session import SessionManager, requested_model
from claude_swap.switcher import ClaudeAccountSwitcher
from claude_swap.usage_store import FetchRecord, UsageEntry
from claude_swap.vision import VisionClient, VisionError
from claude_swap.vision_pool import CentralPoolCredential
from claude_swap.vision_proxy import InferenceProxy
from claude_swap.vision_registry import RegistryPool
from tests.test_vision_registry import item

RESET_IN_SECONDS = 2 * 3600
PROMPT_ARGS = ["-p", "reply with the word ok", "--model", "opus"]


def iso(timestamp):
    return datetime.fromtimestamp(timestamp, UTC).isoformat()


def spent_five_hour_window(now):
    return {
        "five_hour": {"pct": 100.0, "resets_at": iso(now + RESET_IN_SECONDS)},
        "seven_day": {"pct": 40.0, "resets_at": iso(now + 3 * 86400)},
    }


def healthy(now):
    return {
        "five_hour": {"pct": 10.0, "resets_at": iso(now + 3600)},
        "seven_day": {"pct": 10.0, "resets_at": iso(now + 3 * 86400)},
    }


class VisionHost:
    """A key-only host: one granted Vision account and no local Claude login."""

    def __init__(self, monkeypatch):
        self.now = time.time()
        self.switcher = ClaudeAccountSwitcher()
        self.switcher.backup_dir.mkdir(parents=True, exist_ok=True)
        self.manager = SessionManager(self.switcher)
        self.client = VisionClient("https://vision.example.invalid", "synthetic-key")
        self.client.discover = Mock(return_value=[item(1)])
        self.client.credential = Mock(side_effect=self._credential)
        self.client.refresh = Mock(side_effect=AssertionError("unexpected refresh"))
        # Vision's usage readings, by registry item index.
        self.usage = {1: spent_five_hour_window(self.now)}
        monkeypatch.setattr("claude_swap.vision_pool.read_usage", self._read_usage)
        monkeypatch.setattr("claude_swap.vision.configured_client", lambda: self.client)
        monkeypatch.setattr(
            "claude_swap.switcher.configured_vision_client", lambda: self.client
        )
        monkeypatch.setattr(
            "claude_swap.session.shutil.which", lambda _: "/synthetic/claude"
        )
        self.run_native = Mock(side_effect=SystemExit("native Claude started"))
        monkeypatch.setattr("claude_swap.vision_proxy.run_native", self.run_native)
        # Patched on the class, so the SessionManager that `cli._run_command`
        # builds for itself is covered too. Mock attributes do not bind, so
        # calls record the same arguments as the real methods receive.
        monkeypatch.setattr(
            SessionManager, "_exec", Mock(side_effect=SystemExit("claude exec"))
        )
        self.setup_session = Mock(side_effect=self._session_for)
        monkeypatch.setattr(SessionManager, "setup_session", self.setup_session)

    def _session_for(self, identifier, _share, _share_history=False):
        """A session-mode profile for a local account, without bootstrapping it."""
        row = self.switcher._get_sequence_data()["accounts"][identifier]
        directory = self.switcher.backup_dir / "sessions" / identifier
        return directory, identifier, row["email"]

    def _read_usage(self, *_args, **_kwargs):
        return {
            item(index)["login_id"]: (
                item(index)["account_id"],
                UsageEntry(last_good=usage, age_s=10),
            )
            for index, usage in self.usage.items()
        }

    @staticmethod
    def _credential(account, login):
        row = item(int(login[-12:]))
        return {
            "account_id": account,
            "login_id": login,
            "email": row["email"],
            "organization_id": row["organization_id"],
            "generation": 1,
            "expires_at": None,
            "kind": "login_oauth",
            "accessToken": "synthetic-vision-token",
        }

    def add_local_account(self, number, email):
        RegistryPool(self.switcher, self.client).sync()
        roster = self.switcher._get_sequence_data()
        roster["accounts"][str(number)] = {
            "email": email,
            "organizationUuid": "",
            "added": "2026-09-01T00:00:00Z",
        }
        roster["sequence"].append(number)
        self.switcher._write_json(self.switcher.sequence_file, roster)

    def record_local_usage(self, number, email, usage):
        """What `cswap list` or `cswap auto` last measured for a local account."""
        self.switcher._usage_store.record(
            {str(number): FetchRecord(usage=usage)}, {str(number): (email, "")}
        )


@pytest.fixture
def host(temp_home, monkeypatch):
    for name in ("VISION_API_KEY", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    return VisionHost(monkeypatch)


@pytest.fixture
def native_login(temp_home):
    """An ordinary subscription login made with native Claude on this machine."""
    (temp_home / ".claude" / ".credentials.json").write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": "synthetic-native-access",
                    "refreshToken": "synthetic-native-refresh",
                    "expiresAt": 1,
                    "scopes": ["user:inference", "user:profile"],
                }
            }
        )
    )
    (temp_home / ".claude.json").write_text(
        json.dumps({"oauthAccount": {"emailAddress": "native@example.invalid"}})
    )


# -- the adapter, mid-session ------------------------------------------------


def test_spent_pool_answers_native_once_with_a_non_retryable_reason(host):
    pool = CentralPoolCredential(
        host.switcher, host.client, {"visionLoginId": item(1)["login_id"]}
    )
    with InferenceProxy(pool) as proxy:
        request = urllib.request.Request(
            proxy.url + "/v1/messages",
            data=json.dumps({"model": "claude-opus-4-6"}).encode(),
            headers={
                "Authorization": "Bearer " + proxy.capability,
                "Content-Type": "application/json",
            },
        )
        with pytest.raises(urllib.error.HTTPError) as answered:
            urllib.request.urlopen(request, timeout=5)
    response = answered.value
    message = json.loads(response.read())["error"]["message"]
    assert response.code == 503
    # Native honors this header: it fails the turn at once instead of
    # retrying a 503 ten times over about three minutes.
    assert response.headers["x-should-retry"] == "false"
    assert abs(int(response.headers["Retry-After"]) - RESET_IN_SECONDS) <= 5
    assert "user1@example.invalid" in message
    assert "5h window at 100%" in message
    assert "cswap run" in message
    host.client.credential.assert_not_called()


# -- launch ------------------------------------------------------------------


def test_spent_pool_stops_the_launch_before_native_starts(host):
    with pytest.raises(NoUsableLogin) as refused:
        host.manager.exec_default(PROMPT_ARGS)
    message = str(refused.value)
    assert "user1@example.invalid" in message
    assert "5h window at 100%" in message
    assert "UTC" in message
    assert abs(refused.value.retry_after_seconds - RESET_IN_SECONDS) <= 5
    host.run_native.assert_not_called()
    host.manager._exec.assert_not_called()


def test_cli_exits_with_the_documented_code_and_says_why(host, capsys):
    with pytest.raises(SystemExit) as exited:
        cli._run_command(["--", *PROMPT_ARGS])
    assert exited.value.code == EXIT_NO_USABLE_LOGIN == 75
    stderr = capsys.readouterr().err
    assert "user1@example.invalid" in stderr
    assert "no local claude login" in stderr.lower()
    host.run_native.assert_not_called()


def test_explicit_vision_account_gets_the_same_launch_check(host):
    RegistryPool(host.switcher, host.client).sync()
    with pytest.raises(NoUsableLogin):
        host.manager.run("1", PROMPT_ARGS)
    host.run_native.assert_not_called()


def test_pool_with_quota_launches_vision_without_reading_local_logins(host):
    host.usage[1] = healthy(host.now)
    host.switcher._read_credentials = Mock(
        side_effect=AssertionError("local login lookup")
    )
    with pytest.raises(SystemExit, match="native Claude started"):
        host.manager.exec_default(PROMPT_ARGS)
    host.run_native.assert_called_once()


def test_registry_outage_during_the_check_launches_as_before(host):
    # Membership synced a minute ago, so the launch check asks Vision again.
    RegistryPool(host.switcher, host.client, now=lambda: host.now - 60).sync()
    host.client.discover.side_effect = VisionError("service_unavailable")
    with pytest.raises(SystemExit, match="native Claude started"):
        host.manager.run("1", PROMPT_ARGS)


def test_launch_without_model_ignores_another_models_weekly_limit(host):
    usage = healthy(host.now)
    usage["scoped"] = [{"name": "Opus", "pct": 100.0}]
    host.usage[1] = usage
    with pytest.raises(SystemExit, match="native Claude started"):
        host.manager.exec_default(["-p", "reply with the word ok"])
    with pytest.raises(NoUsableLogin, match="Opus window at 100%"):
        host.manager.exec_default(PROMPT_ARGS)


@pytest.mark.parametrize(
    "arguments,model",
    [
        (["-p", "reply with the word ok"], None),
        (["--model", "opus", "-p", "reply with the word ok"], "opus"),
        (["--model=sonnet"], "sonnet"),
        (["--model", "opus", "--model", "haiku"], "haiku"),
    ],
)
def test_launch_check_uses_the_model_native_was_asked_for(arguments, model):
    assert requested_model(arguments) == model


# -- falling back to this machine's own logins -------------------------------


def test_spent_pool_falls_back_to_the_native_login_and_warns_on_stderr(
    host, native_login, monkeypatch, capsys
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-synthetic-metered-key")
    with pytest.raises(SystemExit, match="claude exec"):
        host.manager.exec_default(PROMPT_ARGS)
    (claude_bin, arguments), kwargs = host.manager._exec.call_args
    assert (claude_bin, arguments) == ("/synthetic/claude", PROMPT_ARGS)
    # Falling back must not quietly become metered API-key billing.
    assert "ANTHROPIC_API_KEY" not in kwargs["env"]
    output = capsys.readouterr()
    assert "native@example.invalid" in output.err
    assert "user1@example.invalid" in output.err
    assert "native@example.invalid" not in output.out
    host.run_native.assert_not_called()


def test_spent_pool_falls_back_to_a_local_cswap_account(host, capsys):
    host.add_local_account(5, "local@example.invalid")
    with pytest.raises(SystemExit, match="claude exec"):
        host.manager.exec_default(PROMPT_ARGS)
    host.setup_session.assert_called_once()
    assert host.setup_session.call_args.args[0] == "5"
    assert "local@example.invalid" in capsys.readouterr().err
    host.run_native.assert_not_called()


def test_disabled_or_api_key_local_accounts_are_not_fallbacks(host):
    host.add_local_account(5, "disabled@example.invalid")
    host.add_local_account(6, "key@example.invalid")
    roster = host.switcher._get_sequence_data()
    roster["accounts"]["5"]["disabled"] = True
    roster["accounts"]["6"]["kind"] = "api_key"
    host.switcher._write_json(host.switcher.sequence_file, roster)
    with pytest.raises(NoUsableLogin):
        host.manager.exec_default(PROMPT_ARGS)
    host.manager._exec.assert_not_called()


def test_metered_api_key_alone_is_not_a_fallback(host, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-synthetic-metered-key")
    with pytest.raises(NoUsableLogin, match="ANTHROPIC_API_KEY"):
        host.manager.exec_default(PROMPT_ARGS)
    host.manager._exec.assert_not_called()


def test_empty_pool_falls_back_to_the_native_login(host, native_login, capsys):
    host.client.discover.return_value = []
    with pytest.raises(SystemExit, match="claude exec"):
        host.manager.exec_default(PROMPT_ARGS)
    assert "authorized through Vision" in capsys.readouterr().err


# -- issue #16: a launch without a named account falls back instead of failing


def test_issue_16_spent_default_account_falls_back_to_a_local_account(host, capsys):
    """#16 as reported: no account argument, the default Vision account's
    5-hour window full, and a second, local account registered. The launch
    uses the local account and says so on stderr; native never starts on the
    spent account."""
    host.add_local_account(5, "luke-3@example.invalid")
    with pytest.raises(SystemExit, match="claude exec"):
        cli._run_command(["--", *PROMPT_ARGS])
    host.setup_session.assert_called_once()
    assert host.setup_session.call_args.args[0] == "5"
    stderr = capsys.readouterr().err
    assert "user1@example.invalid (5h window at 100%" in stderr
    assert "Falling back to local cswap account 5 (luke-3@example.invalid)" in stderr
    host.run_native.assert_not_called()


def test_spent_default_account_launches_the_vision_account_with_quota(host, capsys):
    host.client.discover.return_value = [item(1), item(2)]
    host.usage[2] = healthy(host.now)
    with pytest.raises(SystemExit, match="native Claude started"):
        cli._run_command(["--", *PROMPT_ARGS])
    launched_record = host.run_native.call_args.args[4]
    assert launched_record["visionLoginId"] == item(2)["login_id"]
    output = capsys.readouterr()
    # The switch and the launch banner are announced on stderr, which
    # `claude -p` callers keep apart from native's answer on stdout.
    assert output.out == ""
    assert "Account-1 (user1@example.invalid)" in output.err
    assert "5h window at 100%" in output.err
    assert "Launching Account-2 (user2@example.invalid) [Vision]" in output.err


def test_a_named_account_is_not_replaced_by_a_local_login(host, native_login, capsys):
    RegistryPool(host.switcher, host.client).sync()
    with pytest.raises(SystemExit) as exited:
        cli._run_command(["1", "--", *PROMPT_ARGS])
    assert exited.value.code == EXIT_NO_USABLE_LOGIN
    stderr = capsys.readouterr().err
    assert "user1@example.invalid (5h window at 100%" in stderr
    assert "named" in stderr
    host.manager._exec.assert_not_called()
    host.run_native.assert_not_called()


def test_a_mapped_directory_is_not_replaced_by_a_local_login(
    host, native_login, temp_home, monkeypatch, capsys
):
    from claude_swap.mappings import MappingStore

    RegistryPool(host.switcher, host.client).sync()
    project = temp_home / "project"
    project.mkdir()
    MappingStore(host.switcher.backup_dir).set(
        project, "user1@example.invalid", "organization"
    )
    monkeypatch.chdir(project)
    with pytest.raises(SystemExit) as exited:
        cli._run_command(["--", *PROMPT_ARGS])
    assert exited.value.code == EXIT_NO_USABLE_LOGIN
    assert "named" in capsys.readouterr().err
    host.manager._exec.assert_not_called()


def test_fallback_skips_a_local_account_known_to_be_spent(host, capsys):
    host.add_local_account(5, "spent@example.invalid")
    host.add_local_account(6, "fresh@example.invalid")
    host.record_local_usage(5, "spent@example.invalid", spent_five_hour_window(host.now))
    with pytest.raises(SystemExit, match="claude exec"):
        host.manager.exec_default(PROMPT_ARGS)
    assert host.setup_session.call_args.args[0] == "6"
    stderr = capsys.readouterr().err
    assert "spent@example.invalid (5h window at 100%" in stderr
    assert "Falling back to local cswap account 6 (fresh@example.invalid)" in stderr


def test_fallback_skips_a_local_account_that_needs_a_relogin(host, capsys):
    host.add_local_account(5, "dead@example.invalid")
    host.add_local_account(6, "fresh@example.invalid")
    host.switcher._usage_store.record(
        {"5": FetchRecord(error="invalid_grant")},
        {"5": ("dead@example.invalid", "")},
    )
    with pytest.raises(SystemExit, match="claude exec"):
        host.manager.exec_default(PROMPT_ARGS)
    assert host.setup_session.call_args.args[0] == "6"
    assert "dead@example.invalid (re-login needed" in capsys.readouterr().err


def test_fallback_skips_the_native_login_when_its_account_is_known_spent(
    host, native_login, capsys
):
    # The native login is also cswap's account 5; account 6 is untouched.
    host.add_local_account(5, "native@example.invalid")
    host.add_local_account(6, "fresh@example.invalid")
    host.record_local_usage(5, "native@example.invalid", spent_five_hour_window(host.now))
    with pytest.raises(SystemExit, match="claude exec"):
        host.manager.exec_default(PROMPT_ARGS)
    assert host.setup_session.call_args.args[0] == "6"
    assert "native@example.invalid (5h window at 100%" in capsys.readouterr().err


def test_no_fallback_when_every_local_login_is_known_spent(host):
    host.add_local_account(5, "spent@example.invalid")
    host.record_local_usage(5, "spent@example.invalid", spent_five_hour_window(host.now))
    with pytest.raises(NoUsableLogin) as refused:
        host.manager.exec_default(PROMPT_ARGS)
    message = str(refused.value)
    assert "user1@example.invalid (5h window at 100%" in message
    assert "spent@example.invalid (5h window at 100%" in message
    host.manager._exec.assert_not_called()
    host.setup_session.assert_not_called()


def test_vision_credential_outage_falls_back_to_a_local_account(host, capsys):
    host.usage[1] = healthy(host.now)
    host.add_local_account(5, "local@example.invalid")
    host.client.credential.side_effect = VisionError("service_unavailable", status=503)
    with pytest.raises(SystemExit, match="claude exec"):
        host.manager.exec_default(PROMPT_ARGS)
    assert host.setup_session.call_args.args[0] == "5"
    assert "service_unavailable" in capsys.readouterr().err
    host.run_native.assert_not_called()


def test_vision_credential_outage_fails_a_named_account_as_before(host):
    host.usage[1] = healthy(host.now)
    host.add_local_account(5, "local@example.invalid")
    host.client.credential.side_effect = VisionError("service_unavailable", status=503)
    with pytest.raises(VisionError, match="service_unavailable"):
        host.manager.run("1", PROMPT_ARGS)
    host.setup_session.assert_not_called()
    host.run_native.assert_not_called()


def test_vision_registry_outage_falls_back_to_a_local_account(host, capsys):
    host.add_local_account(5, "local@example.invalid")
    # Membership synced a minute ago, so the launch asks Vision again.
    RegistryPool(host.switcher, host.client, now=lambda: host.now - 60).sync(
        force=True
    )
    host.client.discover.side_effect = VisionError("service_unavailable", status=503)
    with pytest.raises(SystemExit, match="claude exec"):
        host.manager.exec_default(PROMPT_ARGS)
    assert host.setup_session.call_args.args[0] == "5"
    assert "service_unavailable" in capsys.readouterr().err


def test_vision_login_needing_reauthentication_falls_back_to_a_local_account(
    host, capsys
):
    host.usage[1] = healthy(host.now)
    host.add_local_account(5, "local@example.invalid")
    host.client.credential.side_effect = VisionError("credential_unavailable")
    host.client.refresh = Mock(return_value={"state": "reauth_required"})
    with pytest.raises(SystemExit, match="claude exec"):
        host.manager.exec_default(PROMPT_ARGS)
    assert host.setup_session.call_args.args[0] == "5"
    assert "needs reauthentication" in capsys.readouterr().err
    host.run_native.assert_not_called()


def test_vision_outage_without_a_local_login_fails_as_before(host):
    host.usage[1] = healthy(host.now)
    host.client.credential.side_effect = VisionError("service_unavailable", status=503)
    with pytest.raises(VisionError, match="service_unavailable"):
        host.manager.exec_default(PROMPT_ARGS)
    host.manager._exec.assert_not_called()
    host.run_native.assert_not_called()
