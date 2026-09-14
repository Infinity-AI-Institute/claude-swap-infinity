"""Managed provider login and recoverable registration commands."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid

from claude_swap.exceptions import ClaudeSwitchError, SessionError
from claude_swap.locking import FileLock
from claude_swap.models import normalize_alias
from claude_swap.session import AUTH_OVERRIDE_ENV_VARS, _mkdir_private
from claude_swap.vision import VisionError, configured_client
from claude_swap.vision_handoff import (
    ManagedLoginHandoff,
    _credential,
    _read_private,
    _write_private,
)
from claude_swap.vision_registration import RegistrationClient
from claude_swap.vision_registry import RegistryPool
from claude_swap.vision_session import ROUTE_OVERRIDES
from claude_swap.vision_signin import VisionSignIn


class ManagedProfiles:
    def __init__(self, backup_dir):
        self.root = backup_dir
        self.path = backup_dir / "vision-profiles.json"

    def read(self):
        raw = _read_private(self.path)
        if raw is None:
            return {"version": 1, "auto_register": True, "profiles": {}, "bindings": {}}
        try:
            value = json.loads(raw)
            if (
                not isinstance(value, dict)
                or set(value)
                not in (
                    {"version", "auto_register", "profiles"},
                    {"version", "auto_register", "profiles", "bindings"},
                )
                or type(value["version"]) is not int
                or value["version"] != 1
                or type(value["auto_register"]) is not bool
                or not isinstance(value["profiles"], dict)
            ):
                raise ValueError()
            value.setdefault("bindings", {})
            if not isinstance(value["bindings"], dict):
                raise TypeError()
            for binding in value["bindings"].values():
                if not isinstance(binding, dict) or set(binding) != {
                    "url",
                    "account_id",
                    "login_id",
                }:
                    raise ValueError()
            for name, profile in value["profiles"].items():
                if normalize_alias(name) != name or str(uuid.UUID(profile)) != profile:
                    raise ValueError()
            return value
        except (ValueError, TypeError, AttributeError):
            raise SessionError(
                "Vision profile preferences need repair; refusing to reset your upload preference."
            ) from None

    def profile(self, name, *, create=False):
        try:
            name = normalize_alias(name)
        except ValueError:
            raise SessionError("Choose a valid local profile alias.") from None
        _mkdir_private(self.root)
        with FileLock(self.root / ".vision-profiles.lock"):
            state = self.read()
            profile = state["profiles"].get(name)
            if profile is None:
                if not create:
                    raise SessionError("No managed Vision login has that name.")
                profile = str(uuid.uuid4())
                state["profiles"][name] = profile
                _write_private(self.path, json.dumps(state))
        return profile

    def set_auto_register(self, enabled):
        _mkdir_private(self.root)
        with FileLock(self.root / ".vision-profiles.lock"):
            state = self.read()
            state["auto_register"] = enabled
            _write_private(self.path, json.dumps(state))
            if self.read()["auto_register"] is not enabled:
                raise SessionError("The upload preference was not saved.")


def name_committed_login(switcher, client, name, receipt):
    """Make the user's profile alias resolve to the newly verified remote login."""
    if receipt["state"] != "committed":
        return
    try:
        RegistryPool(switcher, client).sync(force=True)
    except VisionError:
        # Ownership is already committed. A later normal discovery can recover
        # roster availability; do not report the successful handoff as failed.
        return
    profiles = ManagedProfiles(switcher.backup_dir)
    alias = normalize_alias(name)
    with FileLock(profiles.root / ".vision-profiles.lock"):
        preferences = profiles.read()
        previous = preferences["bindings"].get(alias)
    with FileLock(switcher.lock_file):
        data = switcher._get_sequence_data() or {}
        accounts = data.get("accounts", {})
        target = next(
            (
                record
                for record in accounts.values()
                if record.get("source") == "vision"
                and record.get("visionUrl") == client.url
                and record.get("visionLoginId") == receipt["login_id"]
                and record.get("visionAccountId") == receipt["account_id"]
            ),
            None,
        )
        if target is None:
            return
        if previous is not None:
            for record in accounts.values():
                if (
                    record is not target
                    and record.get("source") == "vision"
                    and record.get("alias") == alias
                    and record.get("visionUrl") == previous["url"]
                    and record.get("visionAccountId") == previous["account_id"]
                    and record.get("visionLoginId") == previous["login_id"]
                ):
                    base = "vision-" + record["visionLoginId"][4:].replace("-", "")
                    available = base
                    suffix = 1
                    used = {
                        other.get("alias")
                        for other in accounts.values()
                        if other is not record
                    }
                    while available in used:
                        suffix += 1
                        available = f"{base}-{suffix}"
                    record["alias"] = available
        if any(
            record is not target and record.get("alias") == alias
            for record in accounts.values()
        ):
            return
        target["alias"] = alias
        switcher._write_json(switcher.sequence_file, data)
    with FileLock(profiles.root / ".vision-profiles.lock"):
        preferences = profiles.read()
        preferences["bindings"][alias] = {
            "url": client.url,
            "account_id": receipt["account_id"],
            "login_id": receipt["login_id"],
        }
        _write_private(profiles.path, json.dumps(preferences))


def _profile_environment(directory):
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in set(AUTH_OVERRIDE_ENV_VARS) | ROUTE_OVERRIDES
        and key not in {"VISION_API_KEY", "VISION_API_URL"}
    }
    env["CLAUDE_CONFIG_DIR"] = str(directory)
    env["CLAUDE_SECURESTORAGE_CONFIG_DIR"] = str(directory)
    return env


def run_profile(switcher, name, native_args):
    """Keep the ownership lease until the native local process has exited."""
    profiles = ManagedProfiles(switcher.backup_dir)
    profile_id = profiles.profile(name)
    native = shutil.which("claude")
    if native is None:
        raise SessionError("Install Claude Code before running a provider profile.")
    with ManagedLoginHandoff(switcher.backup_dir, profile_id, None) as lease:
        lease.prepare_login()
        before = lease._stores()
        material = before["keychain"] or before["file"]
        if material is None:
            raise SessionError(
                "This profile has no local login. Use its central account alias "
                "with cswap run, or use account-login for a new login."
            )
        _credential(material)
        # The managed lease prevents an upload while native holds refresh
        # material in memory. Native refresh locks are released before launch.
        result = subprocess.run(
            [native, *native_args], env=_profile_environment(lease.profile), check=False
        )
        if (
            result.returncode == 0
            and profiles.read()["auto_register"]
            and lease._stores() != before
        ):
            client = configured_client()
            if client is not None:
                lease.registry = RegistrationClient(client)
                receipt = lease.upload()
                name_committed_login(switcher, client, name, receipt)
    raise SystemExit(result.returncode)


def login_profile(switcher, name):
    profiles = ManagedProfiles(switcher.backup_dir)
    profile_id = profiles.profile(name, create=True)
    client = configured_client()
    registry = RegistrationClient(client) if client is not None else None
    native = shutil.which("claude")
    if native is None:
        raise SessionError("Install Claude Code before starting a provider login.")
    with ManagedLoginHandoff(switcher.backup_dir, profile_id, registry) as lease:
        lease.prepare_login()
        before = lease._stores()
        env = _profile_environment(lease.profile)
        result = subprocess.run([native, "auth", "login"], env=env, check=False)
        if result.returncode != 0:
            raise SystemExit(result.returncode)
        after = lease._stores()
        if after == before:
            return {"state": "unchanged", "profile": name}
        # Re-read the persisted preference after login, so another terminal's
        # explicit opt-out applies before any provider credential is uploaded.
        if not profiles.read()["auto_register"] or registry is None:
            return {"state": "local", "profile": name}
        receipt = lease.upload()
        name_committed_login(switcher, client, name, receipt)
    return receipt


def run_command(argv, switcher):
    parser = argparse.ArgumentParser(prog="cswap vision")
    parser.add_argument(
        "--url", default=os.environ.get("VISION_API_URL", "https://vision.infinity.inc")
    )
    commands = parser.add_subparsers(dest="command", required=True)
    browser = commands.add_parser("login")
    browser.add_argument("--host-label", default=socket.gethostname())
    browser.add_argument("--no-wait", action="store_true")
    commands.add_parser("status")
    commands.add_parser("cancel")
    for command in ("account-login", "upload", "cancel-upload"):
        commands.add_parser(command).add_argument("name")
    local_run = commands.add_parser("account-run")
    local_run.add_argument("name")
    local_run.add_argument("native_args", nargs=argparse.REMAINDER)
    batch = commands.add_parser("batch-upload")
    batch.add_argument("names", nargs="+")
    batch.add_argument("--confirm")
    commands.add_parser("existing-logins").add_argument(
        "--profile", action="append", default=[]
    )
    commands.add_parser("profiles")
    commands.add_parser("auto-register").add_argument("value", choices=("on", "off"))
    args = parser.parse_args(argv)
    profiles = ManagedProfiles(switcher.backup_dir)
    if args.command in {"login", "status", "cancel"}:
        if args.command == "login" and os.environ.get("VISION_API_KEY"):
            client = configured_client()
            return {"state": "configured", "source": "environment", "url": client.url}
        flow = VisionSignIn(switcher.backup_dir, args.url)
        if args.command == "cancel":
            return flow.cancel()
        if args.command == "status":
            if flow.state.read("pending") is not None:
                return flow.poll()
            client = configured_client(switcher.backup_dir)
            if client is None:
                return {"state": "signed_out"}
            return {"state": "signed_in", "url": client.url}
        public = flow.begin(args.host_label)
        if args.no_wait:
            return public
        print(json.dumps(public), flush=True)
        while True:
            result = flow.poll()
            if result["state"] != "pending":
                return result
            time.sleep(min(30, result["retry_after_seconds"]))
    if args.command == "existing-logins":
        from claude_swap.vision_inventory import capture_inventory

        return capture_inventory(switcher, args.profile).public()
    if args.command == "profiles":
        return profiles.read()
    if args.command == "auto-register":
        profiles.set_auto_register(args.value == "on")
        return {"auto_register": profiles.read()["auto_register"]}
    if args.command == "account-run":
        native_args = args.native_args
        if native_args[:1] == ["--"]:
            native_args = native_args[1:]
        return run_profile(switcher, args.name, native_args)
    if args.command == "account-login":
        return login_profile(switcher, args.name)
    if args.command == "batch-upload":
        from claude_swap.vision_batch import BatchUpload

        client = configured_client()
        if client is None:
            raise SessionError("Sign in to Vision before preparing an upload batch.")
        batch = BatchUpload(switcher, client)
        if args.confirm is None:
            return batch.preview(args.names)
        return batch.apply(args.names, args.confirm)
    profile_id = profiles.profile(args.name)
    client = configured_client()
    if client is None:
        raise SessionError("Sign in to Vision before uploading or cancelling a login.")
    with ManagedLoginHandoff(
        switcher.backup_dir, profile_id, RegistrationClient(client)
    ) as lease:
        receipt = lease.upload() if args.command == "upload" else lease.cancel()
        name_committed_login(switcher, client, args.name, receipt)
    return receipt


def main(argv):
    from claude_swap.cli import _guard_root
    from claude_swap.switcher import ClaudeAccountSwitcher

    try:
        switcher = ClaudeAccountSwitcher()
        _guard_root(switcher)
        print(json.dumps(run_command(argv, switcher)))
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except ClaudeSwitchError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from None
