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
from claude_swap.vision import VisionError, configured_client, origin
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
            return {
                "version": 2,
                "auto_register": True,
                "auto_register_by_origin": {},
                "profiles": {},
                "bindings": {},
            }
        try:
            value = json.loads(raw)
            if (
                not isinstance(value, dict)
                or set(value)
                not in (
                    {"version", "auto_register", "profiles"},
                    {"version", "auto_register", "profiles", "bindings"},
                    {
                        "version", "auto_register", "auto_register_by_origin",
                        "profiles", "bindings",
                    },
                )
                or type(value["version"]) is not int
                or value["version"] not in (1, 2)
                or type(value["auto_register"]) is not bool
                or not isinstance(value["profiles"], dict)
            ):
                raise ValueError()
            if (value["version"] == 2) != ("auto_register_by_origin" in value):
                raise ValueError()
            overrides = value.setdefault("auto_register_by_origin", {})
            if not isinstance(overrides, dict) or len(overrides) > 1000:
                raise ValueError()
            for destination, enabled in overrides.items():
                if origin(destination) != destination or type(enabled) is not bool:
                    raise ValueError()
            # Keep the legacy global choice as the fallback. In particular,
            # upgrading an opt-out must never enable uploads to a new origin.
            value["version"] = 2
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
        except (ValueError, TypeError, AttributeError, VisionError):
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

    def auto_register(self, destination):
        state = self.read()
        return state["auto_register_by_origin"].get(
            origin(destination), state["auto_register"]
        )

    def set_auto_register(self, enabled, destination):
        if type(enabled) is not bool:
            raise SessionError("The upload preference must be enabled or disabled.")
        destination = origin(destination)
        _mkdir_private(self.root)
        with FileLock(self.root / ".vision-profiles.lock"):
            state = self.read()
            state["auto_register_by_origin"][destination] = enabled
            _write_private(self.path, json.dumps(state))
            if self.auto_register(destination) is not enabled:
                raise SessionError("The upload preference was not saved.")


def disclose_registration(profiles, destination, *, configured=True):
    destination = origin(destination)
    enabled = profiles.auto_register(destination)
    status = "enabled" if enabled else "disabled"
    context = "" if configured else "Vision is not configured; this login stays local. "
    print(
        context + f"Automatic credential registration is {status} for {destination}. "
        f"To disable it: cswap vision --url {destination} auto-register off",
        file=sys.stderr,
        flush=True,
    )


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
        if result.returncode == 0 and lease._stores() != before:
            client = configured_client()
            if client is not None and profiles.auto_register(client.url):
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
    destination = client.url if client is not None else os.environ.get(
        "VISION_API_URL", "https://vision.infinity.inc"
    )
    disclose_registration(profiles, destination, configured=client is not None)
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
        if registry is None or not profiles.auto_register(client.url):
            return {"state": "local", "profile": name}
        receipt = lease.upload()
        name_committed_login(switcher, client, name, receipt)
    return receipt


def run_command(argv, switcher):
    parser = argparse.ArgumentParser(prog="cswap vision")
    parser.add_argument(
        "--url", default=None
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
    migration = commands.add_parser("migrate-login")
    migration.add_argument("source_id")
    migration.add_argument("--request-id")
    migration.add_argument("--confirm")
    migration.add_argument("--profile", action="append", default=[])
    migration.add_argument(
        "--allow-live-handoff",
        action="store_true",
        help="Import while sessions run; acknowledge their cached credentials may refresh later.",
    )
    for command in ("recover-migration", "cancel-migration"):
        recovery = commands.add_parser(command)
        recovery.add_argument("request_id")
        recovery.add_argument("--profile", action="append", default=[])
        recovery.add_argument("--allow-live-handoff", action="store_true")
    commands.add_parser("existing-logins").add_argument(
        "--profile", action="append", default=[]
    )
    commands.add_parser("profiles")
    commands.add_parser("auto-register").add_argument("value", choices=("on", "off"))
    args = parser.parse_args(argv)
    profiles = ManagedProfiles(switcher.backup_dir)
    if args.command in {"login", "status", "cancel"}:
        if args.command in {"login", "status"}:
            from claude_swap.vision_token import saved_token_client

            environment_key = os.environ.get("VISION_API_KEY")
            client = configured_client() if environment_key else saved_token_client()
            if client is not None:
                if args.url and origin(args.url) != client.url:
                    raise SessionError("Configured Vision key belongs to another origin.")
                if args.command == "login":
                    disclose_registration(profiles, client.url)
                source = "environment" if environment_key else "shared_config"
                return {"state": "configured", "source": source, "url": client.url}
        destination = args.url or os.environ.get("VISION_API_URL", "https://vision.infinity.inc")
        flow = VisionSignIn(switcher.backup_dir, destination)
        if args.command == "cancel":
            return flow.cancel()
        if args.command == "status":
            if flow.state.read("pending") is not None:
                return flow.poll()
            client = configured_client(switcher.backup_dir)
            if client is None:
                return {"state": "signed_out"}
            return {"state": "signed_in", "url": client.url}
        disclose_registration(profiles, flow.url)
        public = flow.begin(args.host_label)
        if args.no_wait:
            return public
        print(json.dumps(public), flush=True)
        while True:
            result = flow.poll()
            if result["state"] != "pending":
                return result
            time.sleep(min(30, result["retry_after_seconds"]))
    if args.command in {"migrate-login", "recover-migration", "cancel-migration"}:
        from claude_swap.vision_existing_handoff import ExistingLoginHandoff

        client = configured_client()
        if client is None:
            raise SessionError("Sign in to Vision before migrating an existing login.")
        registry = RegistrationClient(client)
        if args.command == "migrate-login" and args.request_id is None:
            if args.confirm is not None:
                raise SessionError("Apply the preview with its original --request-id.")
            transaction = ExistingLoginHandoff.new(
                switcher,
                registry,
                args.profile,
                allow_live_handoff=args.allow_live_handoff,
            )
        else:
            transaction = ExistingLoginHandoff(
                switcher,
                registry,
                args.request_id,
                args.profile,
                allow_live_handoff=args.allow_live_handoff,
            )
        if args.command == "migrate-login" and args.confirm is None:
            return transaction.preview(args.source_id)
        if args.command == "cancel-migration":
            receipt = transaction.cancel()
        elif args.command == "recover-migration":
            receipt = transaction.upload()
        else:
            receipt = transaction.upload(args.source_id, args.confirm)
        if receipt["state"] == "committed":
            return transaction.route_committed()
        return receipt
    if args.command == "existing-logins":
        from claude_swap.vision_inventory import capture_inventory

        return capture_inventory(switcher, args.profile).public()
    if args.command == "profiles":
        return profiles.read()
    if args.command == "auto-register":
        client = None if args.url else configured_client()
        if args.url:
            destination = origin(args.url)
        elif client is not None:
            destination = client.url
        else:
            destination = origin(
                os.environ.get("VISION_API_URL", "https://vision.infinity.inc")
            )
        profiles.set_auto_register(args.value == "on", destination)
        return {"auto_register": profiles.auto_register(destination), "url": destination}
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
