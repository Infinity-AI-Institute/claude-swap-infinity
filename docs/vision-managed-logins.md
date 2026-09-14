# Managed Claude logins (development branch)

The Vision client commands below are implemented on this branch. They are not
released yet; existing/default-profile migration and unattended session recovery
still need implementation and acceptance testing.

## Keep a managed login local

```sh
cswap vision auto-register off
cswap vision account-login personal
cswap vision account-run personal
cswap vision account-run personal -- --resume CONVERSATION_ID
```

The upload preference persists across invocations. `account-run` uses the same
native profile as `account-login`, forwards the native arguments and terminal
streams, and returns Claude's exit status. A local run with uploads disabled does
not require Vision configuration. It does not copy the default Claude profile.

The wrapper holds the profile's ownership lease until the child process exits.
Another managed command cannot upload that profile during the run. Native refresh
locks remain available to Claude itself. An unresolved upload prevents a local
launch even when automatic upload is disabled; recover or cancel that transaction
first.

## Transfer the login to Vision

```sh
cswap vision login
cswap vision upload personal
cswap run personal
```

After a committed transfer, the native refresh credential is removed and Vision
owns refresh. Use the central alias with `cswap run`; `account-run` refuses to
reuse the retired local grant. `account-login personal` creates a new native login
in that managed profile when reauthentication is needed.

Automatic registration is enabled by default. To enable it again:

```sh
cswap vision auto-register on
```

A successful managed login uploads when Vision is configured and this preference
is enabled. If native credentials change during `account-run`, the wrapper checks
the preference after a successful native exit and then uploads the new material.
It does not replace a running process's credentials. An upload failure preserves
the recovery journal; retry `upload personal` or use `cancel-upload personal`.
Cancellation restores local credentials only when the registry confirms that
ownership was not committed, and never overwrites a newer local login.

## Inspect existing credentials before migration

```sh
cswap vision existing-logins
cswap vision existing-logins --profile /path/to/existing/claude-profile
```

This read-only inventory reports source IDs, storage locations, and which known
sources contain the same refresh token. It examines the default and active native
profiles, saved file and Keychain backups, retained previous generations, orphaned
recovery files, and detached swap-managed profiles. Repeated `--profile` options
include additional native profiles. Credential values are never printed.

Unreadable or malformed credential sources stop the inventory. A live or
unreadable native session is reported explicitly. Pending managed handoff escrow
must be reconciled before inventorying existing credentials. Arbitrary portable
exports elsewhere on disk are not searched.

The result is an inventory, not an ownership transfer or proof that native
refreshers have stopped. The existing-profile transfer command is not implemented
yet; do not copy these credentials into a new managed profile to bypass handoff.
