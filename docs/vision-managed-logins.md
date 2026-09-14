# Managed Claude logins (development branch)

The Vision client commands below are implemented on this branch. They are not
released yet; unattended session recovery and full rollout acceptance
still need implementation and testing.

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
refreshers have stopped. Do not copy these credentials into a new managed profile
to bypass handoff.

## Transfer a selected existing login

Exit native sessions that can use the selected grant before transfer. This
includes every known matching credential copy and its saved-slot session home;
unrelated active profiles do not block the transfer. The selected writer scope
is saved before any credential deletion and remains enforced during recovery.
Older recovery journals lack that scope and conservatively require all known
profiles to be idle. An explicit native config/secure-storage split fences both
homes. A selected credential found in an earlier central secure store also
requires all known profiles idle, because its original caller home is not
recorded. Unreadable session records anywhere in inventory remain a blocker.
Use a `source_id` from the inventory:

```sh
cswap vision migrate-login SOURCE_ID
cswap vision migrate-login SOURCE_ID --request-id REQUEST_ID --confirm CONFIRMATION
```

The first command returns the request ID, confirmation value and all known copies
of the selected refresh grant. The second command applies that preview. Repeat
any `--profile PATH` options used during inventory on both commands and recovery.
A changed source, registry, API key or request ID requires a new preview.

The transfer stores exact backend bytes in a private recovery journal before
removing them. It waits for local refresh operations, takes native refresh locks,
and refuses to confirm central ownership while a known native session is active
or a credential source changes. It does not stop native sessions automatically.
Unselected refresh grants remain local.

After commit, slots whose current backup held the selected grant become one
central account using provider-verified identity. Their aliases continue to resolve
to that account, including after normal registry discovery. `cswap alias` listings
include retained aliases. An explicit alias rename or removal replaces those
retained names. Slots holding another grant are preserved.

```sh
cswap vision recover-migration REQUEST_ID
cswap vision cancel-migration REQUEST_ID
```

Recovery reuses the original registration request and proof. Cancellation restores
local bytes only after the registry confirms a cancelled or expired transfer; a
committed transfer stays central. Newer local logins are preserved. If a slot was
reassigned during transfer, recovery requires reconciling that local assignment
before restoring credentials. Interrupted central routing can be retried with
`recover-migration` without registering the login again.

Credential handoff leaves native conversations in their existing directory. No
session snapshot, import, or transfer is needed. Resume using the same native
home and original conversation ID after handoff or an account switch:

```sh
cswap run work -- --resume CONVERSATION_ID
```

For a conversation created under an explicitly selected native profile, retain
that profile when launching (for example, `CLAUDE_CONFIG_DIR=/path/to/profile`).
The wrapper isolates credential storage independently; it does not relocate or
merge conversations from other native homes.

Pinned Claude 2.1.270 has passed synthetic access-only inference, cold resume after
an account change, and resume of a conversation created in the default native
home. The live multiuser pilot and clean EC2 rollout acceptance remain separate
required checks.

## Central credentials during a native session

Central-account launches keep a local adapter running beside Claude:

```text
native message request
    -> obtain the current authorized Vision access credential
    -> attach it to the fixed Anthropic message endpoint
    -> stream the response back to the same native process
```

The adapter binds to loopback and gives the native process a random per-process
capability. Provider access tokens stay in the adapter; no provider refresh token
is delivered to either component. Only `/v1/messages` and its `count_tokens`
endpoint are forwarded. Provider redirects are rejected, and Vision revocation
blocks subsequent credential issuance before another upstream request.

Native arguments, terminal input/output and exit status are preserved. Terminal
Ctrl+C remains available to Claude to cancel a turn or exit. The adapter stops
when its child exits. It does not replay an upstream request after an uncertain
connection failure or replace credentials inside native storage.

Pinned Claude 2.1.270 has completed two synthetic turns in one process with a
central access-token change between turns and the same conversation ID. Automatic
quota-driven account switching and in-session relogin recovery remain unfinished.
