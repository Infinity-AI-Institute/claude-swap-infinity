# Vision account registry integration

A Vision API key is the only credential or configuration a new user needs. With
`VISION_API_KEY` set, or a key saved with `cswap --set-vision-token`, `cswap list`
pulls the Claude accounts that Vision grants the key's owner. `cswap run` then
launches native Claude with an access credential from Vision. No provider
credential, `/login`, host registration, or hand-written configuration file is
needed. For the install and first-run commands, see the
[README](../README.md#get-started-with-infinity-vision).

Without an account argument, `cswap run` uses the enabled, use-authorized Claude
accounts. An explicit `cswap run ACCOUNT` or directory mapping still selects that
account. Without Vision configuration, the existing local default launch is
unchanged. Managed provider login uploads by default when Vision is configured.
Existing-profile handoff is available through the migration commands in
[vision-managed-logins.md](vision-managed-logins.md). Automatic recovery from
provider authentication rejection requests a successor from Vision or selects
another eligible account. On September 14, 2026, this wrapper launched and resumed
native Claude on a fresh Linux EC2 client with only a Vision API key. Vision's
[qualification evidence](https://github.com/Infinity-AI-Institute/vision/blob/main/docs/agent-accounts/qualification.md#live-launch-and-resume-evidence) records that run.

The saved key is shared by both Infinity swap tools. Its file is
`${XDG_CONFIG_HOME:-~/.config}/vision/credentials.json` (private directory and file).
The key is bound to the URL saved with it. Configuration precedence is
`VISION_API_KEY`, shared saved key, then this wrapper's browser sign-in.

Alternatively, with neither an environment nor shared key configured, run
`cswap vision login`, open the printed approval URL, and compare its code before
approving. Browser sign-in saves a wrapper-specific key privately. If an environment
or shared key already exists, `vision login` reports that configuration instead.
To return to browser sign-in, remove the shared file and unset `VISION_API_KEY`.
Removing the file affects both wrappers and does not revoke the server key.
`VISION_API_URL` defaults to
`https://vision.infinity.inc`; an override must be an HTTPS origin, except for
loopback HTTP during local integration testing. The client does not follow HTTP
redirects. No provider credential is required to discover accounts or read usage.

Listing, usage snapshots and explicit launch resolution synchronize the complete
authorized Claude pool. Sync retains local profiles, preserves existing remote
aliases and disable preferences, and removes revoked remote rows. It refreshes
membership at most once per 30 seconds for the same origin and API key, unless
forced. A failed request preserves the roster, and a late response cannot overwrite
a newer sync. Each remote row stores identity and subscription metadata only.

If the sync fails, `cswap list`, `cswap run`, and `cswap run ACCOUNT` report
Vision's error. For `unauthorized` and `not_permitted`, the error names the fix: a
current API key, or access from a Vision admin. If there are no accounts to show,
`cswap list` exits 1 with that error (a JSON error envelope with `--json`). It does
not start the local first-run prompt. If Vision grants the key no Claude accounts
and no other accounts exist, `cswap list` says so on stderr.

Remote usage comes from Vision's usage worker. The client reads all pages before
returning observations and matches each result to the registry origin, account ID
and login ID. It does not call the provider usage endpoint or refresh a provider
token for these rows. Native local accounts retain their existing usage workflow.

The displayed observation age starts at the provider measurement time, not at the
client's download time. Stale, unsupported, failed or unrepresentable observations
cannot guide account selection; available historical window values remain visible.
The client's existing maximum age for selection also applies to observations that
Vision still labels fresh. An unavailable or unauthorized registry never implies
zero usage or available capacity.

Validation is synthetic unless a test explicitly opts into a real local Vision
stack. Run `.venv/bin/pytest -q -n 4` for the regression suite. Registry tests live
in `tests/test_vision.py`, `tests/test_vision_registry.py`, and
`tests/test_vision_usage.py`.

Remote launch runs a loopback inference adapter beside the native process. Native
receives a random process capability through `CLAUDE_CODE_OAUTH_TOKEN` and sends
message requests to the adapter. The adapter gets authorized access credentials
from Vision for each request and attaches them to the fixed Anthropic endpoint.
It rejects redirects and forwards only message and token-count routes. It never
writes a refresh token or copies the local account backup. An unavailable
credential requests recovery from Vision with the observed generation and waits
up to 90 seconds. Revoked access
fails without requesting refresh. Claude keeps its existing native config and
conversation home (`CLAUDE_CONFIG_DIR`, or `~/.claude` when unset). Swapping
accounts changes credentials, not the session store: `--resume` and `--continue`
work across accounts and after restarting without copying conversations.
Only secure credential storage is isolated under
`vision-sessions/<registry-hash>/<login-id>` in the backup directory, using native
`CLAUDE_SECURESTORAGE_CONFIG_DIR`. Existing local credentials are untouched.
A credential in that isolated store blocks launch until ownership handoff is
resolved. Sharing flags do not alter the existing native home for Vision launches.

Conversations created by earlier versions under
`<backup>/vision-sessions/<registry-hash>/<login-id>` remain untouched. To resume
one, select that existing directory as the native home:

```sh
CLAUDE_CONFIG_DIR=/absolute/path/to/the/old/vision-sessions/registry-hash/login-id \
  cswap run work -- --resume CONVERSATION_ID
```

Keep that native home selected when changing accounts. No copy or import is
required, and new default launches use the ordinary native home.

Request-time selection reads the complete central usage snapshot and chooses among
enabled, authorized remote logins. It uses the existing threshold, per-window
threshold, hysteresis, cooldown and consume-first ranking settings. The model in
each native request adds its weekly limit to the decision; an unknown model name
includes all reported model windows. Hard exhaustion or removed membership can
bypass proactive cooldown. Selection stays private to this process and does not
change the global active account or the native conversation directory.

Each request reads local disable preferences and obtains a central credential,
including when membership is still within its 30-second discovery cache. Unknown
or stale observations never qualify a new account. If observations are unavailable,
the existing authorized login may continue; the client does not infer spare quota
from a failed observation. If the current account is known exhausted and no eligible
account has known headroom, the adapter returns an error before provider inference.
An explicit provider 401 can trigger one replay before any response is sent to
native. Recovery accepts only a newer central generation with a different token.
It respects the server's refresh scheduling and does not refresh subscription-only
tokens. If the rejected login cannot recover, it is excluded for 60 seconds (or
until a new discovered generation appears) and selection tries another account
with known quota. A second 401 is returned without another replay. Transport
failures and partial streams are not replayed by the adapter.

An explicit provider 429 records an account-wide retry deadline and may replay
once on another account with fresh known quota. All logins and credential
generations for the limited account share that deadline. `Retry-After` accepts
delta seconds or an HTTP date; missing or malformed values use 60 seconds.
A second 429 is returned without another replay, and its account deadline
is also recorded. With no eligible alternative, native receives the provider's
429 and retry header. Subsequent requests during a fully blocked pool receive 429
with the earliest remaining delay without sending provider requests. These
backoffs are local to the running adapter; central usage remains authoritative
for selection across processes and hosts.

The native macOS qualification uses Claude 2.1.270 with SHA-256
`a506b6d970a4cf44f6abdb53a81ddcd5d3b0ce042a95c502fe9d1f946bdb8807`.
`tests/test_vision_native.py` opts in via `CLAUDE_NATIVE_TEST_BINARY`: it blocks
external network access and user-home reads and serves synthetic inference. The
cases cover native resume across accounts, preexisting default-home resume, token rotation between
two turns of one live process, and quota-driven account switching between two turns
of one live process. Another case rejects the first access token and verifies
recovery without replacing the native process. A provider-429 case verifies
account fallback within the same conversation. They check the bearer, conversation
ID, retained message context, and absence of a local credential file. This does
not establish live-provider or Linux acceptance. The live Linux run is recorded
separately in Vision's [qualification evidence](https://github.com/Infinity-AI-Institute/vision/blob/main/docs/agent-accounts/qualification.md#live-launch-and-resume-evidence).

`ManagedLoginHandoff` implements recoverable registration for dedicated profiles
under `vision-logins/<profile-id>`. Its caller must keep the profile lease across
native login and upload. The journal is private, bounded, atomically replaced and
fsynced before native credentials are removed. It retains both the plaintext seed
and Keychain value until Vision confirms ownership, recovers lost replies with
the original proof, and restores a cancelled grant without overwriting a newer
login. An active or unreadable native session prevents handoff confirmation.

The managed login CLI uses this backend. It does not inventory existing default
profiles, portable backups or external Claude homes. For those, use the
`existing-logins` and `migrate-login` commands in
[vision-managed-logins.md](vision-managed-logins.md). A dedicated profile
must not be populated by copying an existing refresh-token backup into it.


## Managed provider login

With your Vision key configured, run:

```sh
cswap vision account-login work
cswap run work
```

The first command starts `claude auth login` in a dedicated private profile. A
successful login that changes credential material uploads by default, completes
handoff after the native login process exits, and maps `work` to the verified
remote login when the alias is available. A failed or unchanged login does not
upload. If Vision is not configured, the login stays local; no fabricated key is
used. The returned registration receipt contains no provider token or handoff proof.

To retain managed login credentials locally, persist the opt-out before login:

```sh
cswap vision auto-register off
cswap vision account-login personal
```

`cswap vision auto-register on` re-enables automatic upload for future managed
logins. The preference is read back after writing and checked again after native
login exits; damaged preferences fail explicitly instead of reverting to upload.
Existing default-profile logins are not intercepted by this command yet.

Recover or cancel an interrupted upload with its existing name:

```sh
cswap vision upload work
cswap vision cancel-upload work
cswap vision profiles
```

An explicit upload remains available when auto-register is off. A pending upload
must be recovered or cancelled before starting another login in the same profile.
The profile listing contains names, IDs and the upload preference, never secrets.


## Browser sign-in recovery

`cswap vision login --no-wait` prints public approval metadata and returns.
`cswap vision status` polls the pending request and saves its issued key;
`cswap vision cancel` cancels it and removes any matching partially saved key.
Interrupted polling reuses the original proof and honors the saved next-poll time.
The approval URL must belong to the requested Vision origin. To use an isolated
local stack, put `--url http://localhost:PORT` before the command name.

Private proofs and keys live under `vision-auth/` in the backup directory, outside
preferences. A key is saved durably before pending proofs are removed, allowing a
lost response or local write failure to recover the same issued key. Cancellation
keeps recovery state until the server confirms cancellation. Keys and device
proofs never appear in command output or approval URLs.

`tests/test_vision_stack.py` is an opt-in real local-stack test using
`VISION_TEST_URL`, `SUPABASE_URL`, and `SUPABASE_ANON_KEY`. It requires loopback
origins and the local seeded administrator, approves its own request, checks
registry access and hardware denial, revokes its key, then checks revocation. It
uses no provider credentials and cleans up its own request/key in a finalizer.

## Selected managed-profile batches

Preview only the named profiles, then pass the returned confirmation value:

```sh
cswap vision batch-upload first second
cswap vision batch-upload first second --confirm CONFIRMATION
```

Preview performs no registry request and shows no credential material. Confirmation
binds the ordered selection, exact local store/transaction state, registry origin
and API key. Changed credentials or a changed key require a new preview. Apply
checks each profile again under its handoff lease; one unavailable profile does
not abandon the remaining selection. Each successful item uses the same durable
registration recovery as a single upload. Profiles not selected are untouched.
This command selects dedicated managed profiles. For inventory and handoff of
existing default/external profiles, use the migration commands documented in
[vision-managed-logins.md](vision-managed-logins.md).
