# Vision account registry integration

This branch can discover authorized Claude accounts and display their centrally
observed usage. Explicit `cswap run ACCOUNT` also launches a remote account with
an access-only credential. Managed provider login now uploads by default when
Vision is configured. Existing-profile handoff and automatic recovery during a
running session remain in progress. Do not release this integration until the
remaining client and deployment acceptance checks pass.

Sign in with `cswap vision login`, open the printed approval URL in your browser,
and compare the displayed code before approving. The CLI saves a registry-only
key privately; it does not print it. Alternatively, set `VISION_API_KEY` to your
own Vision API key, which takes precedence over the saved sign-in.
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

Remote launch uses `CLAUDE_CODE_OAUTH_TOKEN`; it never writes a refresh token or
copies the local account backup. An unavailable credential requests recovery from
Vision with the observed generation and waits up to 90 seconds. Revoked access
fails without requesting refresh. The native profile lives under
`vision-sessions/<registry-hash>/<login-id>` in the backup directory, preserving
conversation history across alias changes and credential generations. Native
arguments, including `--resume`, pass through unchanged. A local credential file
in that profile is preserved and blocks launch until ownership handoff is resolved.

The current launcher fetches credentials before starting Claude. It does not yet
replace the token in an already-running native process. The native macOS
qualification uses Claude 2.1.270 with SHA-256
`a506b6d970a4cf44f6abdb53a81ddcd5d3b0ce042a95c502fe9d1f946bdb8807`.
`tests/test_vision_native.py` opts in via `CLAUDE_NATIVE_TEST_BINARY`: it blocks
external network access and user-home reads, serves synthetic inference, then
resumes the same native conversation with a second synthetic access token. It
checks the bearer, conversation ID, retained message context, and absence of a
local credential file. This does not establish live-provider or Linux acceptance.

`ManagedLoginHandoff` implements recoverable registration for dedicated profiles
under `vision-logins/<profile-id>`. Its caller must keep the profile lease across
native login and upload. The journal is private, bounded, atomically replaced and
fsynced before native credentials are removed. It retains both the plaintext seed
and Keychain value until Vision confirms ownership, recovers lost replies with
the original proof, and restores a cancelled grant without overwriting a newer
login. An active or unreadable native session prevents handoff confirmation.

The managed login CLI uses this backend. It does not inventory existing default
profiles, portable backups or external Claude homes; those remain a
separate required integration before this series is complete. A dedicated profile
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
