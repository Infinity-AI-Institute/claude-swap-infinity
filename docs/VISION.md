# Vision account registry integration

This branch can discover authorized Claude accounts and display their centrally
observed usage. Explicit `cswap run ACCOUNT` also launches a remote account with
an access-only credential. Login handoff and automatic recovery during a running
session remain in progress. Do not release this integration until the remaining
client and deployment acceptance checks pass.

Set `VISION_API_KEY` to your own Vision API key. `VISION_API_URL` defaults to
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
