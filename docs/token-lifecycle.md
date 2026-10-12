# How tokens are stored, retrieved, and refreshed

Vision-managed accounts have one central refresh owner. claude-swap obtains access
tokens from Vision and attaches them through a process-local inference proxy.
Native Claude still owns the conversation, tools, and terminal. Local-only
accounts retain their local credential lifecycle.

For setup, start with the [README](../README.md#get-started-with-infinity-vision).
For transfer and recovery commands, see [managed logins](vision-managed-logins.md).

## Which tokens are involved?

| Token | Purpose | Holder after Vision handoff |
| --- | --- | --- |
| Vision API key | Authorizes the wrapper to request accounts and credentials from Vision | Local wrapper configuration |
| Provider access token | Authenticates inference requests to Anthropic | Vision and the requesting wrapper; native exposure differs below |
| Provider refresh token | Obtains successor provider credentials | Vision; wrapped inference instances do not receive it |

The Vision API key is the only credential a Vision user configures. `cswap list`
and `cswap run` need no provider login on the machine. The key does not create a
provider login, and Vision's account permissions still decide which accounts the
key's owner can use (see [Which accounts you get](../README.md#which-accounts-you-get)).

## How are tokens stored?

cswap uses the first Vision API key source that is configured:

1. The `VISION_API_KEY` environment variable, sent to the origin in
   `VISION_API_URL` (default `https://vision.infinity.inc`).
2. The shared key file `${XDG_CONFIG_HOME:-~/.config}/vision/credentials.json`.
   `cswap --set-vision-token` writes it, and Infinity's `codex-swap` reads the
   same file.
3. A key saved by this wrapper's `cswap vision login` browser sign-in, in
   `vision-auth/key.json` under the cswap backup directory (see
   [Data locations](../README.md#data-locations)).

The shared file and the browser sign-in key each store the Vision origin with the
key. If `VISION_API_URL` is set to a different origin, cswap refuses the saved key
instead of sending it to that origin. cswap creates the shared file with mode
`0600` in a directory with mode `0700`. On macOS and Linux, cswap refuses to read
the key unless you own the file and its directory and neither grants group or
other access. It also refuses to save the key into a directory that fails this
check. The key is a plaintext secret protected by filesystem permissions, not
encrypted storage.

Vision stores provider credentials encrypted with AES-256-GCM, with versioned
server encryption keys. Credential generations identify successive versions of a
login. The server decrypts credentials when it issues an access token, refreshes
the login, or runs a periodic inference check. Encryption at rest does not prevent
an authorized server process from reading credentials.

During handoff, private local journals contain provider credentials so that an
interrupted transfer can recover. A managed login uses
`vision-logins/<profile-id>/.vision-handoff.json`, and an existing-login transfer
uses `vision-migrations/<request-id>.json`, both under the cswap backup directory.
Before Vision commits a handoff, the wrapper deletes the known local copies of the
selected refresh grant. After the commit, it clears them from the journal. A
cancelled or expired handoff restores the local copies, unless a newer local login
has replaced them. This is not a claim that arbitrary backups or credentials
already cached by another process have been erased.

## How are tokens retrieved?

```text
wrapper authenticates to Vision using the Vision API key
  → discovers accounts the user can use
  → requests credentials for the selected account and login
  → receives access token, identity, expiry, and generation
  → supplies authentication to the native client as described below
```

The credential response does not deliver the provider refresh token. The wrapper
rejects a response that contains fields other than the expected ones. An existing,
authorized Vision login needs no new provider `/login` on the consuming machine.
Periodic inference checks are advisory; they do not impose a separate qualification
gate before credential delivery. Provider authentication and quota can still fail.

If Vision refuses the key, the error names the fix. `unauthorized` means that
Vision did not accept the key. Create a new key, then put it in `VISION_API_KEY` if
that variable is set, or save it with `cswap --set-vision-token`. `not_permitted`
means that the key's owner is not an invited member or the key lacks
agent-account access. Ask a Vision admin.

## How does refresh work across multiple machines?

```text
access credential needs recovery
  → wrapper requests central refresh for its observed generation
  → Vision answers current or superseded, or queues one refresh job for that generation
  → a Vision refresh worker claims the job and one credential operation for that login
  → the worker exchanges the refresh token at the provider's OAuth endpoint
  → saves the encrypted successor before identity verification
  → commits the new generation for subsequent credential requests
```

If the current credential expires more than five minutes from now, or has no
known expiry and is less than five minutes old, Vision answers `current`. If the
wrapper names a generation older than the current one, Vision answers
`superseded`. In both cases Vision does not queue a refresh.

Vision performs the OAuth HTTP exchange directly. It does not launch a native CLI
to perform this central refresh. Vision's periodic inference checks do run a
native CLI in an isolated runner; those checks are separate from refresh ownership.

Database coordination permits only one active credential operation per login and
credential kind. An atomic dispatch claim happens before the provider request.
Concurrent workers cannot both dispatch that operation. A request for an old
generation does not authorize independently refreshing an old credential copy.

If the provider might have consumed the refresh token but its response is lost,
Vision records or preserves an ambiguous operation instead of blindly replaying
the exchange. This prevents duplicate dispatch; it does not guarantee automatic
recovery after every network failure. Reauthentication can be necessary.

## What if Vision is unavailable or access is revoked?

The wrapper cannot obtain new credentials without authorized Vision access. It
does not fall back to refreshing a centrally owned grant locally. Already issued
provider access tokens are not automatically revoked merely because the Vision
key is removed or revoked; their provider-side validity is separate.

Deleting the shared key file removes it for both cswap and `codex-swap`. If
`VISION_API_KEY` is not set, cswap then uses a browser sign-in key if one exists.
Deleting the file does not revoke the key on the server or revoke provider
credentials. Use Vision's key controls for server-side key revocation.

## Where do the original refresh tokens come from?

Native Claude login obtains access and refresh tokens from Anthropic. The wrapper
can register a new managed login or transfer selected existing credentials from
native credential files, macOS Keychain entries, or known saved copies. The
existing-login inventory identifies duplicate copies of the selected grant.
Handoff journals preserve recovery information while known selected local copies
are retired. Use the managed-login procedure instead of copying credentials by hand.

If automatic registration is disabled, or no Vision key is configured, a managed
login remains local and native Claude can refresh it. Vision's exclusive refresh
ownership applies only after a committed handoff.

## How is Claude's proxy configured?

The wrapper starts a loopback HTTP server on a dynamically assigned port and
launches the native process with these environment overrides:

```text
ANTHROPIC_BASE_URL = http://127.0.0.1:<assigned-port>
CLAUDE_CODE_OAUTH_TOKEN = <random per-process proxy token>
CLAUDE_SECURESTORAGE_CONFIG_DIR = <backup directory>/vision-sessions/<origin hash>/<login id>

Claude message request
  → local proxy validates the per-process token
  → proxy selects an enabled, authorized Vision login and obtains its current access credential
  → proxy attaches Authorization: Bearer <provider access token>
  → https://api.anthropic.com
```

The random proxy token is not an Anthropic credential. Launch preparation initially
obtains a provider access token, but `InferenceProxy.environment()` replaces the
child's token with the proxy token before spawning Claude. `VISION_API_KEY` and
`VISION_API_URL` are removed from the child's launch environment.

The proxy accepts the message and token-count routes (`/v1/messages` and
`/v1/messages/count_tokens`), forwards selected headers, and streams responses.
It authenticates callers, rejects browser Origin headers, and uses a fixed upstream.
It is not a general-purpose forwarding proxy. Inference traffic passes through
this local process and then directly to Anthropic; Vision supplies credentials
but is not the inference traffic relay.

`cswap run` chooses among the enabled, authorized Vision logins on each request,
using the autoswitch thresholds and central usage. The selection stays inside the
running proxy and does not change the global active account. For the selection
rules, see [VISION.md](VISION.md).

## Why does native Claude not refresh behind the wrapper?

The child has the proxy token and an isolated secure-storage location, rather
than the provider refresh grant. The launcher refuses an isolated store containing
an unexpected local file or Keychain login. It removes inherited authentication
and provider-route overrides before applying its own settings.

The proxy obtains credentials at request time. After an explicit provider 401, it
requests central recovery for the rejected login and accepts only a newer
generation with a different access token. If that login does not recover, the
proxy can replay the request on another eligible login. It excludes the rejected
login for 60 seconds, or until Vision reports a newer generation. After an explicit
provider 429, the proxy can replay the request on another account with known
quota. It replays a request at most once, and it does not replay an inference
request whose upstream outcome is unknown. Central refresh remains Vision's
responsibility.

The original native configuration and conversation home remain in use. Credential
storage isolation does not require copying or transferring conversation sessions.

## Can an existing native session still refresh independently?

Yes. A process started before handoff can retain the original refresh token.
Strict handoff checks known native writers and matching credential copies. It
does not stop sessions automatically. The explicit `--allow-live-handoff` option
acknowledges that this refresh race remains; it does not disable native refresh
inside an already-running process.

A running old process can refresh or rewrite the grant later and invalidate
Vision's copy. Unknown backups are also outside the inventory's guarantees.
Neither environment configuration nor private directories form an OS sandbox
against another process running as the same user.

## Which parts do we maintain?

The loopback proxy, secure-storage isolation, central recovery, and handoff rules
are our integration. This document does not claim that Anthropic guarantees this
entire combination as a stable external-auth protocol. Native upgrades need
compatibility checks for routing, token handling, secure-storage selection, and
streaming. Tests with the supported native version provide evidence; they do not
guarantee compatibility with every future release.

## Implementation references

- [vision.py](../src/claude_swap/vision.py): Vision API key precedence, registry client, and credential response validation.
- [vision_token.py](../src/claude_swap/vision_token.py): shared Vision key file.
- [vision_session.py](../src/claude_swap/vision_session.py): launch isolation and credential recovery.
- [vision_proxy.py](../src/claude_swap/vision_proxy.py): loopback server, environment overrides, forwarding, and 401 and 429 replay.
- [vision_pool.py](../src/claude_swap/vision_pool.py): per-request login selection and fallback to another login.
- [vision_existing_handoff.py](../src/claude_swap/vision_existing_handoff.py): existing-grant transfer and live-handoff limits.
- [vision_handoff.py](../src/claude_swap/vision_handoff.py): managed-profile ownership transfer.
- Vision server, in the [vision repository](https://github.com/Infinity-AI-Institute/vision):
  `apps/server/src/lib/agent-credentials/{cipher,refresh,provider-refresh,refresh-worker}.ts`,
  `supabase/migrations/00040_agent_credential_storage.sql`, and
  `supabase/migrations/00047_agent_refresh_worker.sql` define encryption, the refresh
  queue, and dispatch coordination.
