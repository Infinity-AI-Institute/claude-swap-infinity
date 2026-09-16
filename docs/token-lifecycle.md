# How tokens are stored, retrieved, and refreshed

Vision-managed accounts have one central refresh owner. claude-swap obtains access
tokens from Vision and attaches them through a process-local inference proxy.
Native Claude still owns the conversation, tools, and terminal. Local-only
accounts retain their local credential lifecycle.

For setup, start with the [README](../README.md). For transfer and recovery commands,
see [managed logins](vision-managed-logins.md).

## Which tokens are involved?

| Token | Purpose | Holder after Vision handoff |
| --- | --- | --- |
| Vision API key | Authorizes the wrapper to request accounts and credentials from Vision | Local wrapper configuration |
| Provider access token | Authenticates inference requests to OpenAI or Anthropic | Vision and the requesting wrapper; native exposure differs below |
| Provider refresh token | Obtains successor provider credentials | Vision; wrapped inference instances do not receive it |

A Vision API key does not create a provider login. Account permissions still apply.

## How are tokens stored?

The shared Vision key is stored in
`${XDG_CONFIG_HOME:-~/.config}/vision/credentials.json`, together with its Vision
URL. The file is private (mode `0600`), in a private directory (mode `0700`). This
is a plaintext secret protected by filesystem permissions, not encrypted storage.
`VISION_API_KEY` overrides the shared file; the shared file overrides legacy
per-wrapper browser-sign-in credentials. Configuration must match the Vision origin.

Vision stores provider credentials encrypted with AES-256-GCM, with versioned
server encryption keys. Credential generations identify successive versions of a
login. The server decrypts credentials when issuing an access token or refreshing
the login. Encryption at rest does not prevent an authorized server process from
reading credentials.

During handoff, private local journals or escrow can temporarily contain provider
credentials so interrupted transfers can recover. A committed handoff retires the
selected local refresh grant. This is not a claim that arbitrary backups or
credentials already cached by another process have been erased.

## How are tokens retrieved?

```text
wrapper authenticates to Vision using the Vision API key
  → discovers accounts the user can use
  → requests credentials for the selected account and login
  → receives access token, identity, expiry, and generation
  → supplies authentication to the native client as described below
```

The credential response does not deliver the provider refresh token. An existing,
authorized Vision login needs no new provider `/login` on the consuming machine.
Periodic inference checks are advisory; they do not impose a separate qualification
gate before credential delivery. Provider authentication and quota can still fail.

## How does refresh work across multiple machines?

```text
access credential needs recovery
  → wrapper requests central refresh for its observed generation
  → Vision claims one credential operation for that login
  → Vision exchanges the refresh token at the provider's OAuth endpoint
  → saves the encrypted successor before identity verification
  → commits the new generation for subsequent credential requests
```

Vision performs the OAuth HTTP exchange directly. It does not launch a native CLI
to perform this central refresh. Native CLIs can be used for other operations,
including account checks; those are separate from refresh ownership.

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

Deleting the shared key file removes local configuration for both wrappers. It
does not revoke that key on the server or revoke provider credentials. Use Vision's
key controls for server-side key revocation.

## Where do the original refresh tokens come from?

Native Claude login obtains access and refresh tokens from Anthropic. The wrapper
can register a new managed login or transfer selected existing credentials from
native credential files, macOS Keychain entries, or known saved copies. The
existing-login inventory identifies duplicate copies of the selected grant.
Handoff journals preserve recovery information while known selected local copies
are retired. Use the managed-login procedure instead of copying credentials by hand.

With automatic registration disabled, a managed local account remains local and
native Claude can refresh it. Vision's exclusive refresh ownership applies only
after a committed handoff.

## How is Claude's proxy configured?

The wrapper starts a loopback HTTP server on a dynamically assigned port and
launches the native process with these environment overrides:

```text
ANTHROPIC_BASE_URL = http://127.0.0.1:<assigned-port>
CLAUDE_CODE_OAUTH_TOKEN = <random per-process proxy token>
CLAUDE_SECURESTORAGE_CONFIG_DIR = <isolated credential directory>

Claude message request
  → local proxy validates the per-process token
  → proxy obtains the current Vision access credential
  → proxy attaches Authorization: Bearer <provider access token>
  → https://api.anthropic.com
```

The random proxy token is not an Anthropic credential. Launch preparation initially
obtains a provider access token, but `InferenceProxy.environment()` replaces the
child's token with the proxy token before spawning Claude. The Vision API key is
removed from the child's launch environment.

The proxy accepts the message and token-count routes (`/v1/messages` and
`/v1/messages/count_tokens`), forwards selected headers, and streams responses.
It authenticates callers, rejects browser Origin headers, and uses a fixed upstream.
It is not a general-purpose forwarding proxy. Inference traffic passes through
this local process and then directly to Anthropic; Vision supplies credentials
but is not the inference traffic relay.

## Why does native Claude not refresh behind the wrapper?

The child has the proxy token and an isolated secure-storage location, rather
than the provider refresh grant. The launcher refuses an isolated store containing
an unexpected local file or Keychain login. It removes inherited authentication
and provider-route overrides before applying its own settings.

The proxy obtains credentials at request time. After an explicit provider 401,
it requests central recovery and requires a newer generation with a different
access token before retrying. It does not blindly replay an inference request
whose upstream outcome is unknown. Central refresh remains Vision's responsibility.

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

- [vision_session.py](../src/claude_swap/vision_session.py): launch isolation and credential recovery.
- [vision_proxy.py](../src/claude_swap/vision_proxy.py): loopback server, environment overrides, forwarding, and 401 recovery.
- [vision_existing_handoff.py](../src/claude_swap/vision_existing_handoff.py): existing-grant transfer and live-handoff limits.
- [vision_handoff.py](../src/claude_swap/vision_handoff.py): managed-profile ownership transfer.
- [vision_token.py](../src/claude_swap/vision_token.py): shared Vision key configuration.
- Vision server: `apps/server/src/lib/agent-credentials/{cipher,refresh,provider-refresh}.ts`
  and `supabase/migrations/00040_agent_credential_storage.sql` define encryption and dispatch coordination.
