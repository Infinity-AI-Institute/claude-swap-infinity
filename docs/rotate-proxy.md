# Shared rotation proxy

`cswap rotate-proxy` serves one local endpoint that every Claude Code process on a
machine can use. It forwards each request with one setup token from a pool. When
the provider rate-limits that account, the proxy moves to the next account and
replays the request, so a rate limit is handled once for every running agent
instead of once per session.

Use it when several agents on one machine should share a set of setup-token
accounts. To run one terminal as one account, use `cswap run`. To change the
machine's active login, use `cswap switch` or `cswap auto`.

Read [Security](#security) before using it on a machine other people can log in to.

## Set up

1. Register each setup token to pool. See "Add an account from a raw token or API
   key" in the [README](../README.md).

   ```bash
   cswap add-token sk-ant-oat01-...
   ```

2. Start the proxy and leave it running. It prints its address and the accounts
   it pooled.

   ```bash
   cswap rotate-proxy
   ```

3. In each shell that starts an agent, load the two variables that point Claude
   Code at the proxy, then start Claude Code as usual.

   ```bash
   export $(cswap rotate-env)
   claude
   ```

`rotate-env` prints `ANTHROPIC_BASE_URL`, the proxy's address, and
`CLAUDE_CODE_OAUTH_TOKEN`, a local capability described under
[Security](#security). It is not a provider token.

## Which accounts are pooled

Every enabled account whose stored credential is a setup token: an OAuth access
token (`sk-ant-oat...`) stored without a refresh token. The account's label does
not matter, so a token registered with `--email` is pooled too.

These are never pooled:

- **API keys** (`sk-ant-api...`). They bill per token.
- **Browser logins.** They carry a refresh token that Claude Code owns and rotates.
- **Disabled accounts** (`cswap disable`).

Accounts are used in slot order. An account with the alias `sweep` is used last.

## How rotation works

- The proxy stays on one account until the provider answers a request with 429.
- It then sets that account aside for the time the provider's `Retry-After` header
  gives, kept between 30 seconds and 6 hours, or for 15 minutes when the header
  is missing. The same request is then tried on each remaining account, once
  each, until one accepts it.
- An account whose token the provider rejects (401) is dropped until the proxy
  restarts.
- While every account is set aside, the proxy still sends each request to the
  account due back first. If that account is still limited, the agent receives
  the provider's 429.
- When no account can serve, the agent's request fails at once with a 503 that
  states the reason and tells Claude Code not to retry. This happens when every
  token was rejected, or when one was rejected while the rest are rate-limited.

Only an explicit 401 or 429 is replayed. A connection failure, a server error, or
a response that has started streaming is passed to the agent, because the request
may already have run.

## Inspect and clear cooldowns

The proxy cannot see a usage reset that happens at the provider, for example when
you reset an account's plan usage by hand. Until the cooldown it recorded runs
out, it keeps that account set aside.

```bash
cswap rotate-status                # every pooled account and its cooldown
cswap rotate-reset <alias|slot>    # clear one cooldown and switch to that account
```

Both commands talk to the running proxy and print its answer as JSON. For each
account, the status gives the slot, the name, whether it is the active account,
whether its token was rejected (`dead`), and the seconds of cooldown left
(`cooling_s`).

They change only the proxy's own state. `rotate-reset` does not reset usage at
the provider, and it cannot restore an account whose token was rejected. If you
clear a cooldown while the account is still limited, the next request is
rate-limited once and the proxy moves on.

The commands use the same capability as agents, so any process that can use the
proxy can also read the status and clear a cooldown.

## Limits

- **Account changes need a restart.** The proxy reads the account list when it
  starts. After you add, enable, disable, or replace an account, restart the
  proxy. A restart interrupts requests that are in flight.
- **State is in memory and per machine.** A restart clears every cooldown and
  restores dropped accounts.
- **A cooldown is the proxy's estimate.** It neither shows nor changes the
  provider's usage limits.
- **One proxy per port.** The default is 41871. To use another, set
  `CSWAP_ROTATE_PORT` to the same value for every `rotate-` command.

## Security

The proxy holds the pooled setup tokens in memory and attaches one to each request
it forwards to `api.anthropic.com`. Agents never receive them. An agent holds the
*capability* instead: a random value the proxy checks on every request.

- The proxy listens on the loopback interface only (`127.0.0.1`).
- The capability is stored in the file `rotate-capability` in the claude-swap data
  directory: `~/.claude-swap-backup` on macOS and Windows, and
  `$XDG_DATA_HOME/claude-swap` on Linux (`~/.local/share/claude-swap` by
  default). On macOS and Linux the file is created with owner-only permissions.
- The capability is long-lived and shared by every agent. Anything that obtains
  it can spend the pooled accounts through the proxy while the proxy runs.
  `rotate-env` prints it, so keep that output out of logs.
- To replace the capability, stop the proxy, delete `rotate-capability`, start
  the proxy, and run `rotate-env` again in each shell.

**Do not use the proxy on a machine with local users you do not trust.** The port
is fixed, and nothing proves to an agent that the listener on that port is the
proxy. Any local user or process can listen on the port while the proxy is not
running, either before it starts or between restarts. Agents would then send
that listener the capability and their complete requests, and the listener could
later use the capability against the real proxy. `cswap run` does not have this
exposure, because it gives each session its own random port and capability.

## Commands

| Command | What it does |
|---|---|
| `cswap rotate-proxy` | Serves the endpoint in the foreground until interrupted. |
| `cswap rotate-env` | Prints `ANTHROPIC_BASE_URL` and `CLAUDE_CODE_OAUTH_TOKEN` for agents. |
| `cswap rotate-status` | Prints the running proxy's accounts and cooldowns as JSON. |
| `cswap rotate-reset <alias\|slot>` | Clears one account's cooldown in the running proxy and switches to it. |
