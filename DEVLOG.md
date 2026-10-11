<!-- ai: Codex -->
Rotation cooldown and retry fix, 2026-10-10.

The running proxy cannot observe a manual provider usage reset. Added authenticated local commands `cswap rotate-reset <alias|slot>` to clear one account's cooldown and select it, and `cswap rotate-status` to inspect local state. These commands do not reset provider quota or revive rejected credentials. Each machine has its own proxy state.

Shared rotation now retries explicit 401/429 rejections across the pool, bounded by its account count and without replaying a token twice. Transport errors, server errors, and streamed responses are not replayed. The separate native adapter retains its two-attempt limit.

Validation: 16 rotation tests passed in 0.33 seconds after reducing synthetic server shutdown polling from 0.5 to 0.01 seconds. The initial combined proxy check took 6.06 seconds. Final rotation, native proxy, and CLI checks passed: 157 tests in 5.65 seconds. Run with `env -u VISION_API_KEY -u VISION_API_URL uv run --no-sync pytest tests/test_rotate_proxy.py tests/test_vision_proxy.py tests/test_cli.py -n 4 -q`; an earlier unisolated CLI check inherited live Vision configuration and failed its empty-account expectation.

Installed locally with `uv tool install --reinstall .` and restarted the local service once to load the code. Verified both control commands. One live Haiku request received two successive account rate limits, reached the third account, and returned HTTP 200 with `Ok.`. No additional client request or restart was needed to cross the two limited accounts.

This fix is based on the still-open rotation feature in PR #25.
<!-- /ai -->
