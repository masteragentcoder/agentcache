# agentcache

Cache-aware orchestration for LLM agents.

Run helper agents that **reuse the parent's cacheable prefix** while keeping **mutable state isolated**. Detect cache breaks, compact stale context before expensive calls, and orchestrate workers with summarized handoffs instead of raw transcript sharing.

## Install

```bash
pip install "git+https://github.com/masterFoad/agentcache.git@main"
```

With CLI and rich output:

```bash
pip install "git+https://github.com/masterFoad/agentcache.git@main#egg=agentcache[cli]"
```

## Quick start

```python
from agentcache import AgentSession, LiteLLMSDKProvider, ForkPolicy

provider = LiteLLMSDKProvider()

session = AgentSession(
    model="anthropic/claude-sonnet-4-20250514",
    provider=provider,
    system_prompt="You are a careful code assistant.",
)

# Main conversation turn
reply = await session.respond("Analyze my cache layer and suggest improvements.")
print(reply.text)

# Cache-safe side helper (shares parent prefix, isolated state)
summary = await session.fork(
    prompt="Summarize the current state in 3 bullets.",
    policy=ForkPolicy.cache_safe_ephemeral(),
)
print(summary.final_text)

# Cache diagnostics
status = session.cache_status()
print(status)
```

## Supply-chain note

LiteLLM is pinned to `==1.83.0`. Versions 1.82.7 and 1.82.8 had reported malicious releases on PyPI. Upgrades should be deliberate, hash-locked, and tested against cache-compatibility regressions.

## License

MIT
