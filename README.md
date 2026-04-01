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

## Examples

We've provided a suite of examples showing different ways to use `agentcache`, from simple helpers to complex multi-agent swarms. Check out the `examples/` directory:

| Example | Description |
|---|---|
| [`basic_chat.py`](examples/basic_chat.py) | The simplest usage: create a session, send a message, check cache status. |
| [`side_question.py`](examples/side_question.py) | Shows how to branch a conversation: fork a cache-safe, ephemeral helper *after* a main turn. |
| [`deep_research.py`](examples/deep_research.py) | **(Recommended)** A powerful multi-agent workflow. Spawns N parallel workers exploring different angles of a topic, then synthesizes their reports—all sharing a single cacheable prefix. |
| [`smoke_test_cache.py`](examples/smoke_test_cache.py) | Validates that cache tracking actually works by padding a system prompt past the 1,024-token minimum (for Anthropic/OpenAI) and reporting hit rates. |
| [`interactive_research.ipynb`](examples/interactive_research.ipynb) | A Jupyter Notebook version of the deep research example, demonstrating how `agentcache` works cleanly in `nest_asyncio` interactive environments. |
| [`agentcache-research-skill/`](examples/agentcache-research-skill/) | A Cursor/Codex AI Skill demonstrating how you can wrap `agentcache` into a reusable tool for your own AI coding assistant. |

## Supply-chain note

LiteLLM is pinned to `==1.83.0`. Versions 1.82.7 and 1.82.8 had reported malicious releases on PyPI. Upgrades should be deliberate, hash-locked, and tested against cache-compatibility regressions.

## License

MIT
