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

## How cache-safe forks save tokens

The core idea: when you fork a helper agent, it shares the parent session's exact prompt prefix. The provider caches that prefix, so every worker pays only for its unique suffix -- not the full context again.

Here's a real run of [`deep_research.py`](examples/deep_research.py) on OpenAI `gpt-4o-mini` with a ~2,700-token system prompt:

```
Call 1 (plan research):    2,706 input tokens | 0 cached      (cache creation)
Call 2 (worker fork):      2,783 input tokens | 2,688 cached   (96% cache hit)
Call 3 (worker fork):      2,847 input tokens | 2,816 cached   (99% cache hit)
Call 4 (synthesis fork):   2,872 input tokens | 2,816 cached   (98% cache hit)
```

**Without cache-safe forks**, every worker would re-process the full prefix. With 3 workers + synthesis, that's ~8,500 tokens billed at full price. **With agentcache**, only the first call pays full price; the rest get a 50% discount on the cached portion (OpenAI pricing) or 90% discount (Anthropic pricing).

On a larger research run (3 parallel workers + synthesis, ~27k total tokens):

```
Workers:            3
Total input tokens: 16,346
Cache read tokens:  2,036
Total tokens:       26,971
Cache hit rate:     46.9%
Wall time:          37.0s
```

The savings compound as the system prompt and conversation history grow. A 100k-token context with 5 parallel workers would save ~450k input tokens worth of re-processing.

### Cache-break detection

When something does break the cache (system prompt edit, tool schema change, model swap), agentcache tells you exactly what happened:

```
Cache break detected.
Primary causes:
  - system prompt changed (+321 chars)
  - tool schema changed
Previous cache-read tokens: 142,880
Current cache-read tokens: 22,110
```

No more guessing why your bill spiked.

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
