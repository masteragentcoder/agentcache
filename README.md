# agentcache ⚡️

Cache-aware multi-agent orchestration for LLM agents.

**Most multi-agent workflows rebuild near-identical context in every call. agentcache shares one prompt prefix across all agents, so the provider's KV cache kicks in automatically.** You pay full price once, then get a significant discount on every subsequent agent call -- up to 90% off on Anthropic and Gemini 2.5+, and model-dependent discounts on OpenAI (often 50-90% off cached input tokens).

In our benchmark runs, forked agents hit **80-99% cache rates** on the shared prefix. A 4-worker research team saves ~9,000 cached tokens per run. A 7-task DAG runs 1.6x faster than sequential with 35% overall cache hits.


## Install

```bash
pip install "git+https://github.com/masteragentcoder/agentcache.git@main"
```

With CLI and rich output:

```bash
pip install "git+https://github.com/masteragentcoder/agentcache.git@main#egg=agentcache[cli]"
```

## Features

| Feature | What it does | Why it's different |
|---|---|---|
| **Multi-agent teams** | Coordinator + specialist roles, parallel execution | All roles share one cached prefix -- role differentiation in fork prompts, not system prompts |
| **Task DAG scheduling** | Dependency graph with topological wave execution | Independent tasks run in parallel, all forking from the same session |
| **Cache-safe forks** | Branch helpers that reuse the parent's prompt prefix | The core primitive -- every fork is a cache hit opportunity |
| **Cache-break detection** | Explains exactly what caused a cache miss | System prompt diff, tool schema change, model swap -- no more guessing |
| **Microcompaction** | Trim stale tool results and thinking blocks | Reduce context before expensive calls without breaking cache alignment |
| **Tool execution in forks** | Forks can execute tools via callback | Not just text generation -- forks can act |

## Quick start

### 1. Simple session + fork

The lowest-level API. One session, one fork that shares the cached prefix:

```python
import asyncio
from agentcache import AgentSession, LiteLLMSDKProvider, ForkPolicy

async def main():
    provider = LiteLLMSDKProvider()
    session = AgentSession(
        model="gpt-4o-mini",
        provider=provider,
        system_prompt="You are a careful code assistant.",
    )

    # First call creates the cache
    reply = await session.respond("Analyze my cache layer.")
    print(reply.text)

    # Fork reuses the cached prefix -- only pays for the new prompt
    summary = await session.fork(
        prompt="Summarize the current state in 3 bullets.",
        policy=ForkPolicy.cache_safe_ephemeral(),
    )
    print(summary.final_text)

    # See what the cache did
    status = session.cache_status()
    print(f"Cache hit rate: {status.hit_rate_recent:.1%}")

asyncio.run(main())
```

### 2. Multi-agent team

Define roles, and `TeamRunner` handles the full flow: coordinator plans, specialists work in parallel, coordinator synthesizes. All from one cached session:

```python
import asyncio
from agentcache import AgentRole, TeamConfig, TeamRunner, LiteLLMSDKProvider

config = TeamConfig(
    system_prompt=(
        "You are a versatile product development assistant who can adopt "
        "different specialist roles on demand. Output style: dense, specific, "
        "no filler. Use bullet points."
    ),
    roles=[
        AgentRole(
            name="Technical Architect",
            instructions="Focus on system architecture, tech stack, data models, "
                         "security, and scalability. Name actual technologies.",
        ),
        AgentRole(
            name="UX Designer",
            instructions="Focus on user flows, information architecture, "
                         "interaction patterns, and accessibility.",
        ),
        AgentRole(
            name="Business Analyst",
            instructions="Focus on market analysis, competitive landscape, "
                         "success metrics, and monetization.",
        ),
    ],
)

async def main():
    runner = TeamRunner(LiteLLMSDKProvider(), config)
    result = await runner.run(
        "Design a mobile app for tracking personal finances",
        model="gpt-4o-mini",
    )

    # Coordinator's plan
    print("PLAN:", result.plan_text[:200], "...")

    # Each specialist's report
    for report in result.specialist_reports:
        print(f"\n[{report.role.name}]")
        print(f"  Cached tokens: {report.result.usage.cache_read_input_tokens:,}")
        print(f"  {report.text[:100]}...")

    # Synthesized final output
    print("\nFINAL:", result.final_text[:300], "...")

    # Aggregate stats
    print(f"\nTotal tokens: {result.usage.total_tokens:,}")
    print(f"Cache hit rate: {result.usage.cache_hit_rate:.1%}")

asyncio.run(main())
```

### 3. Task DAG with dependencies

Define tasks with `depends_on` edges. `DAGRunner` resolves them topologically and runs independent tasks in parallel:

```python
import asyncio
from agentcache import TaskDAG, DAGRunner, LiteLLMSDKProvider

dag = TaskDAG()

# Wave 1: independent tasks run in parallel
dag.add_task(
    id="market_research",
    name="Market Research",
    prompt_template="Analyze the market for: {product}",
)
dag.add_task(
    id="competitor_scan",
    name="Competitor Scan",
    prompt_template="Analyze competitors for: {product}",
)

# Wave 2: depends on wave 1
dag.add_task(
    id="feature_spec",
    name="Feature Spec",
    prompt_template=(
        "Write a feature spec for: {product}\n\n"
        "Market research:\n{market_research}\n\n"
        "Competitors:\n{competitor_scan}"
    ),
    depends_on=["market_research", "competitor_scan"],
)

# Wave 3: depends on wave 2
dag.add_task(
    id="go_no_go",
    name="Final Review",
    prompt_template="Go/no-go recommendation based on:\n{feature_spec}",
    depends_on=["feature_spec"],
)

async def main():
    runner = DAGRunner(LiteLLMSDKProvider())
    result = await runner.run(
        dag,
        model="gpt-4o-mini",
        system_prompt="You are a product planning assistant.",
        context_vars={"product": "AI meal planning app"},
    )

    # Inspect the schedule
    for i, wave in enumerate(result.waves):
        names = [result.tasks[tid].name for tid in wave]
        print(f"Wave {i+1}: {', '.join(names)}")

    # Final task output
    print(f"\nDecision: {result.task_result('go_no_go')[:200]}...")

    # Stats
    print(f"\nTasks: {len(result.tasks)}")
    print(f"Parallelized: {result.parallelized_count}/{len(result.tasks)}")
    print(f"Cache hit rate: {result.usage.cache_hit_rate:.1%}")
    print(f"DAG speedup: {result.elapsed:.1f}s")

asyncio.run(main())
```

### 4. Tool execution in forks

Forks can execute tools instead of stubbing them:

```python
import asyncio
from agentcache import AgentSession, LiteLLMSDKProvider, ForkPolicy

def my_tool_executor(tool_call_id: str, name: str, args: dict) -> str:
    if name == "get_weather":
        return f"Sunny, 22C in {args.get('city', 'unknown')}"
    return f"Unknown tool: {name}"

async def main():
    provider = LiteLLMSDKProvider()
    session = AgentSession(
        model="gpt-4o-mini",
        provider=provider,
        system_prompt="You are a helpful assistant with access to tools.",
    )
    await session.respond("Hello")

    result = await session.fork(
        prompt="What's the weather in Paris?",
        policy=ForkPolicy(cache_safe=True, max_turns=5),
        tool_executor=my_tool_executor,
    )
    print(result.final_text)

asyncio.run(main())
```

## Core algorithm: prefix-preserving forks

Many multi-agent implementations end up giving each agent its own system prompt, message history, or tool schema. Each agent pays full price for its own context. agentcache does the opposite: **all agents share one session, and every sub-task is a fork that preserves the parent's exact prompt prefix.**

Here's why this works. LLM providers (OpenAI, Anthropic, Google) cache the KV attention state of prompt prefixes on exact prefix matches. If two API calls share the same prefix byte-for-byte, the second call skips recomputing attention for that prefix and charges a discounted rate. The exact discount varies by provider and model -- Anthropic cached reads cost 10% of base input price (though cache writes cost 25% extra), Google gives up to 90% off on Gemini 2.5+ with implicit caching enabled by default, and OpenAI's discount is model-dependent (up to 90% off on newer models). The setup also differs: OpenAI caches automatically on prefix match, Anthropic requires `cache_control` headers, and Google uses implicit caching above a minimum token threshold.

The algorithm:

```
1. ESTABLISH PREFIX
   Create one AgentSession with a shared system prompt.
   The first LLM call computes and caches the KV state for this prefix.

2. FORK, DON'T BRANCH
   When you need parallel workers (research angles, team specialists, DAG tasks),
   don't create N new sessions. Instead, fork from the parent:

     parent messages:  [system_prompt, user_msg_1, assistant_msg_1, ...]
     fork messages:    [system_prompt, user_msg_1, assistant_msg_1, ..., NEW_USER_PROMPT]
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        identical prefix = cache hit

   Each fork appends only its unique instruction as a new user message.
   The provider sees the shared prefix and serves it from cache.

3. FREEZE THE PREFIX (CacheSafeParams)
   Before forking, snapshot everything that must be bit-identical for a cache hit:
   system_prompt, model, tool_specs, messages, reasoning_config, cache_control.
   This frozen snapshot is shared across all forks -- no drift, no surprises.

4. CLONE REPLACEMENT STATE
   If tool results were truncated in the parent (microcompaction), forks must
   make the same truncation decisions on the inherited prefix. ReplacementState
   is cloned (not reset) so the prefix bytes stay identical.

5. DETECT CACHE BREAKS
   Hash every cache-relevant parameter before and after each call.
   When cache hits drop, diff the snapshots and report exactly what changed:
   "system prompt changed (+321 chars)", "tool schema changed", "model swapped".
```

In our runs, the per-fork cache hit rate is **80-99%**. The overall session rate looks lower (34-38%) because the first call that establishes the prefix can't benefit from cache -- but every subsequent fork does.

### Why not just let the provider cache automatically?

Providers *do* cache automatically -- but only when you send the same prefix. Many common multi-agent patterns in frameworks like CrewAI, AutoGen, and LangGraph can end up fragmenting cacheable prefixes, unless you specifically design around it:

| Common pattern | What happens | Cache hit? |
|---|---|---|
| Separate session per agent | Each agent has its own system prompt and history | No -- different prefixes |
| Shared system prompt, separate histories | Prefix diverges after first message | Partial -- only system prompt cached |
| **agentcache fork** | **Exact same prefix + append-only** | **Yes -- 80-99% of prefix cached in our runs** |

agentcache doesn't replace your provider's caching -- it structures your calls so the caching actually fires. Some frameworks (notably LangGraph) support shared state and could achieve similar results with careful design, but agentcache makes prefix preservation the default.

### Text injection vs prefix forks (head-to-head)

Most multi-agent frameworks (including [open-multi-agent](https://github.com/openai/open-multi-agent)) coordinate workers by injecting context as text into **separate sessions**. Each worker starts a fresh conversation, so there's no shared prefix and no cache hits.

[`cache_comparison.py`](examples/cache_comparison.py) runs both patterns on the same task (coordinator + 3 workers, `gpt-4o-mini`):

```
                                Text injection    Prefix fork
Total input tokens                     3,804          5,571
Cached tokens                              0          4,224
Cache hit rate                          0.0%          75.8%
Wall time                              85.7s          37.4s
```

Text injection: each worker gets a fresh session with the system prompt + task pasted in. The provider sees 4 unrelated calls -- zero cache reuse.

Prefix fork: all workers fork from one session. The provider sees the same prefix repeated -- **90% cache hit per worker**, plus parallel execution cuts wall time in half.

The total input is slightly higher with forks (workers inherit the full conversation prefix, not just a summary), but the cache discount more than compensates. On Anthropic, those 4,224 cached tokens would cost 90% less; on OpenAI, model-dependent discounts apply.

### Benchmark results (from repo examples)

Run of [`deep_research.py`](examples/deep_research.py) on `gpt-4o-mini`:

```
Call 1 (plan research):    2,706 input tokens | 0 cached      (cache creation)
Call 2 (worker fork):      2,783 input tokens | 2,688 cached   (96% cache hit)
Call 3 (worker fork):      2,847 input tokens | 2,816 cached   (99% cache hit)
Call 4 (synthesis fork):   2,872 input tokens | 2,816 cached   (98% cache hit)
```

### Multi-agent team

[`agent_team.py`](examples/agent_team.py) -- coordinator + 3 specialists on `gpt-4o-mini`:

```
[ARCH]  fork:       2,829 input | 2,304 cached    (81% hit)
[UX]    fork:       2,827 input | 2,304 cached    (82% hit)
[BIZ]   fork:       2,827 input | 2,304 cached    (82% hit)

Total: 17,190 tokens | 9,088 cached (38.2% overall)
Without caching, full-price input would be 23,789 tokens.
```

### Task DAG scheduling

[`task_dag.py`](examples/task_dag.py) -- 7 tasks with dependency edges, topological sort:

```
  market_research ──┐
                    ├──> feature_spec ──┐
  competitor_scan ──┘                   ├──> implementation_plan ──> final_review
                    ┌──> ux_design ─────┘
  user_interviews ──┘
```

```
Tasks:         7 (5 parallelized across waves)
Total tokens:  17,099
Cache read:    7,168 (34.8% hit rate)
Wall time:     46.7s (vs 74.7s sequential = 1.64x speedup)
```

### Cache-break detection

When something breaks the cache, agentcache tells you exactly what happened:

```
Cache break detected.
Primary causes:
  - system prompt changed (+321 chars)
  - tool schema changed
Previous cache-read tokens: 142,880
Current cache-read tokens: 22,110
```

### Microcompaction: trimming context without breaking cache

In long-running agent sessions (coding assistants, research loops), the context window fills up with stale tool results, old thinking blocks, and verbose outputs from earlier turns. The naive fix is to truncate the history -- but that changes the prefix bytes and kills your cache.

agentcache's microcompaction solves this with two mechanisms:

**1. Tool-result budgeting** (`ToolResultBudgeter`) -- before each LLM call, scan all tool results in the message history. If a result exceeds the per-turn budget (default 8,000 tokens), replace it with a short placeholder. Crucially, replacement decisions are recorded in `ReplacementState`, which is **cloned** when forking. This means forks make the exact same replacements on the shared prefix, so the bytes stay identical and cache still hits.

**2. Policy-driven compaction** (`MicroCompactor`) -- when context approaches a threshold (default 180k tokens), automatically clear stale tool results for specific tools (shell, grep, file_read, web_fetch), strip old thinking blocks (keeping only the most recent), and remove verbose tool-use blocks. Recent turns are always preserved.

```python
from agentcache import AgentSession, LiteLLMSDKProvider
from agentcache.compact.policy import CompactPolicy

session = AgentSession(
    model="gpt-4o-mini",
    provider=LiteLLMSDKProvider(),
    system_prompt="You are a coding assistant.",
    compact_policy=CompactPolicy(
        max_input_tokens=120_000,
        preserve_last_turns=6,
        clear_thinking=True,
    ),
)

# After many turns, preview what compaction would do:
print(session.compact_preview())
# Estimated tokens before: 145,200
# Estimated tokens after:  38,400
# Actions:
#   - cleared stale tool results (~82,000 est tokens)
#   - cleared stale thinking blocks (~24,800 est tokens)
```

This matters because coding agents like Cursor, Windsurf, and Claude Code generate huge tool results (file reads, grep output, shell output) that bloat context fast. Without compaction, you either hit the context limit and crash, or you naively truncate and lose your cache prefix. Microcompaction trims the fat while keeping the prefix structure intact.

## Examples

| Example | Description |
|---|---|
| [`basic_chat.py`](examples/basic_chat.py) | Simplest usage: create a session, send a message, check cache status. |
| [`side_question.py`](examples/side_question.py) | Branch a conversation with a cache-safe ephemeral helper fork. |
| [`deep_research.py`](examples/deep_research.py) | **(Recommended)** Parallel workers explore different angles, then synthesize -- all sharing one cached prefix. |
| [`agent_team.py`](examples/agent_team.py) | Multi-agent team using `TeamRunner`: coordinator + 3 specialists. 38.2% cache hit rate in our runs. |
| [`task_dag.py`](examples/task_dag.py) | Task DAG using `DAGRunner`: 7 tasks, topological waves, 1.64x speedup. |
| [`cache_comparison.py`](examples/cache_comparison.py) | **Head-to-head**: text-injection (0% cache) vs prefix forks (75.8% cache, 2.3x faster). |
| [`compaction_demo.py`](examples/compaction_demo.py) | Microcompaction in action: tool-result budgeting, stale-result clearing, thinking-block trimming. No API key needed. |
| [`smoke_test_cache.py`](examples/smoke_test_cache.py) | Validates cache tracking with a padded system prompt past the 1,024-token minimum. |
| [`interactive_research.ipynb`](examples/interactive_research.ipynb) | Jupyter Notebook version of deep research with `nest_asyncio`. |
| [`agentcache-research-skill/`](examples/agentcache-research-skill/) | Cursor/Codex AI Skill wrapping agentcache for your coding assistant. |

## Architecture

```
agentcache/
  team/          -- AgentRole, TeamConfig, TeamRunner (plan -> specialists -> synthesis)
  dag/           -- Task, TaskDAG, DAGRunner (topological wave scheduling)
  fork/          -- ForkRunner, ForkPolicy, ForkResult (cache-safe forking primitive)
  cache/         -- CacheSafeParams, PromptStateTracker, CacheBreakExplanation
  compact/       -- MicroCompactor, ToolResultBudgeter (context trimming)
  coord/         -- Coordinator (delegates to TeamRunner), WorkerSpec
  memory/        -- SessionMemory, FileSessionMemoryStore
  providers/     -- LiteLLMSDKProvider (OpenAI, Anthropic, Gemini, Azure, etc.)
  core/          -- AgentSession, Message, Usage, ToolSpec
```

## Technical contributions

agentcache is an independent Python implementation for cache-aware multi-agent orchestration, inspired by ideas from Claude Code-style agent workflows but designed around a stronger cache-first execution model. Where most frameworks treat caching as a side optimization, agentcache makes prefix preservation the core abstraction.

This project introduces two design patterns for cache-aware LLM orchestration:

**1. Prefix-Preserving Fork (PPF)**

A multi-agent orchestration primitive where all sub-agents inherit an identical, frozen prompt prefix from a parent session, ensuring provider-side KV cache hits on every fork. The prefix is captured as an immutable snapshot (`CacheSafeParams`) before forking, and a compatibility checker validates that no fork parameter can cause prefix drift. Cache misses are diagnosed automatically by hashing prompt state before and after each call and diffing the snapshots.

```
1. Start one session with a shared system prompt
2. First call establishes the cached prefix
3. For N workers, fork instead of creating new sessions:
   parent: [system, msg1, msg2, ...]
   fork:   [system, msg1, msg2, ..., WORKER_TASK]
           ^ identical prefix = cache hit
4. Freeze prefix (model, tools, messages) before forking
5. On cache miss, diff state and report what changed
```

> Key invariant: `fork_messages[:len(prefix)] == parent_messages` byte-for-byte, for every fork.

**2. Cache-Safe Compaction (CSC)**

A context-trimming strategy that preserves cache-hit eligibility across forks. Tool results exceeding a token budget are replaced with deterministic placeholders *before* the LLM call, making the compacted form the canonical cache key. Replacement decisions are recorded in a `ReplacementState` log that is *cloned* (not reset) on fork, guaranteeing that all forks produce bit-identical prefixes after compaction. This allows aggressive context reduction (e.g. 150K to 38K tokens) without breaking prefix alignment.

```
1. Scan tool outputs before each call
2. If too large, replace with a placeholder + log it
3. The compacted version becomes the cache key
4. Forks reuse the same replacement log = identical bytes
5. Result: smaller context, same cacheable prefix
```

> Key invariant: `budgeter.enforce(prefix, cloned_state) == budgeter.enforce(prefix, original_state)` for all prefixes.

**Citation**

If you use or build on these patterns, please cite this repository:

```
@software{agentcache2025,
  author       = {masteragentcoder},
  title        = {agentcache: Cache-aware multi-agent orchestration for LLM agents},
  year         = {2025},
  url          = {https://github.com/masteragentcoder/agentcache},
  note         = {Introduces Prefix-Preserving Fork (PPF) and Cache-Safe Compaction (CSC) patterns}
}
```

## Supply-chain note

LiteLLM is pinned to `==1.83.0`. Versions 1.82.7 and 1.82.8 had reported malicious releases on PyPI. Upgrades should be deliberate, hash-locked, and tested against cache-compatibility regressions.

## License

MIT
