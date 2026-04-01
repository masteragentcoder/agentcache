# agentcache ⚡️

Cache-aware multi-agent orchestration for LLM agents.

**Every other multi-agent framework creates a fresh context per agent. agentcache shares one prompt prefix across all agents, so the provider's KV cache kicks in automatically.** You pay full price once, then get 50% off (OpenAI), 75% off (Google), or 90% off (Anthropic) on every subsequent agent call -- with zero code changes to your prompts.

In practice, forked agents hit **80-99% cache rates** on the shared prefix. A 4-worker research team saves ~9,000 cached tokens per run. A 7-task DAG runs 1.6x faster than sequential with 35% overall cache hits.


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

Most multi-agent frameworks create a fresh context per agent. Each agent pays full price for its own system prompt and message history. agentcache does the opposite: **all agents share one session, and every sub-task is a fork that preserves the parent's exact prompt prefix.**

Here's why this works. LLM providers (OpenAI, Anthropic, Google) cache the KV attention state of prompt prefixes. If two API calls share the same prefix byte-for-byte, the second call skips recomputing attention for that prefix and charges a discounted rate (50% off on OpenAI, 75% off on Google, 90% off on Anthropic).

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

The per-fork cache hit rate is typically **80-99%**. The overall session rate looks lower (34-38%) because the first call that establishes the prefix can't benefit from cache -- but every subsequent fork does.

### Why not just let the provider cache automatically?

Providers *do* cache automatically -- but only if you send the same prefix. Most frameworks break this by accident:

| Framework pattern | What happens | Cache hit? |
|---|---|---|
| Separate session per agent | Each agent has its own system prompt and history | No -- different prefixes |
| Shared system prompt, separate histories | Prefix diverges after first message | Partial -- only system prompt cached |
| **agentcache fork** | **Exact same prefix + append-only** | **Yes -- 80-99% of prefix cached** |

agentcache doesn't replace your provider's caching -- it structures your calls so the caching actually fires.

### Proven results

Real run of [`deep_research.py`](examples/deep_research.py) on `gpt-4o-mini`:

```
Call 1 (plan research):    2,706 input tokens | 0 cached      (cache creation)
Call 2 (worker fork):      2,783 input tokens | 2,688 cached   (96% cache hit)
Call 3 (worker fork):      2,847 input tokens | 2,816 cached   (99% cache hit)
Call 4 (synthesis fork):   2,872 input tokens | 2,816 cached   (98% cache hit)
```

### Multi-agent team (real run)

[`agent_team.py`](examples/agent_team.py) -- coordinator + 3 specialists on `gpt-4o-mini`:

```
[ARCH]  fork:       2,829 input | 2,304 cached    (81% hit)
[UX]    fork:       2,827 input | 2,304 cached    (82% hit)
[BIZ]   fork:       2,827 input | 2,304 cached    (82% hit)

Total: 17,190 tokens | 9,088 cached (38.2% overall)
Without caching, full-price input would be 23,789 tokens.
```

### Task DAG scheduling (real run)

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
| [`agent_team.py`](examples/agent_team.py) | Multi-agent team using `TeamRunner`: coordinator + 3 specialists. 38.2% cache hit rate proven. |
| [`task_dag.py`](examples/task_dag.py) | Task DAG using `DAGRunner`: 7 tasks, topological waves, 1.64x speedup. |
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

## Supply-chain note

LiteLLM is pinned to `==1.83.0`. Versions 1.82.7 and 1.82.8 had reported malicious releases on PyPI. Upgrades should be deliberate, hash-locked, and tested against cache-compatibility regressions.

## License

MIT
