"""Side-by-side comparison: text-injection vs prefix-preserving forks.

Most multi-agent frameworks (including open-multi-agent-main) coordinate
agents by injecting context as text into separate conversations. Each agent
starts a fresh session, so there's no shared prefix and no cache hits.

agentcache uses the opposite pattern: one session, forked prefixes. Every
worker shares the parent's exact prompt prefix, triggering provider-side
cache hits automatically.

This example runs both patterns on the same task and compares cache behavior.
"""

import asyncio
import os
import time

from dotenv import load_dotenv

load_dotenv()

from agentcache import AgentSession, ForkPolicy, LiteLLMSDKProvider


SYSTEM_PROMPT = (
    "You are a product strategy assistant. You provide dense, specific analysis "
    "with concrete recommendations. Use bullet points. Be thorough but concise.\n\n"
    + "Reference framework:\n" + "=" * 80 + "\n"
    + "\n".join(
        f"- Principle {i+1}: {'Analyze market dynamics and competitive positioning. ' * 3}"
        for i in range(30)
    )
    + "\n" + "=" * 80
)

GOAL = "Design a mobile app for tracking personal carbon footprint"

WORKER_TASKS = [
    "Analyze the competitive landscape: who are the top 5 competitors, what are their strengths and weaknesses?",
    "Define the target user personas: who would use this app, what are their pain points?",
    "Propose a monetization strategy: freemium, subscription, partnerships, or other models?",
]


async def pattern_a_text_injection():
    """How most frameworks do it: separate session per worker, context via text."""
    print("=" * 60)
    print("PATTERN A: Text injection (separate sessions)")
    print("  Each worker gets a fresh session with the task pasted in.")
    print("=" * 60)

    provider = LiteLLMSDKProvider()
    total_input = 0
    total_output = 0
    total_cached = 0
    start = time.time()

    coordinator_session = AgentSession(
        model=MODEL,
        provider=provider,
        system_prompt=SYSTEM_PROMPT,
    )
    plan_reply = await coordinator_session.respond(
        f"Break down this goal into worker tasks: {GOAL}"
    )
    total_input += coordinator_session.last_usage.input_tokens
    total_output += coordinator_session.last_usage.output_tokens
    total_cached += coordinator_session.last_usage.cache_read_input_tokens
    print(f"\n  [COORD] Plan: {coordinator_session.last_usage.input_tokens} input | "
          f"{coordinator_session.last_usage.cache_read_input_tokens} cached")

    worker_results = []
    for i, task in enumerate(WORKER_TASKS):
        worker_session = AgentSession(
            model=MODEL,
            provider=provider,
            system_prompt=SYSTEM_PROMPT,
        )
        injected_prompt = (
            f"Goal: {GOAL}\n\n"
            f"Coordinator plan:\n{plan_reply.text[:500]}\n\n"
            f"Your specific task:\n{task}"
        )
        reply = await worker_session.respond(injected_prompt)
        total_input += worker_session.last_usage.input_tokens
        total_output += worker_session.last_usage.output_tokens
        total_cached += worker_session.last_usage.cache_read_input_tokens
        worker_results.append(reply.text[:200])
        print(f"  [WORKER {i+1}] {worker_session.last_usage.input_tokens} input | "
              f"{worker_session.last_usage.cache_read_input_tokens} cached")

    elapsed = time.time() - start
    total = total_input + total_output
    hit_rate = (total_cached / total_input * 100) if total_input > 0 else 0

    print(f"\n  Total: {total:,} tokens | {total_cached:,} cached "
          f"({hit_rate:.1f}% hit rate) | {elapsed:.1f}s")
    return total_input, total_cached, elapsed


async def pattern_b_prefix_fork():
    """How agentcache does it: one session, forked prefixes."""
    print("\n" + "=" * 60)
    print("PATTERN B: Prefix-preserving forks (agentcache)")
    print("  All workers fork from one session, sharing the cached prefix.")
    print("=" * 60)

    provider = LiteLLMSDKProvider()
    total_input = 0
    total_output = 0
    total_cached = 0
    start = time.time()

    session = AgentSession(
        model=MODEL,
        provider=provider,
        system_prompt=SYSTEM_PROMPT,
    )
    plan_reply = await session.respond(
        f"Break down this goal into worker tasks: {GOAL}"
    )
    total_input += session.last_usage.input_tokens
    total_output += session.last_usage.output_tokens
    total_cached += session.last_usage.cache_read_input_tokens
    print(f"\n  [COORD] Plan: {session.last_usage.input_tokens} input | "
          f"{session.last_usage.cache_read_input_tokens} cached")

    async def run_worker(i: int, task: str):
        result = await session.fork(
            prompt=f"Your specific task:\n{task}",
            policy=ForkPolicy.cache_safe_ephemeral(),
        )
        return i, result

    results = await asyncio.gather(
        *[run_worker(i, task) for i, task in enumerate(WORKER_TASKS)]
    )

    for i, result in sorted(results):
        total_input += result.usage.input_tokens
        total_output += result.usage.output_tokens
        total_cached += result.usage.cache_read_input_tokens
        print(f"  [WORKER {i+1}] {result.usage.input_tokens} input | "
              f"{result.usage.cache_read_input_tokens} cached "
              f"({result.usage.cache_read_input_tokens / result.usage.input_tokens * 100:.0f}% hit)")

    elapsed = time.time() - start
    total = total_input + total_output
    hit_rate = (total_cached / total_input * 100) if total_input > 0 else 0

    print(f"\n  Total: {total:,} tokens | {total_cached:,} cached "
          f"({hit_rate:.1f}% hit rate) | {elapsed:.1f}s")
    return total_input, total_cached, elapsed


async def main():
    print(f"Model: {MODEL}")
    print(f"System prompt: ~{len(SYSTEM_PROMPT):,} chars")
    print(f"Workers: {len(WORKER_TASKS)}")
    print()

    input_a, cached_a, time_a = await pattern_a_text_injection()
    input_b, cached_b, time_b = await pattern_b_prefix_fork()

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"  {'':30s} {'Text injection':>16s}  {'Prefix fork':>16s}")
    print(f"  {'Total input tokens':30s} {input_a:>16,}  {input_b:>16,}")
    print(f"  {'Cached tokens':30s} {cached_a:>16,}  {cached_b:>16,}")
    hit_a = (cached_a / input_a * 100) if input_a else 0
    hit_b = (cached_b / input_b * 100) if input_b else 0
    print(f"  {'Cache hit rate':30s} {hit_a:>15.1f}%  {hit_b:>15.1f}%")
    print(f"  {'Wall time':30s} {time_a:>15.1f}s  {time_b:>15.1f}s")
    if cached_b > cached_a:
        print(f"\n  Prefix fork saved {cached_b - cached_a:,} more cached tokens.")
    print()


if __name__ == "__main__":
    if os.getenv("OPENAI_API_KEY"):
        MODEL = "gpt-4o-mini"
    elif os.getenv("GEMINI_API_KEY"):
        MODEL = "gemini/gemini-2.5-flash"
    elif os.getenv("ANTHROPIC_API_KEY"):
        MODEL = "anthropic/claude-sonnet-4-20250514"
    else:
        print("Set OPENAI_API_KEY, GEMINI_API_KEY, or ANTHROPIC_API_KEY in .env")
        exit(1)

    asyncio.run(main())
