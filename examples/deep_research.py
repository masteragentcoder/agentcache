"""Deep research: parallel cache-safe forks investigate a topic, then synthesize.

Usage:
    python examples/deep_research.py "Why did Rome fall?"
    python examples/deep_research.py  # uses default topic
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import time

from dotenv import load_dotenv

load_dotenv()

from agentcache import AgentSession, ForkPolicy, ForkResult, LiteLLMSDKProvider, Usage

SYSTEM_PROMPT = (
    "You are a senior research analyst. You break complex questions into "
    "distinct investigation angles, then produce structured reports with "
    "evidence-backed findings.\n\n"
    "Research methodology:\n"
    "1. Identify 3-4 orthogonal angles that together cover the question\n"
    "2. For each angle, gather key facts, competing theories, and evidence\n"
    "3. Synthesize findings into a coherent narrative with confidence levels\n"
    "4. Flag unresolved questions and areas needing further investigation\n\n"
    "Output style: clear, dense, no filler. Use bullet points for findings. "
    "Cite specific names, dates, and mechanisms rather than vague summaries.\n\n"
    + "".join(
        f"Reference principle #{i}: In multi-agent research, each worker should "
        f"investigate one focused angle and return a structured report. Leaders "
        f"consume reports, never raw worker transcripts. Cache-safe forks share "
        f"the parent prefix so parallel workers benefit from prompt caching. "
        f"Workers should be ephemeral: skip cache writes for one-shot helpers. "
        f"(principle {i})\n"
        for i in range(30)
    )
)

DEFAULT_TOPIC = "What are the most promising approaches to extending human healthspan beyond 120 years?"


async def main() -> None:
    topic = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TOPIC

    if os.getenv("GEMINI_API_KEY"):
        model = "gemini/gemini-2.5-flash"
    elif os.getenv("OPENAI_API_KEY"):
        model = "gpt-4o-mini"
    else:
        print("Set GEMINI_API_KEY or OPENAI_API_KEY in .env")
        return

    provider = LiteLLMSDKProvider()
    session = AgentSession(
        model=model,
        provider=provider,
        system_prompt=SYSTEM_PROMPT,
    )

    print(f"Model: {model}")
    print(f"Topic: {topic}")
    print(f"System prompt: ~{len(SYSTEM_PROMPT):,} chars")
    print("=" * 60)

    # --- Step 1: Plan the research ---
    print("\n[1/4] Planning research angles...")
    t0 = time.time()
    plan_response = await session.respond(
        f"Break this research question into 3-4 distinct investigation angles. "
        f"For each angle, write one line: the angle name and a one-sentence scope.\n\n"
        f"Question: {topic}\n\n"
        f"Format each angle as: '1. Name: scope'"
    )
    plan_text = plan_response.text
    print(f"  ({time.time() - t0:.1f}s)")
    print(f"\n{plan_text}\n")

    angles = _parse_angles(plan_text)
    if not angles:
        print("Could not parse angles from plan. Raw output above.")
        return
    print(f"Parsed {len(angles)} angles.")

    # --- Step 2: Parallel worker forks ---
    print(f"\n[2/4] Launching {len(angles)} parallel workers...")
    t1 = time.time()
    results: list[ForkResult] = [None] * len(angles)  # type: ignore[list-item]

    async def run_worker(idx: int, angle: str) -> None:
        prompt = (
            f"Investigate this specific angle of the research question.\n\n"
            f"Original question: {topic}\n"
            f"Your angle: {angle}\n\n"
            f"Return a structured report with:\n"
            f"- 3-5 key findings (bullet points, specific facts)\n"
            f"- Competing theories or open debates\n"
            f"- Confidence level (high/medium/low) with brief justification\n\n"
            f"Be specific. Cite names, dates, mechanisms."
        )
        results[idx] = await session.fork(
            prompt=prompt,
            policy=ForkPolicy.cache_safe_ephemeral(),
        )

    tasks = [run_worker(i, angle) for i, angle in enumerate(angles)]
    await asyncio.gather(*tasks)
    elapsed_workers = time.time() - t1

    total_worker_usage = Usage()
    for i, (angle, result) in enumerate(zip(angles, results)):
        print(f"\n  Worker {i+1}: {angle[:60]}...")
        print(f"    Tokens: {result.usage.total_tokens:,} | Cache read: {result.usage.cache_read_input_tokens:,}")
        total_worker_usage = total_worker_usage + result.usage

    print(f"\n  All workers done in {elapsed_workers:.1f}s")
    print(f"  Total worker tokens: {total_worker_usage.total_tokens:,}")

    # --- Step 3: Synthesis fork ---
    print("\n[3/4] Synthesizing findings...")
    t2 = time.time()

    worker_reports = "\n\n---\n\n".join(
        f"## Angle {i+1}: {angle}\n\n{result.final_text}"
        for i, (angle, result) in enumerate(zip(angles, results))
    )

    synthesis = await session.fork(
        prompt=(
            f"You have received reports from {len(angles)} research workers "
            f"investigating different angles of this question:\n\n"
            f"Question: {topic}\n\n"
            f"Worker reports:\n\n{worker_reports}\n\n"
            f"Synthesize these into a final research brief:\n"
            f"1. Executive summary (3-4 sentences)\n"
            f"2. Key findings across all angles\n"
            f"3. Emerging consensus and open debates\n"
            f"4. Recommended next steps for deeper investigation"
        ),
        policy=ForkPolicy.cache_safe_ephemeral(),
    )
    print(f"  ({time.time() - t2:.1f}s)")

    # --- Step 4: Final report ---
    print("\n" + "=" * 60)
    print("RESEARCH REPORT")
    print("=" * 60)
    print(f"\nTopic: {topic}\n")
    print(synthesis.final_text)

    # --- Stats ---
    total_usage = plan_response.usage + total_worker_usage + synthesis.usage
    status = session.cache_status()

    print("\n" + "=" * 60)
    print("STATS")
    print("=" * 60)
    print(f"  Workers:            {len(angles)}")
    print(f"  Total input tokens: {total_usage.input_tokens:,}")
    print(f"  Total output tokens:{total_usage.output_tokens:,}")
    print(f"  Cache read tokens:  {total_usage.cache_read_input_tokens:,}")
    print(f"  Total tokens:       {total_usage.total_tokens:,}")
    print(f"  Cache hit rate:     {status.hit_rate_recent:.1%}")
    print(f"  Wall time:          {time.time() - t0:.1f}s")

    explanation = session.explain_last_cache_break()
    if explanation:
        print(f"\n  Cache break: {explanation.pretty()}")


def _parse_angles(text: str) -> list[str]:
    """Extract numbered items like '1. Name: scope' from LLM output."""
    lines = text.strip().splitlines()
    angles = []
    for line in lines:
        match = re.match(r"^\s*\d+[\.\)]\s*(.+)", line.strip())
        if match:
            angles.append(match.group(1).strip())
    return angles


if __name__ == "__main__":
    asyncio.run(main())
