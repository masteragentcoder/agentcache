"""Multi-agent team: heterogeneous agents with roles collaborate via a coordinator.

All specialists fork from ONE shared session so they share the same cached
prefix.  Role differentiation happens in the fork prompt, not the system
prompt -- this is the key insight for getting cache hits with agentcache.

Usage:
    python examples/agent_team.py "Design a mobile app for tracking personal finances"
    python examples/agent_team.py  # uses default goal
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

from agentcache import AgentSession, ForkPolicy, ForkResult, LiteLLMSDKProvider, Usage

PADDING = "".join(
    f"Teamwork principle #{i}: Specialists produce focused, structured reports. "
    f"The coordinator consumes summaries, never raw transcripts. Each specialist "
    f"stays within their expertise boundary. Reports use bullet points, concrete "
    f"recommendations, and confidence levels. Cache-safe forks share the parent "
    f"prefix so parallel workers benefit from prompt caching. (principle {i})\n"
    for i in range(30)
)

SYSTEM_PROMPT = (
    "You are a versatile product development assistant who can adopt different "
    "specialist roles on demand. When assigned a role, you become that expert "
    "and produce deliverables appropriate to the role.\n\n"
    "Available roles:\n"
    "- Coordinator: breaks goals into tasks, delegates, synthesizes reports\n"
    "- Technical Architect: system design, tech stack, data models, security\n"
    "- UX Designer: user flows, wireframes, interaction patterns, accessibility\n"
    "- Business Analyst: market analysis, metrics, monetization, competition\n\n"
    "Output style: dense, specific, no filler. Use bullet points. Cite real "
    "technologies, patterns, benchmarks.\n\n"
    + PADDING
)


@dataclass
class AgentRole:
    name: str
    tag: str
    brief_instructions: str


COORDINATOR = AgentRole(
    name="Coordinator",
    tag="[COORD]",
    brief_instructions=(
        "You are acting as the Coordinator. Break the goal into specialist "
        "tasks and produce delegation briefs."
    ),
)

SPECIALISTS = [
    AgentRole(
        name="Technical Architect",
        tag="[ARCH]",
        brief_instructions=(
            "You are acting as the Technical Architect. Focus on system "
            "architecture, tech stack, data models, security, and scalability. "
            "Name actual technologies and trade-offs."
        ),
    ),
    AgentRole(
        name="UX Designer",
        tag="[UX]",
        brief_instructions=(
            "You are acting as the UX Designer. Focus on user flows, "
            "information architecture, interaction patterns, and accessibility. "
            "Reference real-world apps as examples."
        ),
    ),
    AgentRole(
        name="Business Analyst",
        tag="[BIZ]",
        brief_instructions=(
            "You are acting as the Business Analyst. Focus on market analysis, "
            "competitive landscape, success metrics, and monetization. Use "
            "specific numbers and benchmarks."
        ),
    ),
]

DEFAULT_GOAL = "Design a mobile app for tracking personal finances with AI-powered insights"


async def main() -> None:
    goal = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_GOAL

    if os.getenv("OPENAI_API_KEY"):
        model = "gpt-4o-mini"
    elif os.getenv("GEMINI_API_KEY"):
        model = "gemini/gemini-2.5-flash"
    else:
        print("Set OPENAI_API_KEY or GEMINI_API_KEY in .env")
        return

    provider = LiteLLMSDKProvider()
    session = AgentSession(
        model=model,
        provider=provider,
        system_prompt=SYSTEM_PROMPT,
    )
    total_usage = Usage()
    t0 = time.time()

    print(f"Model: {model}")
    print(f"Goal:  {goal}")
    print(f"System prompt: ~{len(SYSTEM_PROMPT):,} chars")
    print("=" * 60)

    # --- Phase 1: Coordinator plans delegation ---
    print(f"\n{COORDINATOR.tag} Planning delegation...")

    plan_response = await session.respond(
        f"{COORDINATOR.brief_instructions}\n\n"
        f"You have these specialists: "
        f"{', '.join(s.name for s in SPECIALISTS)}.\n\n"
        f"Project goal: {goal}\n\n"
        f"For each specialist, write a focused one-paragraph brief describing "
        f"exactly what they should investigate and deliver. Keep it concise."
    )
    total_usage = total_usage + plan_response.usage
    print(f"  Tokens: {plan_response.usage.total_tokens:,} "
          f"| Input: {plan_response.usage.input_tokens:,} "
          f"| Cached: {plan_response.usage.cache_read_input_tokens:,}")
    print(f"\n{plan_response.text}\n")

    # --- Phase 2: Specialists work in parallel (all fork from same session) ---
    print("=" * 60)
    print("SPECIALIST REPORTS (parallel forks from shared session)")
    print("=" * 60)

    specialist_results: list[tuple[AgentRole, ForkResult]] = []
    results_lock: list[tuple[AgentRole, ForkResult] | None] = [None] * len(SPECIALISTS)

    async def run_specialist(idx: int, role: AgentRole) -> None:
        result = await session.fork(
            prompt=(
                f"{role.brief_instructions}\n\n"
                f"Project goal: {goal}\n\n"
                f"Coordinator's delegation plan:\n{plan_response.text}\n\n"
                f"Produce your specialist report with:\n"
                f"- 3-5 concrete recommendations (bullet points)\n"
                f"- Key risks or concerns in your area\n"
                f"- Priority ranking (what to build first)\n"
                f"- Confidence level (high/medium/low)"
            ),
            policy=ForkPolicy.cache_safe_ephemeral(),
        )
        results_lock[idx] = (role, result)

    t1 = time.time()
    await asyncio.gather(*[run_specialist(i, role) for i, role in enumerate(SPECIALISTS)])
    specialist_elapsed = time.time() - t1

    for item in results_lock:
        assert item is not None
        role, result = item
        specialist_results.append(item)
        total_usage = total_usage + result.usage
        print(f"\n{role.tag} [{role.name}]")
        print(f"  Tokens: {result.usage.total_tokens:,} "
              f"| Input: {result.usage.input_tokens:,} "
              f"| Cached: {result.usage.cache_read_input_tokens:,}")
        preview = result.final_text[:300].replace("\n", " ")
        print(f"  Preview: {preview}...")
        print("-" * 40)

    print(f"\nAll specialists done in {specialist_elapsed:.1f}s")

    # --- Phase 3: Coordinator synthesizes ---
    print(f"\n{COORDINATOR.tag} Synthesizing all reports...")

    reports_text = "\n\n---\n\n".join(
        f"## {role.name}\n\n{result.final_text}"
        for role, result in specialist_results
    )

    synthesis = await session.fork(
        prompt=(
            f"{COORDINATOR.brief_instructions}\n\n"
            f"Your specialists have delivered reports for: {goal}\n\n"
            f"Reports:\n\n{reports_text}\n\n"
            f"Synthesize into a final project plan:\n"
            f"1. Executive summary (3-4 sentences)\n"
            f"2. Unified key decisions and recommendations\n"
            f"3. Top risks and mitigations\n"
            f"4. 3-phase roadmap with milestones"
        ),
        policy=ForkPolicy.cache_safe_ephemeral(),
    )
    total_usage = total_usage + synthesis.usage
    print(f"  Synthesis tokens: {synthesis.usage.total_tokens:,} "
          f"| Input: {synthesis.usage.input_tokens:,} "
          f"| Cached: {synthesis.usage.cache_read_input_tokens:,}")

    # --- Final output ---
    print("\n" + "=" * 60)
    print("FINAL PROJECT PLAN")
    print("=" * 60)
    print(f"\nGoal: {goal}\n")
    print(synthesis.final_text)

    # --- Stats ---
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("TEAM STATS")
    print("=" * 60)
    print(f"  Team size:          {len(SPECIALISTS)} specialists + 1 coordinator")
    print(f"  LLM calls:         {1 + len(SPECIALISTS) + 1} "
          f"(1 plan + {len(SPECIALISTS)} specialist forks + 1 synthesis fork)")
    print(f"  Total input tokens: {total_usage.input_tokens:,}")
    print(f"  Total output tokens:{total_usage.output_tokens:,}")
    print(f"  Cache read tokens:  {total_usage.cache_read_input_tokens:,}")
    print(f"  Cache hit rate:     {total_usage.cache_hit_rate:.1%}")
    print(f"  Total tokens:       {total_usage.total_tokens:,}")
    print(f"  Wall time:          {elapsed:.1f}s")

    if total_usage.cache_read_input_tokens > 0:
        no_cache_input = total_usage.input_tokens + total_usage.cache_read_input_tokens
        print(f"\n  Without caching, full-price input: {no_cache_input:,}")
        print(f"  Tokens at discounted rate:         {total_usage.cache_read_input_tokens:,}")


if __name__ == "__main__":
    asyncio.run(main())
