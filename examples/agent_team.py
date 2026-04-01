"""Multi-agent team using agentcache's built-in TeamRunner.

All specialists fork from ONE shared session so they share the same cached
prefix.  Role differentiation happens in the fork prompt, not the system
prompt -- this is the key insight for getting cache hits.

Usage:
    python examples/agent_team.py "Design a mobile app for tracking personal finances"
    python examples/agent_team.py  # uses default goal
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

from agentcache import AgentRole, LiteLLMSDKProvider, TeamConfig, TeamRunner

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

CONFIG = TeamConfig(
    system_prompt=SYSTEM_PROMPT,
    roles=[
        AgentRole(
            name="Technical Architect",
            instructions=(
                "You are acting as the Technical Architect. Focus on system "
                "architecture, tech stack, data models, security, and scalability. "
                "Name actual technologies and trade-offs."
            ),
        ),
        AgentRole(
            name="UX Designer",
            instructions=(
                "You are acting as the UX Designer. Focus on user flows, "
                "information architecture, interaction patterns, and accessibility. "
                "Reference real-world apps as examples."
            ),
        ),
        AgentRole(
            name="Business Analyst",
            instructions=(
                "You are acting as the Business Analyst. Focus on market analysis, "
                "competitive landscape, success metrics, and monetization. Use "
                "specific numbers and benchmarks."
            ),
        ),
    ],
)

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
    runner = TeamRunner(provider, CONFIG)

    print(f"Model: {model}")
    print(f"Goal:  {goal}")
    print(f"System prompt: ~{len(SYSTEM_PROMPT):,} chars")
    print(f"Team:  {', '.join(CONFIG.role_names())}")
    print("=" * 60)

    t0 = time.time()
    result = await runner.run(goal, model=model)
    elapsed = time.time() - t0

    print(f"\n[COORD] Plan:\n{result.plan_text}\n")

    for report in result.specialist_reports:
        print(f"[{report.role.name}]")
        print(f"  Tokens: {report.result.usage.total_tokens:,} "
              f"| Input: {report.result.usage.input_tokens:,} "
              f"| Cached: {report.result.usage.cache_read_input_tokens:,}")
        preview = report.text[:200].replace("\n", " ")
        print(f"  Preview: {preview}...")
        print()

    print("=" * 60)
    print("FINAL PROJECT PLAN")
    print("=" * 60)
    print(f"\n{result.final_text}")

    print("\n" + "=" * 60)
    print("TEAM STATS")
    print("=" * 60)
    print(f"  Team size:          {len(CONFIG.roles)} specialists + 1 coordinator")
    print(f"  Total input tokens: {result.usage.input_tokens:,}")
    print(f"  Total output tokens:{result.usage.output_tokens:,}")
    print(f"  Cache read tokens:  {result.usage.cache_read_input_tokens:,}")
    print(f"  Cache hit rate:     {result.usage.cache_hit_rate:.1%}")
    print(f"  Total tokens:       {result.usage.total_tokens:,}")
    print(f"  Wall time:          {elapsed:.1f}s")

    if result.usage.cache_read_input_tokens > 0:
        no_cache = result.usage.input_tokens + result.usage.cache_read_input_tokens
        print(f"\n  Without caching, full-price input: {no_cache:,}")
        print(f"  Tokens at discounted rate:         {result.usage.cache_read_input_tokens:,}")


if __name__ == "__main__":
    asyncio.run(main())
