#!/usr/bin/env python3
"""Run agentcache deep research as a CLI tool.

Usage:
    python run_research.py "Topic to research" [--output path/to/report.md]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys

from dotenv import load_dotenv

load_dotenv()

from agentcache import AgentSession, ForkPolicy, ForkResult, LiteLLMSDKProvider

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


async def main() -> None:
    parser = argparse.ArgumentParser(description="agentcache deep research tool")
    parser.add_argument("topic", help="The topic to research")
    parser.add_argument(
        "--output", "-o", default="research_report.md", help="Output file path"
    )
    args = parser.parse_args()

    if os.getenv("GEMINI_API_KEY"):
        model = "gemini/gemini-2.5-flash"
    elif os.getenv("OPENAI_API_KEY"):
        model = "gpt-4o-mini"
    else:
        print("Error: Set GEMINI_API_KEY or OPENAI_API_KEY in .env", file=sys.stderr)
        sys.exit(1)

    provider = LiteLLMSDKProvider()
    session = AgentSession(
        model=model,
        provider=provider,
        system_prompt=SYSTEM_PROMPT,
    )

    print(f"Topic: {args.topic}")
    print("Planning research angles...", file=sys.stderr)

    plan_response = await session.respond(
        f"Break this research question into 3-4 distinct investigation angles. "
        f"For each angle, write one line: the angle name and a one-sentence scope.\n\n"
        f"Question: {args.topic}\n\n"
        f"Format each angle exactly as: '1. Name: scope'"
    )

    angles = _parse_angles(plan_response.text)
    if not angles:
        print("Error: Could not parse angles from plan.", file=sys.stderr)
        sys.exit(1)

    print(f"Launching {len(angles)} parallel workers...", file=sys.stderr)
    results: list[ForkResult] = [None] * len(angles)  # type: ignore[list-item]

    async def run_worker(idx: int, angle: str) -> None:
        prompt = (
            f"Investigate this specific angle of the research question.\n\n"
            f"Original question: {args.topic}\n"
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
        print(f"  Worker finished: {angle[:40]}...", file=sys.stderr)

    await asyncio.gather(*(run_worker(i, angle) for i, angle in enumerate(angles)))

    print("Synthesizing final report...", file=sys.stderr)
    worker_reports = "\n\n---\n\n".join(
        f"## Angle {i+1}: {angle}\n\n{result.final_text}"
        for i, (angle, result) in enumerate(zip(angles, results))
    )

    synthesis = await session.fork(
        prompt=(
            f"You have received reports from {len(angles)} research workers "
            f"investigating different angles of this question:\n\n"
            f"Question: {args.topic}\n\n"
            f"Worker reports:\n\n{worker_reports}\n\n"
            f"Synthesize these into a final research brief:\n"
            f"1. Executive summary (3-4 sentences)\n"
            f"2. Key findings across all angles\n"
            f"3. Emerging consensus and open debates\n"
            f"4. Recommended next steps for deeper investigation"
        ),
        policy=ForkPolicy.cache_safe_ephemeral(),
    )

    report_content = (
        f"# Research Report: {args.topic}\n\n"
        f"{synthesis.final_text}\n\n"
        f"## Worker Summaries\n\n"
        f"<details><summary>Click to view raw worker reports</summary>\n\n"
        f"{worker_reports}\n\n"
        f"</details>\n"
    )

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report_content)

    status = session.cache_status()
    print(f"\nReport written to {args.output}", file=sys.stderr)
    print(f"Cache hit rate: {status.hit_rate_recent:.1%}", file=sys.stderr)


def _parse_angles(text: str) -> list[str]:
    lines = text.strip().splitlines()
    angles = []
    for line in lines:
        match = re.match(r"^\s*\d+[\.\)]\s*(.+)", line.strip())
        if match:
            angles.append(match.group(1).strip())
    return angles


if __name__ == "__main__":
    asyncio.run(main())
