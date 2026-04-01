"""Task DAG: topological scheduling with parallel execution of independent tasks.

Demonstrates that agentcache sessions + asyncio can implement dependency-aware
task scheduling without a built-in DAG engine.  Independent tasks run in
parallel (cache-safe forks), dependent tasks wait for their prerequisites.

The DAG for this example (a product launch plan):

    market_research ──┐
                      ├──> feature_spec ──┐
    competitor_scan ──┘                   ├──> implementation_plan ──> final_review
                      ┌──> ux_design ─────┘
    user_interviews ──┘

Usage:
    python examples/task_dag.py
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

from agentcache import AgentSession, ForkPolicy, LiteLLMSDKProvider, Usage

PADDING = "".join(
    f"Planning principle #{i}: Each task produces a structured deliverable. "
    f"Downstream tasks consume only the deliverable, not the raw process. "
    f"Keep outputs concise and actionable. (principle {i})\n"
    for i in range(25)
)

SYSTEM_PROMPT = (
    "You are a product planning assistant. You produce concise, structured "
    "deliverables for each phase of product development. Use bullet points, "
    "be specific, and stay within scope of the task assigned to you.\n\n"
    + PADDING
)


@dataclass
class Task:
    name: str
    prompt_template: str
    depends_on: list[str] = field(default_factory=list)
    result: str = ""
    usage: Usage = field(default_factory=Usage)
    elapsed: float = 0.0


PRODUCT = "AI-powered meal planning app that generates personalized weekly meal plans"

TASKS: dict[str, Task] = {
    "market_research": Task(
        name="Market Research",
        prompt_template=(
            "Conduct market research for: {product}\n\n"
            "Deliver:\n"
            "- Target market size and growth rate\n"
            "- Key demographics (age, income, lifestyle)\n"
            "- Willingness to pay (price sensitivity analysis)\n"
            "- 3 market trends supporting this product"
        ),
    ),
    "competitor_scan": Task(
        name="Competitor Scan",
        prompt_template=(
            "Analyze the competitive landscape for: {product}\n\n"
            "Deliver:\n"
            "- Top 5 competitors with their strengths/weaknesses\n"
            "- Feature comparison matrix\n"
            "- Gaps in the market (unmet needs)\n"
            "- Differentiation opportunities"
        ),
    ),
    "user_interviews": Task(
        name="User Interview Synthesis",
        prompt_template=(
            "Synthesize hypothetical user interview findings for: {product}\n\n"
            "Deliver:\n"
            "- 3 user personas with pain points and goals\n"
            "- Top 5 requested features (ranked by frequency)\n"
            "- Dealbreakers (what makes users abandon similar apps)\n"
            "- Surprise insights (unexpected user needs)"
        ),
    ),
    "feature_spec": Task(
        name="Feature Specification",
        prompt_template=(
            "Write a feature specification for: {product}\n\n"
            "Use these inputs from prior tasks:\n\n"
            "Market Research:\n{market_research}\n\n"
            "Competitor Scan:\n{competitor_scan}\n\n"
            "Deliver:\n"
            "- MVP feature list (must-have for launch)\n"
            "- V2 feature list (post-launch)\n"
            "- Feature prioritization rationale\n"
            "- Technical feasibility notes"
        ),
        depends_on=["market_research", "competitor_scan"],
    ),
    "ux_design": Task(
        name="UX Design Brief",
        prompt_template=(
            "Create a UX design brief for: {product}\n\n"
            "Use these inputs from prior tasks:\n\n"
            "User Interview Synthesis:\n{user_interviews}\n\n"
            "Deliver:\n"
            "- Core user flows (3-5 key journeys)\n"
            "- Information architecture (screen hierarchy)\n"
            "- Key interaction patterns\n"
            "- Accessibility requirements"
        ),
        depends_on=["user_interviews"],
    ),
    "implementation_plan": Task(
        name="Implementation Plan",
        prompt_template=(
            "Create an implementation plan for: {product}\n\n"
            "Use these inputs from prior tasks:\n\n"
            "Feature Spec:\n{feature_spec}\n\n"
            "UX Design Brief:\n{ux_design}\n\n"
            "Deliver:\n"
            "- Tech stack recommendation with justification\n"
            "- 3-sprint breakdown (2 weeks each)\n"
            "- Team composition needed\n"
            "- Key technical risks and mitigations"
        ),
        depends_on=["feature_spec", "ux_design"],
    ),
    "final_review": Task(
        name="Final Review & Go/No-Go",
        prompt_template=(
            "Produce a final go/no-go recommendation for: {product}\n\n"
            "Use these inputs from all prior tasks:\n\n"
            "Market Research:\n{market_research}\n\n"
            "Competitor Scan:\n{competitor_scan}\n\n"
            "Implementation Plan:\n{implementation_plan}\n\n"
            "Deliver:\n"
            "- Executive summary (3 sentences)\n"
            "- GO / NO-GO recommendation with confidence level\n"
            "- Top 3 risks\n"
            "- Recommended next steps"
        ),
        depends_on=["implementation_plan"],
    ),
}


def topological_sort(tasks: dict[str, Task]) -> list[list[str]]:
    """Return tasks grouped into waves; tasks within a wave are independent."""
    in_degree: dict[str, int] = {name: 0 for name in tasks}
    dependents: dict[str, list[str]] = {name: [] for name in tasks}

    for name, task in tasks.items():
        for dep in task.depends_on:
            dependents[dep].append(name)
            in_degree[name] += 1

    waves: list[list[str]] = []
    ready = [name for name, deg in in_degree.items() if deg == 0]

    while ready:
        waves.append(sorted(ready))
        next_ready: list[str] = []
        for name in ready:
            for dependent in dependents[name]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    next_ready.append(dependent)
        ready = next_ready

    return waves


async def run_task(
    task_id: str,
    task: Task,
    session: AgentSession,
    completed: dict[str, Task],
) -> None:
    """Execute a single task as a cache-safe fork."""
    format_vars: dict[str, str] = {"product": PRODUCT}
    for cid, ctask in completed.items():
        format_vars[cid] = ctask.result

    prompt = task.prompt_template.format(**format_vars)

    t0 = time.time()
    result = await session.fork(
        prompt=prompt,
        policy=ForkPolicy.cache_safe_ephemeral(),
    )
    task.elapsed = time.time() - t0
    task.result = result.final_text
    task.usage = result.usage


async def main() -> None:
    if os.getenv("OPENAI_API_KEY"):
        model = "gpt-4o-mini"
    elif os.getenv("GEMINI_API_KEY"):
        model = "gemini/gemini-2.5-flash"
    else:
        print("Set GEMINI_API_KEY or OPENAI_API_KEY in .env")
        return

    provider = LiteLLMSDKProvider()
    session = AgentSession(
        model=model,
        provider=provider,
        system_prompt=SYSTEM_PROMPT,
    )

    print(f"Model:   {model}")
    print(f"Product: {PRODUCT}")
    print(f"System prompt: ~{len(SYSTEM_PROMPT):,} chars")
    print()

    waves = topological_sort(TASKS)
    print("DAG schedule:")
    for i, wave in enumerate(waves):
        tasks_str = ", ".join(f"{TASKS[t].name}" for t in wave)
        print(f"  Wave {i+1}: {tasks_str}" + (" (parallel)" if len(wave) > 1 else ""))
    print()

    # Warm up the session so forks have a cached prefix
    warmup = await session.respond(
        f"We are planning a product: {PRODUCT}. Acknowledge briefly."
    )
    total_usage = warmup.usage

    print("=" * 60)
    print("EXECUTING DAG")
    print("=" * 60)

    t0 = time.time()
    completed: dict[str, Task] = {}

    for wave_idx, wave in enumerate(waves):
        wave_names = ", ".join(TASKS[t].name for t in wave)
        parallel_tag = f" ({len(wave)} parallel)" if len(wave) > 1 else ""
        print(f"\n--- Wave {wave_idx + 1}{parallel_tag}: {wave_names} ---")

        async def _run(tid: str) -> None:
            await run_task(tid, TASKS[tid], session, completed)

        await asyncio.gather(*[_run(tid) for tid in wave])

        for tid in wave:
            task = TASKS[tid]
            completed[tid] = task
            total_usage = total_usage + task.usage
            print(f"\n  [{task.name}] ({task.elapsed:.1f}s)")
            print(f"    Tokens: {task.usage.total_tokens:,} "
                  f"| Cached: {task.usage.cache_read_input_tokens:,}")
            preview = task.result[:200].replace("\n", " ")
            print(f"    Preview: {preview}...")

    elapsed = time.time() - t0

    # --- Final review output ---
    print("\n" + "=" * 60)
    print("FINAL REVIEW")
    print("=" * 60)
    print(f"\n{TASKS['final_review'].result}")

    # --- Stats ---
    print("\n" + "=" * 60)
    print("DAG EXECUTION STATS")
    print("=" * 60)
    print(f"  Tasks:              {len(TASKS)}")
    print(f"  Waves:              {len(waves)}")
    parallel_tasks = sum(len(w) for w in waves if len(w) > 1)
    print(f"  Parallelized:       {parallel_tasks}/{len(TASKS)} tasks")
    print(f"  Total input tokens: {total_usage.input_tokens:,}")
    print(f"  Total output tokens:{total_usage.output_tokens:,}")
    print(f"  Cache read tokens:  {total_usage.cache_read_input_tokens:,}")
    print(f"  Cache hit rate:     {total_usage.cache_hit_rate:.1%}")
    print(f"  Total tokens:       {total_usage.total_tokens:,}")
    print(f"  Wall time:          {elapsed:.1f}s")

    sequential_time = sum(t.elapsed for t in TASKS.values())
    if sequential_time > 0:
        speedup = sequential_time / elapsed
        print(f"\n  Sequential would take: {sequential_time:.1f}s")
        print(f"  DAG speedup:           {speedup:.2f}x")


if __name__ == "__main__":
    asyncio.run(main())
