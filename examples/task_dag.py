"""Task DAG using agentcache's built-in TaskDAG and DAGRunner.

Demonstrates dependency-aware topological scheduling with parallel execution
of independent tasks.  All tasks fork from one shared session for cache hits.

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

from dotenv import load_dotenv

load_dotenv()

from agentcache import DAGRunner, LiteLLMSDKProvider, TaskDAG

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

PRODUCT = "AI-powered meal planning app that generates personalized weekly meal plans"


def build_dag() -> TaskDAG:
    dag = TaskDAG()

    dag.add_task(
        id="market_research",
        name="Market Research",
        prompt_template=(
            "Conduct market research for: {product}\n\n"
            "Deliver:\n"
            "- Target market size and growth rate\n"
            "- Key demographics (age, income, lifestyle)\n"
            "- Willingness to pay (price sensitivity analysis)\n"
            "- 3 market trends supporting this product"
        ),
    )

    dag.add_task(
        id="competitor_scan",
        name="Competitor Scan",
        prompt_template=(
            "Analyze the competitive landscape for: {product}\n\n"
            "Deliver:\n"
            "- Top 5 competitors with their strengths/weaknesses\n"
            "- Feature comparison matrix\n"
            "- Gaps in the market (unmet needs)\n"
            "- Differentiation opportunities"
        ),
    )

    dag.add_task(
        id="user_interviews",
        name="User Interview Synthesis",
        prompt_template=(
            "Synthesize hypothetical user interview findings for: {product}\n\n"
            "Deliver:\n"
            "- 3 user personas with pain points and goals\n"
            "- Top 5 requested features (ranked by frequency)\n"
            "- Dealbreakers (what makes users abandon similar apps)\n"
            "- Surprise insights (unexpected user needs)"
        ),
    )

    dag.add_task(
        id="feature_spec",
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
    )

    dag.add_task(
        id="ux_design",
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
    )

    dag.add_task(
        id="implementation_plan",
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
    )

    dag.add_task(
        id="final_review",
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
    )

    return dag


async def main() -> None:
    if os.getenv("OPENAI_API_KEY"):
        model = "gpt-4o-mini"
    elif os.getenv("GEMINI_API_KEY"):
        model = "gemini/gemini-2.5-flash"
    else:
        print("Set OPENAI_API_KEY or GEMINI_API_KEY in .env")
        return

    provider = LiteLLMSDKProvider()
    dag = build_dag()
    runner = DAGRunner(provider)

    print(f"Model:   {model}")
    print(f"Product: {PRODUCT}")
    print(f"System prompt: ~{len(SYSTEM_PROMPT):,} chars")
    print()

    waves = dag.topological_waves()
    print("DAG schedule:")
    for i, wave in enumerate(waves):
        tasks_str = ", ".join(dag.get(t).name for t in wave)
        print(f"  Wave {i+1}: {tasks_str}" + (" (parallel)" if len(wave) > 1 else ""))
    print()

    t0 = time.time()
    result = await runner.run(
        dag,
        model=model,
        system_prompt=SYSTEM_PROMPT,
        context_vars={"product": PRODUCT},
    )
    total_elapsed = time.time() - t0

    print("=" * 60)
    print("TASK RESULTS")
    print("=" * 60)

    for wave in result.waves:
        for tid in wave:
            task = result.tasks[tid]
            print(f"\n  [{task.name}] ({task.elapsed:.1f}s)")
            print(f"    Tokens: {task.usage.total_tokens:,} "
                  f"| Cached: {task.usage.cache_read_input_tokens:,}")
            preview = task.result[:200].replace("\n", " ")
            print(f"    Preview: {preview}...")

    print("\n" + "=" * 60)
    print("FINAL REVIEW")
    print("=" * 60)
    print(f"\n{result.task_result('final_review')}")

    print("\n" + "=" * 60)
    print("DAG EXECUTION STATS")
    print("=" * 60)
    print(f"  Tasks:              {len(result.tasks)}")
    print(f"  Waves:              {len(result.waves)}")
    print(f"  Parallelized:       {result.parallelized_count}/{len(result.tasks)} tasks")
    print(f"  Total input tokens: {result.usage.input_tokens:,}")
    print(f"  Total output tokens:{result.usage.output_tokens:,}")
    print(f"  Cache read tokens:  {result.usage.cache_read_input_tokens:,}")
    print(f"  Cache hit rate:     {result.usage.cache_hit_rate:.1%}")
    print(f"  Total tokens:       {result.usage.total_tokens:,}")
    print(f"  Wall time:          {total_elapsed:.1f}s")
    print(f"  DAG execution:      {result.elapsed:.1f}s")

    sequential_time = sum(t.elapsed for t in result.tasks.values())
    if sequential_time > 0 and result.elapsed > 0:
        speedup = sequential_time / result.elapsed
        print(f"\n  Sequential would take: {sequential_time:.1f}s")
        print(f"  DAG speedup:           {speedup:.2f}x")


if __name__ == "__main__":
    asyncio.run(main())
