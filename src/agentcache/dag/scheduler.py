"""DAGRunner: execute a TaskDAG through wave-parallel cache-safe forks."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from agentcache.core.session import AgentSession
from agentcache.core.usage import Usage
from agentcache.dag.task import Task, TaskDAG, TaskStatus
from agentcache.fork.policies import ForkPolicy
from agentcache.providers.base import Provider


@dataclass
class DAGResult:
    tasks: dict[str, Task] = field(default_factory=dict)
    waves: list[list[str]] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    elapsed: float = 0.0

    @property
    def parallelized_count(self) -> int:
        return sum(len(w) for w in self.waves if len(w) > 1)

    def task_result(self, task_id: str) -> str:
        return self.tasks[task_id].result


class DAGRunner:
    """Execute a task DAG with topological wave scheduling.

    All tasks fork from a single ``AgentSession``, sharing its cached prefix.
    Independent tasks within a wave run in parallel via ``asyncio.gather``.
    Upstream results are injected into downstream prompt templates.
    """

    def __init__(self, provider: Provider) -> None:
        self.provider = provider

    async def run(
        self,
        dag: TaskDAG,
        *,
        model: str,
        system_prompt: str,
        context_vars: dict[str, str] | None = None,
    ) -> DAGResult:
        session = AgentSession(
            model=model,
            provider=self.provider,
            system_prompt=system_prompt,
        )

        warmup = await session.respond("Acknowledged. Ready for tasks.")
        total_usage = warmup.usage

        waves = dag.topological_waves()
        t0 = time.time()
        completed: dict[str, Task] = {}

        for wave in waves:
            async def _run_task(tid: str) -> None:
                task = dag.get(tid)
                task.status = TaskStatus.RUNNING

                fmt: dict[str, str] = dict(context_vars or {})
                for cid, ctask in completed.items():
                    fmt[cid] = ctask.result

                prompt = task.prompt_template.format(**fmt)
                task_t0 = time.time()

                result = await session.fork(
                    prompt=prompt,
                    policy=ForkPolicy.cache_safe_ephemeral(),
                )

                task.elapsed = time.time() - task_t0
                task.result = result.final_text
                task.usage = result.usage
                task.status = TaskStatus.COMPLETED

            await asyncio.gather(*[_run_task(tid) for tid in wave])

            for tid in wave:
                completed[tid] = dag.get(tid)
                total_usage = total_usage + dag.get(tid).usage

        return DAGResult(
            tasks=dict(dag.tasks),
            waves=waves,
            usage=total_usage,
            elapsed=time.time() - t0,
        )
