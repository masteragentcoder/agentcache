"""Coordinator: leader/worker orchestration with summarized handoffs. (Milestone 6)

TODO: Full implementation deferred until fork/cache/memory substrate is stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agentcache.coord.worker import WorkerReport, WorkerSpec

if TYPE_CHECKING:
    from agentcache.core.session import AgentSession
    from agentcache.fork.runner import ForkRunner


@dataclass
class CoordinatorResult:
    reports: list[WorkerReport] = field(default_factory=list)
    synthesis: str = ""


class Coordinator:
    def __init__(self, fork_runner: ForkRunner) -> None:
        self.fork_runner = fork_runner

    async def run(
        self,
        session: AgentSession,
        task: str,
        workers: list[WorkerSpec],
    ) -> CoordinatorResult:
        # TODO(milestone-6): spawn workers, gather reports, synthesize
        raise NotImplementedError(
            "Coordinator is planned for Milestone 6. "
            "Fork/cache/memory substrate must be stable first."
        )
