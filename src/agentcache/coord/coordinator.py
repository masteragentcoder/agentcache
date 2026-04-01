"""Coordinator: leader/worker orchestration with summarized handoffs.

Delegates to ``TeamRunner`` for the plan -> specialists -> synthesis flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agentcache.coord.worker import WorkerReport, WorkerSpec
from agentcache.team.config import AgentRole, TeamConfig
from agentcache.team.runner import TeamResult, TeamRunner

if TYPE_CHECKING:
    from agentcache.providers.base import Provider


@dataclass
class CoordinatorResult:
    reports: list[WorkerReport] = field(default_factory=list)
    synthesis: str = ""
    team_result: TeamResult | None = None


class Coordinator:
    def __init__(self, provider: Provider) -> None:
        self.provider = provider

    async def run(
        self,
        task: str,
        workers: list[WorkerSpec],
        *,
        model: str,
        system_prompt: str,
    ) -> CoordinatorResult:
        roles = [
            AgentRole(
                name=w.name,
                instructions=w.instruction,
                purpose="coord_worker",
            )
            for w in workers
        ]
        config = TeamConfig(system_prompt=system_prompt, roles=roles)
        runner = TeamRunner(self.provider, config)

        team_result = await runner.run(task, model=model)

        reports = [
            WorkerReport(
                worker_id=sr.result.agent_id,
                summary=sr.text,
            )
            for sr in team_result.specialist_reports
        ]

        return CoordinatorResult(
            reports=reports,
            synthesis=team_result.final_text,
            team_result=team_result,
        )
