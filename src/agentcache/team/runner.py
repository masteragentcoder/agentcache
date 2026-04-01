"""TeamRunner: plan -> parallel specialist forks -> synthesis, all cache-safe."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from agentcache.core.session import AgentSession
from agentcache.core.usage import Usage
from agentcache.fork.policies import ForkPolicy
from agentcache.fork.result import ForkResult
from agentcache.providers.base import Provider
from agentcache.team.config import AgentRole, TeamConfig


@dataclass
class SpecialistReport:
    role: AgentRole
    result: ForkResult

    @property
    def text(self) -> str:
        return self.result.final_text


@dataclass
class TeamResult:
    plan_text: str = ""
    specialist_reports: list[SpecialistReport] = field(default_factory=list)
    synthesis: ForkResult | None = None
    usage: Usage = field(default_factory=Usage)

    @property
    def final_text(self) -> str:
        if self.synthesis:
            return self.synthesis.final_text
        return ""


class TeamRunner:
    """Runs a team of specialists from a single cache-safe session.

    All specialists fork from one shared ``AgentSession`` so every LLM call
    after the first reuses the cached prefix.
    """

    def __init__(self, provider: Provider, config: TeamConfig) -> None:
        self.provider = provider
        self.config = config

    async def run(
        self,
        goal: str,
        *,
        model: str,
        plan_prompt: str | None = None,
        synthesis_prompt: str | None = None,
    ) -> TeamResult:
        session = AgentSession(
            model=model,
            provider=self.provider,
            system_prompt=self.config.system_prompt,
        )
        result = TeamResult()

        plan_text = await self._plan(session, goal, plan_prompt)
        result.plan_text = plan_text
        result.usage = result.usage + session.last_usage  # type: ignore[operator]

        reports = await self._run_specialists(session, goal, plan_text)
        result.specialist_reports = reports
        for r in reports:
            result.usage = result.usage + r.result.usage

        synthesis = await self._synthesize(session, goal, reports, synthesis_prompt)
        result.synthesis = synthesis
        result.usage = result.usage + synthesis.usage

        return result

    async def _plan(
        self,
        session: AgentSession,
        goal: str,
        custom_prompt: str | None,
    ) -> str:
        if custom_prompt:
            prompt = custom_prompt
        else:
            role_names = ", ".join(self.config.role_names())
            prompt = (
                f"{self.config.coordinator_instructions}\n\n"
                f"You have these specialists: {role_names}.\n\n"
                f"Project goal: {goal}\n\n"
                f"For each specialist, write a focused one-paragraph brief "
                f"describing exactly what they should investigate and deliver."
            )
        response = await session.respond(prompt)
        return response.text

    async def _run_specialists(
        self,
        session: AgentSession,
        goal: str,
        plan_text: str,
    ) -> list[SpecialistReport]:
        reports: list[SpecialistReport | None] = [None] * len(self.config.roles)

        async def _run_one(idx: int, role: AgentRole) -> None:
            fork_result = await session.fork(
                prompt=(
                    f"{role.instructions}\n\n"
                    f"Project goal: {goal}\n\n"
                    f"Coordinator's delegation plan:\n{plan_text}\n\n"
                    f"Produce your specialist report with:\n"
                    f"- 3-5 concrete recommendations (bullet points)\n"
                    f"- Key risks or concerns in your area\n"
                    f"- Priority ranking (what to build first)\n"
                    f"- Confidence level (high/medium/low)"
                ),
                policy=ForkPolicy.cache_safe_ephemeral(),
            )
            reports[idx] = SpecialistReport(role=role, result=fork_result)

        await asyncio.gather(
            *[_run_one(i, role) for i, role in enumerate(self.config.roles)]
        )
        return [r for r in reports if r is not None]

    async def _synthesize(
        self,
        session: AgentSession,
        goal: str,
        reports: list[SpecialistReport],
        custom_prompt: str | None,
    ) -> ForkResult:
        reports_text = "\n\n---\n\n".join(
            f"## {r.role.name}\n\n{r.text}" for r in reports
        )
        if custom_prompt:
            prompt = custom_prompt.format(
                goal=goal, reports=reports_text,
            )
        else:
            prompt = (
                f"{self.config.coordinator_instructions}\n\n"
                f"Your specialists have delivered reports for: {goal}\n\n"
                f"Reports:\n\n{reports_text}\n\n"
                f"Synthesize into a final project plan:\n"
                f"1. Executive summary (3-4 sentences)\n"
                f"2. Unified key decisions and recommendations\n"
                f"3. Top risks and mitigations\n"
                f"4. 3-phase roadmap with milestones"
            )
        return await session.fork(
            prompt=prompt,
            policy=ForkPolicy.cache_safe_ephemeral(),
        )
