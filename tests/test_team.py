"""Tests for TeamConfig and TeamRunner with fake provider."""

from __future__ import annotations

import pytest

from agentcache.team.config import AgentRole, TeamConfig
from agentcache.team.runner import SpecialistReport, TeamResult, TeamRunner

from .conftest import FakeProvider


class TestTeamConfig:
    def test_role_names(self):
        config = TeamConfig(
            system_prompt="test",
            roles=[
                AgentRole(name="Architect", instructions="design stuff"),
                AgentRole(name="Analyst", instructions="analyze stuff"),
            ],
        )
        assert config.role_names() == ["Architect", "Analyst"]

    def test_empty_roles(self):
        config = TeamConfig(system_prompt="test")
        assert config.role_names() == []


class TestTeamRunner:
    @pytest.mark.asyncio
    async def test_full_run(self):
        responses = [
            "Plan: Architect should do X, Analyst should do Y.",
            "Architect report here.",
            "Analyst report here.",
            "Synthesis: everything looks good.",
        ]
        provider = FakeProvider(responses)

        config = TeamConfig(
            system_prompt="You are a team assistant.",
            roles=[
                AgentRole(name="Architect", instructions="Design systems."),
                AgentRole(name="Analyst", instructions="Analyze markets."),
            ],
        )
        runner = TeamRunner(provider, config)
        result = await runner.run("Build a product", model="test-model")

        assert isinstance(result, TeamResult)
        assert "Plan:" in result.plan_text
        assert len(result.specialist_reports) == 2
        assert all(isinstance(r, SpecialistReport) for r in result.specialist_reports)
        assert result.final_text == "Synthesis: everything looks good."
        assert result.usage.input_tokens > 0
        assert result.usage.total_tokens > 0

    @pytest.mark.asyncio
    async def test_run_with_no_specialists(self):
        responses = ["Plan for nobody.", "Synthesis."]
        provider = FakeProvider(responses)

        config = TeamConfig(system_prompt="test")
        runner = TeamRunner(provider, config)
        result = await runner.run("Do something", model="test-model")

        assert result.plan_text == "Plan for nobody."
        assert len(result.specialist_reports) == 0
        assert result.final_text == "Synthesis."
