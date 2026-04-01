"""Tests for DAGRunner with fake provider."""

from __future__ import annotations

import pytest

from agentcache.dag.scheduler import DAGResult, DAGRunner
from agentcache.dag.task import TaskDAG, TaskStatus

from .conftest import FakeProvider


class TestDAGRunner:
    @pytest.mark.asyncio
    async def test_linear_dag(self):
        responses = [
            "Acknowledged.",
            "Research result.",
            "Design based on research.",
            "Plan based on design.",
        ]
        provider = FakeProvider(responses)

        dag = TaskDAG()
        dag.add_task(id="research", name="Research", prompt_template="Research {product}")
        dag.add_task(
            id="design",
            name="Design",
            prompt_template="Design using:\n{research}",
            depends_on=["research"],
        )
        dag.add_task(
            id="plan",
            name="Plan",
            prompt_template="Plan using:\n{design}",
            depends_on=["design"],
        )

        runner = DAGRunner(provider)
        result = await runner.run(
            dag,
            model="test-model",
            system_prompt="You are a planner.",
            context_vars={"product": "test app"},
        )

        assert isinstance(result, DAGResult)
        assert len(result.tasks) == 3
        assert len(result.waves) == 3
        assert all(t.status == TaskStatus.COMPLETED for t in result.tasks.values())
        assert result.usage.input_tokens > 0
        assert result.elapsed >= 0

    @pytest.mark.asyncio
    async def test_parallel_wave(self):
        responses = ["Acknowledged.", "A result.", "B result.", "C result."]
        provider = FakeProvider(responses)

        dag = TaskDAG()
        dag.add_task(id="a", name="A", prompt_template="Do A")
        dag.add_task(id="b", name="B", prompt_template="Do B")
        dag.add_task(
            id="c",
            name="C",
            prompt_template="Combine:\n{a}\n{b}",
            depends_on=["a", "b"],
        )

        runner = DAGRunner(provider)
        result = await runner.run(
            dag,
            model="test-model",
            system_prompt="test",
        )

        assert result.waves[0] == ["a", "b"]
        assert result.waves[1] == ["c"]
        assert result.parallelized_count == 2

    @pytest.mark.asyncio
    async def test_upstream_injection(self):
        responses = ["Acknowledged.", "First output.", "Got: First output."]
        provider = FakeProvider(responses)

        dag = TaskDAG()
        dag.add_task(id="first", name="First", prompt_template="Produce output")
        dag.add_task(
            id="second",
            name="Second",
            prompt_template="Process: {first}",
            depends_on=["first"],
        )

        runner = DAGRunner(provider)
        result = await runner.run(
            dag, model="test-model", system_prompt="test"
        )

        assert result.tasks["first"].result == "First output."
        assert result.tasks["second"].status == TaskStatus.COMPLETED
