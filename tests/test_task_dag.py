"""Tests for TaskDAG: topological sorting, validation, cycle detection."""

from __future__ import annotations

import pytest

from agentcache.core.errors import AgentCacheError
from agentcache.dag.task import Task, TaskDAG, TaskStatus


class TestTaskDAG:
    def test_add_and_retrieve_task(self):
        dag = TaskDAG()
        t = dag.add_task(id="a", name="Task A", prompt_template="Do {thing}")
        assert dag.get("a") is t
        assert t.status == TaskStatus.PENDING

    def test_topological_waves_linear(self):
        dag = TaskDAG()
        dag.add_task(id="a", name="A", prompt_template="...")
        dag.add_task(id="b", name="B", prompt_template="...", depends_on=["a"])
        dag.add_task(id="c", name="C", prompt_template="...", depends_on=["b"])

        waves = dag.topological_waves()
        assert waves == [["a"], ["b"], ["c"]]

    def test_topological_waves_parallel(self):
        dag = TaskDAG()
        dag.add_task(id="a", name="A", prompt_template="...")
        dag.add_task(id="b", name="B", prompt_template="...")
        dag.add_task(id="c", name="C", prompt_template="...", depends_on=["a", "b"])

        waves = dag.topological_waves()
        assert waves[0] == ["a", "b"]
        assert waves[1] == ["c"]

    def test_topological_waves_diamond(self):
        dag = TaskDAG()
        dag.add_task(id="start", name="Start", prompt_template="...")
        dag.add_task(id="left", name="Left", prompt_template="...", depends_on=["start"])
        dag.add_task(id="right", name="Right", prompt_template="...", depends_on=["start"])
        dag.add_task(id="end", name="End", prompt_template="...", depends_on=["left", "right"])

        waves = dag.topological_waves()
        assert len(waves) == 3
        assert waves[0] == ["start"]
        assert waves[1] == ["left", "right"]
        assert waves[2] == ["end"]

    def test_validate_missing_dependency(self):
        dag = TaskDAG()
        dag.add_task(id="a", name="A", prompt_template="...", depends_on=["nonexistent"])

        with pytest.raises(AgentCacheError, match="unknown task 'nonexistent'"):
            dag.validate()

    def test_validate_cycle(self):
        dag = TaskDAG()
        dag.add_task(id="a", name="A", prompt_template="...", depends_on=["b"])
        dag.add_task(id="b", name="B", prompt_template="...", depends_on=["a"])

        with pytest.raises(AgentCacheError, match="Cycle detected"):
            dag.validate()

    def test_validate_self_cycle(self):
        dag = TaskDAG()
        dag.add_task(id="a", name="A", prompt_template="...", depends_on=["a"])

        with pytest.raises(AgentCacheError, match="Cycle detected"):
            dag.validate()

    def test_empty_dag(self):
        dag = TaskDAG()
        waves = dag.topological_waves()
        assert waves == []

    def test_single_task(self):
        dag = TaskDAG()
        dag.add_task(id="solo", name="Solo", prompt_template="Do it")
        waves = dag.topological_waves()
        assert waves == [["solo"]]
