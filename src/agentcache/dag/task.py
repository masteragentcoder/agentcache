"""Task DAG: dependency graph with topological wave scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from agentcache.core.errors import AgentCacheError
from agentcache.core.usage import Usage


class TaskStatus(str, Enum):
    PENDING = "pending"
    BLOCKED = "blocked"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    id: str
    name: str
    prompt_template: str
    depends_on: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    usage: Usage = field(default_factory=Usage)
    elapsed: float = 0.0


class TaskDAG:
    """Mutable DAG of tasks with dependency edges."""

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}

    def add_task(
        self,
        id: str,
        name: str,
        prompt_template: str,
        depends_on: list[str] | None = None,
    ) -> Task:
        task = Task(
            id=id,
            name=name,
            prompt_template=prompt_template,
            depends_on=depends_on or [],
        )
        self._tasks[id] = task
        return task

    def get(self, task_id: str) -> Task:
        return self._tasks[task_id]

    @property
    def tasks(self) -> dict[str, Task]:
        return self._tasks

    def validate(self) -> None:
        """Check for missing dependencies and cycles."""
        ids = set(self._tasks)
        for task in self._tasks.values():
            for dep in task.depends_on:
                if dep not in ids:
                    raise AgentCacheError(
                        f"Task '{task.id}' depends on unknown task '{dep}'"
                    )
        self._detect_cycle()

    def _detect_cycle(self) -> None:
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {tid: WHITE for tid in self._tasks}

        def dfs(tid: str) -> None:
            color[tid] = GRAY
            for dep in self._tasks[tid].depends_on:
                if color[dep] == GRAY:
                    raise AgentCacheError(
                        f"Cycle detected in task DAG involving '{dep}'"
                    )
                if color[dep] == WHITE:
                    dfs(dep)
            color[tid] = BLACK

        for tid in self._tasks:
            if color[tid] == WHITE:
                dfs(tid)

    def topological_waves(self) -> list[list[str]]:
        """Group tasks into waves; tasks within a wave are independent."""
        self.validate()

        in_degree: dict[str, int] = {tid: 0 for tid in self._tasks}
        dependents: dict[str, list[str]] = {tid: [] for tid in self._tasks}

        for tid, task in self._tasks.items():
            for dep in task.depends_on:
                dependents[dep].append(tid)
                in_degree[tid] += 1

        waves: list[list[str]] = []
        ready = sorted(tid for tid, deg in in_degree.items() if deg == 0)

        while ready:
            waves.append(ready)
            next_ready: list[str] = []
            for tid in ready:
                for dependent in dependents[tid]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        next_ready.append(dependent)
            ready = sorted(next_ready)

        return waves
