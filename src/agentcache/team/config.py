"""Team configuration: roles and shared system prompt.

The cache-aware insight: all roles share ONE system prompt (the cached prefix).
Role differentiation happens in the fork prompt, not the system prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentRole:
    """A specialist role within a team."""

    name: str
    instructions: str
    purpose: str = "specialist"


@dataclass
class TeamConfig:
    """Defines a team of agents that share a single cached prefix.

    The ``system_prompt`` is identical for every role -- this is what the
    provider caches.  Each role's ``instructions`` are injected into the fork
    prompt so they appear *after* the cached prefix.
    """

    system_prompt: str
    roles: list[AgentRole] = field(default_factory=list)
    coordinator_instructions: str = (
        "You are acting as the Coordinator. Break the goal into specialist "
        "tasks and produce delegation briefs."
    )

    def role_names(self) -> list[str]:
        return [r.name for r in self.roles]
