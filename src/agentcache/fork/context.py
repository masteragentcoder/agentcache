"""Isolated execution context for forked subagents.

The critical design invariant: ReplacementState is *cloned*, not reset.
A fresh state would produce different replacement decisions on inherited
transcript items, changing the wire prefix and causing cache misses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agentcache.compact.tool_budget import ReplacementState
from agentcache.core.ids import new_agent_id
from agentcache.core.messages import Message

if TYPE_CHECKING:
    from agentcache.core.session import AgentSession
    from agentcache.fork.policies import ForkPolicy


@dataclass
class AbortHandle:
    """Simple cooperative cancellation token."""

    _aborted: bool = False

    def abort(self) -> None:
        self._aborted = True

    @property
    def aborted(self) -> bool:
        return self._aborted

    def child(self) -> AbortHandle:
        return AbortHandle()


@dataclass
class SubagentContext:
    messages: list[Message]
    replacement_state: ReplacementState | None
    app_state: dict[str, Any]
    agent_id: str
    agent_type: str | None = None
    abort: AbortHandle = field(default_factory=AbortHandle)


class SubagentContextFactory:
    @staticmethod
    def create(parent: AgentSession, policy: ForkPolicy) -> SubagentContext:
        return SubagentContext(
            messages=list(parent.messages),
            replacement_state=(
                parent.replacement_state.clone()
                if parent.replacement_state is not None
                else None
            ),
            app_state={} if not policy.share_state else dict(parent.app_state),
            agent_id=new_agent_id(),
            agent_type=policy.purpose,
            abort=(
                parent.abort
                if policy.share_abort
                else (parent.abort.child() if parent.abort else AbortHandle())
            ),
        )
