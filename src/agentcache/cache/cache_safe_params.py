"""Frozen snapshot of everything a fork must preserve to share prompt cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agentcache.core.messages import Message
from agentcache.core.tools import ToolSpec
from agentcache.providers.base import ReasoningConfig

if TYPE_CHECKING:
    from agentcache.core.session import AgentSession


@dataclass(frozen=True)
class CacheSafeParams:
    system_prompt: str | list[dict[str, Any]]
    model: str
    tool_specs: tuple[ToolSpec, ...]
    messages_prefix: tuple[Message, ...]
    reasoning_config: ReasoningConfig | None = None
    cache_control: dict[str, Any] | None = None


class CacheSafeParamsFactory:
    @staticmethod
    def from_session(session: AgentSession) -> CacheSafeParams:
        return CacheSafeParams(
            system_prompt=session.system_prompt,
            model=session.model,
            tool_specs=tuple(session.tools),
            messages_prefix=tuple(session.messages),
            reasoning_config=session.reasoning_config,
            cache_control=session.cache_control,
        )
