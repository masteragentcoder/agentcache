from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from agentcache.core.messages import Message
from agentcache.core.tools import ToolSpec
from agentcache.core.usage import Usage


@dataclass(frozen=True)
class ReasoningConfig:
    enabled: bool = False
    effort: str | None = None
    budget_tokens: int | None = None


@dataclass
class ProviderResponse:
    message: Message
    usage: Usage
    model: str
    request_id: str | None = None
    raw: Any = None
    stop_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.message.text


class Provider(Protocol):
    async def complete(
        self,
        *,
        model: str,
        system_prompt: str | list[dict[str, Any]],
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        reasoning: ReasoningConfig | None = None,
        metadata: dict[str, Any] | None = None,
        timeout_s: float | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ProviderResponse: ...
