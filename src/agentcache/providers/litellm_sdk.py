from __future__ import annotations

from typing import Any

import litellm

from agentcache.core.errors import ProviderError
from agentcache.core.messages import Message
from agentcache.core.tools import ToolSpec
from agentcache.providers.adapters import build_litellm_payload, normalize_litellm_response
from agentcache.providers.base import ProviderResponse, ReasoningConfig


class LiteLLMSDKProvider:
    """Provider backed by the LiteLLM Python SDK (litellm.acompletion)."""

    def __init__(self, *, default_timeout_s: float = 120.0) -> None:
        self.default_timeout_s = default_timeout_s

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
    ) -> ProviderResponse:
        payload = build_litellm_payload(
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            reasoning=reasoning,
            metadata=metadata,
            extra_body=extra_body,
        )
        try:
            raw = await litellm.acompletion(
                **payload,
                timeout=timeout_s or self.default_timeout_s,
            )
        except Exception as exc:
            raise ProviderError(f"LiteLLM completion failed: {exc}") from exc

        return normalize_litellm_response(raw)
