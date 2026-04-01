"""Shared test fixtures including a FakeProvider for offline testing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from agentcache.core.messages import Message, TextBlock
from agentcache.core.tools import ToolSpec
from agentcache.core.usage import Usage
from agentcache.providers.base import ProviderResponse, ReasoningConfig


class FakeProvider:
    """In-memory provider that returns canned responses for tests."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses or ["Fake response."])
        self._call_count = 0
        self.last_payload: dict[str, Any] = {}

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
        self.last_payload = {
            "model": model,
            "system_prompt": system_prompt,
            "messages": messages,
            "tools": tools,
        }
        idx = min(self._call_count, len(self._responses) - 1)
        text = self._responses[idx]
        self._call_count += 1

        return ProviderResponse(
            message=Message(role="assistant", blocks=[TextBlock(text=text)]),
            usage=Usage(
                input_tokens=100,
                output_tokens=50,
                cache_read_input_tokens=80,
                cache_creation_input_tokens=20,
                total_tokens=150,
            ),
            model=model,
            request_id=f"fake-{self._call_count}",
        )


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()
