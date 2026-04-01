"""Tests for tool execution callback in ForkRunner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from agentcache.cache.cache_safe_params import CacheSafeParamsFactory
from agentcache.core.messages import Message, TextBlock, ToolCallBlock
from agentcache.core.session import AgentSession
from agentcache.core.tools import ToolSpec
from agentcache.core.usage import Usage
from agentcache.fork.policies import ForkPolicy
from agentcache.fork.runner import ForkRunner
from agentcache.providers.base import ProviderResponse


class ToolCallingProvider:
    """Provider that returns a tool call on first call, then a text response."""

    def __init__(self) -> None:
        self._call_count = 0
        self.tool_call_name = "get_weather"
        self.tool_call_args = {"city": "Paris"}

    async def complete(self, **kwargs: Any) -> ProviderResponse:
        self._call_count += 1
        if self._call_count == 1:
            return ProviderResponse(
                message=Message(
                    role="assistant",
                    blocks=[
                        ToolCallBlock(
                            id="call_1",
                            name=self.tool_call_name,
                            arguments=self.tool_call_args,
                        )
                    ],
                ),
                usage=Usage(input_tokens=50, output_tokens=10, total_tokens=60),
                model="test",
                request_id="fake-1",
            )
        return ProviderResponse(
            message=Message(
                role="assistant",
                blocks=[TextBlock(text="The weather in Paris is sunny.")],
            ),
            usage=Usage(input_tokens=60, output_tokens=20, total_tokens=80),
            model="test",
            request_id="fake-2",
        )


class TestToolExecutor:
    @pytest.mark.asyncio
    async def test_tool_executor_called(self):
        provider = ToolCallingProvider()
        session = AgentSession(
            model="test-model",
            provider=provider,
            system_prompt="test",
        )
        cache_safe = CacheSafeParamsFactory.from_session(session)

        calls: list[tuple[str, str, dict]] = []

        def executor(tool_call_id: str, name: str, args: dict) -> str:
            calls.append((tool_call_id, name, args))
            return "Sunny, 22C"

        runner = ForkRunner(provider, tool_executor=executor)
        result = await runner.run(
            parent=session,
            prompt_messages=[Message.user("What's the weather?")],
            cache_safe=cache_safe,
            policy=ForkPolicy(cache_safe=True, max_turns=5),
        )

        assert len(calls) == 1
        assert calls[0] == ("call_1", "get_weather", {"city": "Paris"})
        assert result.final_text == "The weather in Paris is sunny."
        assert result.turns_used == 2

    @pytest.mark.asyncio
    async def test_no_executor_returns_stub(self):
        provider = ToolCallingProvider()
        session = AgentSession(
            model="test-model",
            provider=provider,
            system_prompt="test",
        )
        cache_safe = CacheSafeParamsFactory.from_session(session)

        runner = ForkRunner(provider)
        result = await runner.run(
            parent=session,
            prompt_messages=[Message.user("What's the weather?")],
            cache_safe=cache_safe,
            policy=ForkPolicy(cache_safe=True, max_turns=5),
        )

        assert result.turns_used == 2
        tool_result_msgs = [
            m for m in result.messages if m.role == "tool"
        ]
        assert len(tool_result_msgs) == 1
        result_text = str(tool_result_msgs[0].tool_results[0].result)
        assert "not available in fork" in result_text

    @pytest.mark.asyncio
    async def test_async_executor(self):
        provider = ToolCallingProvider()
        session = AgentSession(
            model="test-model",
            provider=provider,
            system_prompt="test",
        )
        cache_safe = CacheSafeParamsFactory.from_session(session)

        async def async_executor(tool_call_id: str, name: str, args: dict) -> str:
            return "Async result: cloudy"

        runner = ForkRunner(provider, tool_executor=async_executor)
        result = await runner.run(
            parent=session,
            prompt_messages=[Message.user("Weather?")],
            cache_safe=cache_safe,
            policy=ForkPolicy(cache_safe=True, max_turns=5),
        )

        assert result.final_text == "The weather in Paris is sunny."
        assert result.turns_used == 2
