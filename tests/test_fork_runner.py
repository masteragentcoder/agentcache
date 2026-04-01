"""Tests for ForkRunner with fake provider."""

from __future__ import annotations

import pytest

from agentcache.cache.cache_safe_params import CacheSafeParamsFactory
from agentcache.compact.tool_budget import ReplacementState
from agentcache.core.errors import CacheUnsafeForkError
from agentcache.core.messages import Message
from agentcache.core.session import AgentSession
from agentcache.fork.policies import ForkPolicy
from agentcache.fork.runner import ForkRunner

from .conftest import FakeProvider


def _make_session(provider: FakeProvider) -> AgentSession:
    return AgentSession(
        model="test-model",
        provider=provider,
        system_prompt="You are a test assistant.",
    )


class TestForkRunner:
    @pytest.mark.asyncio
    async def test_basic_fork(self):
        provider = FakeProvider(["Fork reply."])
        session = _make_session(provider)
        cache_safe = CacheSafeParamsFactory.from_session(session)

        runner = ForkRunner(provider)
        result = await runner.run(
            parent=session,
            prompt_messages=[Message.user("Summarize.")],
            cache_safe=cache_safe,
            policy=ForkPolicy.cache_safe_ephemeral(),
        )

        assert result.final_text == "Fork reply."
        assert result.turns_used == 1
        assert result.usage.input_tokens == 100

    @pytest.mark.asyncio
    async def test_fork_preserves_messages_prefix(self):
        provider = FakeProvider(["Done."])
        session = _make_session(provider)
        session.messages.append(Message.user("first message"))
        session.messages.append(Message.assistant("first reply"))
        cache_safe = CacheSafeParamsFactory.from_session(session)

        runner = ForkRunner(provider)
        await runner.run(
            parent=session,
            prompt_messages=[Message.user("Second question.")],
            cache_safe=cache_safe,
            policy=ForkPolicy.cache_safe_ephemeral(),
        )

        sent_messages = provider.last_payload["messages"]
        # first msg + first reply + fork prompt (response appended after send)
        assert len(sent_messages) >= 3
        roles = [m.role if isinstance(m, Message) else m["role"] for m in sent_messages[:3]]
        assert roles == ["user", "assistant", "user"]

    @pytest.mark.asyncio
    async def test_fork_does_not_mutate_parent(self):
        provider = FakeProvider(["Fork only."])
        session = _make_session(provider)
        original_count = len(session.messages)
        cache_safe = CacheSafeParamsFactory.from_session(session)

        runner = ForkRunner(provider)
        await runner.run(
            parent=session,
            prompt_messages=[Message.user("Side question.")],
            cache_safe=cache_safe,
            policy=ForkPolicy.cache_safe_ephemeral(),
        )

        assert len(session.messages) == original_count

    @pytest.mark.asyncio
    async def test_max_output_tokens_warns_but_runs(self):
        """max_output_tokens override with cache_safe=True is LIKELY_UNSAFE: warns, doesn't raise."""
        provider = FakeProvider(["Warned but ran."])
        session = _make_session(provider)
        cache_safe = CacheSafeParamsFactory.from_session(session)

        runner = ForkRunner(provider)
        result = await runner.run(
            parent=session,
            prompt_messages=[Message.user("Fork with output limit.")],
            cache_safe=cache_safe,
            policy=ForkPolicy(cache_safe=True, max_output_tokens=99),
        )
        assert result.final_text == "Warned but ran."

    @pytest.mark.asyncio
    async def test_cache_unsafe_policy_with_cache_safe_true_raises(self):
        """cache_safe=True + checker returning UNSAFE should raise."""
        from agentcache.cache.compatibility import (
            CacheCompatibilityChecker,
            CompatibilityLevel,
            CompatibilityResult,
        )
        from unittest.mock import patch

        provider = FakeProvider(["Should not reach."])
        session = _make_session(provider)
        cache_safe = CacheSafeParamsFactory.from_session(session)

        forced_result = CompatibilityResult(
            level=CompatibilityLevel.UNSAFE,
            warnings=["forced unsafe"],
        )
        runner = ForkRunner(provider)
        with patch.object(
            CacheCompatibilityChecker, "check", return_value=forced_result
        ):
            with pytest.raises(CacheUnsafeForkError):
                await runner.run(
                    parent=session,
                    prompt_messages=[Message.user("Bad fork.")],
                    cache_safe=cache_safe,
                    policy=ForkPolicy(cache_safe=True),
                )


class TestForkPolicyPresets:
    def test_cache_safe_ephemeral(self):
        p = ForkPolicy.cache_safe_ephemeral()
        assert p.cache_safe is True
        assert p.isolated is True
        assert p.skip_transcript is True
        assert p.skip_cache_write is True
        assert p.max_turns == 1

    def test_background_summary(self):
        p = ForkPolicy.background_summary()
        assert p.cache_safe is True
        assert p.skip_cache_write is True

    def test_session_memory_update(self):
        p = ForkPolicy.session_memory_update()
        assert p.cache_safe is True
        assert p.skip_cache_write is False

    def test_coord_worker(self):
        p = ForkPolicy.coord_worker()
        assert p.max_turns is None

    def test_cache_unsafe_rewrite(self):
        p = ForkPolicy.cache_unsafe_rewrite()
        assert p.cache_safe is False
