"""Tests for CacheSafeParams construction and fork compatibility."""

from __future__ import annotations

import pytest

from agentcache.cache.cache_safe_params import CacheSafeParams, CacheSafeParamsFactory
from agentcache.cache.compatibility import (
    CacheCompatibilityChecker,
    CompatibilityLevel,
)
from agentcache.compact.tool_budget import ReplacementState
from agentcache.core.messages import Message
from agentcache.core.session import AgentSession
from agentcache.core.tools import ToolSpec
from agentcache.fork.policies import ForkPolicy


def _make_session(fake_provider, **kwargs):
    defaults = {
        "model": "test-model",
        "provider": fake_provider,
        "system_prompt": "You are a test assistant.",
    }
    defaults.update(kwargs)
    return AgentSession(**defaults)


class TestCacheSafeParamsFactory:
    def test_from_session_captures_model_and_system(self, fake_provider):
        session = _make_session(fake_provider)
        params = CacheSafeParamsFactory.from_session(session)

        assert params.model == "test-model"
        assert params.system_prompt == "You are a test assistant."
        assert params.tool_specs == ()
        assert params.messages_prefix == ()

    def test_from_session_captures_tools(self, fake_provider):
        tool = ToolSpec(name="grep", description="Search", parameters={"type": "object"})
        session = _make_session(fake_provider, tools=[tool])
        params = CacheSafeParamsFactory.from_session(session)

        assert len(params.tool_specs) == 1
        assert params.tool_specs[0].name == "grep"

    def test_from_session_captures_messages(self, fake_provider):
        session = _make_session(fake_provider)
        session.messages.append(Message.user("hello"))
        session.messages.append(Message.assistant("hi"))
        params = CacheSafeParamsFactory.from_session(session)

        assert len(params.messages_prefix) == 2

    def test_frozen(self, fake_provider):
        session = _make_session(fake_provider)
        params = CacheSafeParamsFactory.from_session(session)
        with pytest.raises(AttributeError):
            params.model = "other-model"  # type: ignore[misc]


class TestCacheCompatibilityChecker:
    def test_safe_by_default(self, fake_provider):
        session = _make_session(fake_provider)
        params = CacheSafeParamsFactory.from_session(session)
        policy = ForkPolicy.cache_safe_ephemeral()

        result = CacheCompatibilityChecker().check(params, policy)
        assert result.level == CompatibilityLevel.SAFE

    def test_max_output_tokens_warns(self, fake_provider):
        session = _make_session(fake_provider)
        params = CacheSafeParamsFactory.from_session(session)
        policy = ForkPolicy(cache_safe=True, max_output_tokens=4096)

        result = CacheCompatibilityChecker().check(params, policy)
        assert result.level == CompatibilityLevel.LIKELY_UNSAFE
        assert any("max_output_tokens" in w for w in result.warnings)

    def test_cache_unsafe_policy(self, fake_provider):
        session = _make_session(fake_provider)
        params = CacheSafeParamsFactory.from_session(session)
        policy = ForkPolicy.cache_unsafe_rewrite()

        result = CacheCompatibilityChecker().check(params, policy)
        assert result.level == CompatibilityLevel.UNSAFE


class TestReplacementStateCloning:
    """The cloning trick: forked replacement state must match parent decisions."""

    def test_clone_preserves_existing_replacements(self):
        state = ReplacementState()
        from agentcache.compact.tool_budget import ReplacementRecord

        state.replacements["tc_1"] = ReplacementRecord(
            tool_call_id="tc_1",
            original_token_estimate=10000,
            replacement_text="[truncated]",
        )

        cloned = state.clone()

        assert "tc_1" in cloned.replacements
        assert cloned.replacements["tc_1"].replacement_text == "[truncated]"

    def test_clone_is_independent(self):
        state = ReplacementState()
        cloned = state.clone()

        from agentcache.compact.tool_budget import ReplacementRecord

        cloned.replacements["tc_2"] = ReplacementRecord(
            tool_call_id="tc_2",
            original_token_estimate=5000,
            replacement_text="[new]",
        )

        assert "tc_2" not in state.replacements
        assert "tc_2" in cloned.replacements
