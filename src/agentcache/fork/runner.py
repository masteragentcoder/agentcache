"""ForkRunner: the central primitive for cache-safe forked helpers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from agentcache.cache.cache_safe_params import CacheSafeParams
from agentcache.cache.compatibility import CacheCompatibilityChecker, CompatibilityLevel
from agentcache.core.errors import CacheUnsafeForkError
from agentcache.core.messages import Message
from agentcache.core.usage import Usage
from agentcache.fork.context import SubagentContext, SubagentContextFactory
from agentcache.fork.policies import ForkPolicy
from agentcache.fork.result import ForkResult
from agentcache.providers.base import Provider

if TYPE_CHECKING:
    from agentcache.core.session import AgentSession

logger = logging.getLogger(__name__)


class ForkRunner:
    def __init__(self, provider: Provider) -> None:
        self.provider = provider

    async def run(
        self,
        *,
        parent: AgentSession,
        prompt_messages: list[Message],
        cache_safe: CacheSafeParams,
        policy: ForkPolicy,
    ) -> ForkResult:
        compatibility = CacheCompatibilityChecker().check(cache_safe, policy)
        if policy.cache_safe and compatibility.level == CompatibilityLevel.UNSAFE:
            raise CacheUnsafeForkError(compatibility.message)
        if compatibility.warnings:
            for w in compatibility.warnings:
                logger.warning("Fork cache-compat warning: %s", w)

        ctx = SubagentContextFactory.create(parent, policy)
        fork_messages = list(cache_safe.messages_prefix) + prompt_messages

        accumulated_usage = Usage()
        turns_used = 0
        max_turns = policy.max_turns or 10

        while turns_used < max_turns:
            if ctx.abort.aborted:
                break

            response = await self.provider.complete(
                model=cache_safe.model,
                system_prompt=cache_safe.system_prompt,
                messages=fork_messages,
                tools=list(cache_safe.tool_specs) or None,
                reasoning=cache_safe.reasoning_config,
            )

            fork_messages.append(response.message)
            accumulated_usage = accumulated_usage + response.usage
            turns_used += 1

            if not response.message.tool_calls:
                break

            for tc in response.message.tool_calls:
                fork_messages.append(
                    Message.tool_result(tc.id, f"[tool '{tc.name}' not available in fork]")
                )

        return ForkResult(
            messages=fork_messages[len(cache_safe.messages_prefix) :],
            usage=accumulated_usage,
            agent_id=ctx.agent_id,
            purpose=policy.purpose,
            turns_used=turns_used,
        )
