"""AgentSession: the main conversation and orchestration boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from agentcache.cache.cache_safe_params import CacheSafeParams, CacheSafeParamsFactory
from agentcache.cache.explain import CacheBreakExplanation
from agentcache.cache.prompt_state import PromptStateSnapshotFactory
from agentcache.cache.tracker import CacheStatus, PromptStateTracker
from agentcache.compact.microcompact import CompactResult, MicroCompactor
from agentcache.compact.policy import CompactPolicy
from agentcache.compact.tool_budget import ReplacementState, ToolResultBudgeter
from agentcache.core.ids import new_session_id
from agentcache.core.messages import Message
from agentcache.core.tools import ToolSpec
from agentcache.core.usage import Usage
from agentcache.fork.context import AbortHandle
from agentcache.fork.policies import ForkPolicy
from agentcache.fork.result import ForkResult
from agentcache.fork.runner import ForkRunner
from agentcache.providers.base import Provider, ProviderResponse, ReasoningConfig


@dataclass
class AgentSession:
    model: str
    provider: Provider
    system_prompt: str | list[dict[str, Any]] = ""
    tools: Sequence[ToolSpec] = field(default_factory=tuple)
    messages: list[Message] = field(default_factory=list)
    reasoning_config: ReasoningConfig | None = None
    cache_control: dict[str, Any] | None = None
    compact_policy: CompactPolicy | None = None
    replacement_state: ReplacementState | None = field(default_factory=ReplacementState)
    last_cache_safe_params: CacheSafeParams | None = None
    last_usage: Usage | None = None
    session_id: str = field(default_factory=new_session_id)
    app_state: dict[str, Any] = field(default_factory=dict)
    abort: AbortHandle = field(default_factory=AbortHandle)

    _cache_tracker: PromptStateTracker = field(
        default_factory=PromptStateTracker, repr=False
    )
    _compactor: MicroCompactor = field(default_factory=MicroCompactor, repr=False)
    _budgeter: ToolResultBudgeter = field(default_factory=ToolResultBudgeter, repr=False)

    async def respond(self, prompt: str) -> ProviderResponse:
        snapshot = PromptStateSnapshotFactory.from_messages(
            system_prompt=self.system_prompt,
            model=self.model,
            messages=self.messages,
            tools=list(self.tools) if self.tools else None,
            reasoning=self.reasoning_config,
            cache_control=self.cache_control,
        )
        self._cache_tracker.record_pre_call(snapshot)

        self.messages.append(Message.user(prompt))

        self.messages = self._budgeter.enforce(self.messages, self.replacement_state)

        if self.compact_policy:
            compact_result = self._compactor.compact_if_needed(
                self.messages, self.compact_policy
            )
            if compact_result.removed_tokens > 0:
                self.messages = compact_result.messages
                self._cache_tracker.notify_compaction()

        response = await self.provider.complete(
            model=self.model,
            system_prompt=self.system_prompt,
            messages=self.messages,
            tools=list(self.tools) if self.tools else None,
            reasoning=self.reasoning_config,
        )

        self.messages.append(response.message)
        self.last_usage = response.usage
        self._cache_tracker.record_post_call(response.usage)
        self.last_cache_safe_params = CacheSafeParamsFactory.from_session(self)

        return response

    async def fork(
        self,
        prompt: str,
        policy: ForkPolicy | None = None,
        tool_executor: Any | None = None,
    ) -> ForkResult:
        policy = policy or ForkPolicy.cache_safe_ephemeral()
        cache_safe = self.last_cache_safe_params or CacheSafeParamsFactory.from_session(self)

        runner = ForkRunner(self.provider, tool_executor=tool_executor)
        return await runner.run(
            parent=self,
            prompt_messages=[Message.user(prompt)],
            cache_safe=cache_safe,
            policy=policy,
        )

    def cache_status(self) -> CacheStatus:
        return self._cache_tracker.status(self.session_id, self.last_usage)

    def explain_last_cache_break(self) -> CacheBreakExplanation | None:
        return self._cache_tracker.last_explanation

    def compact_preview(self) -> str:
        policy = self.compact_policy or CompactPolicy()
        return self._compactor.preview(self.messages, policy)
