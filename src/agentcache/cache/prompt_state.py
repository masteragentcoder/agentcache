"""Hashable prompt-state snapshots for cache-break detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentcache.cache.hashes import stable_hash
from agentcache.core.messages import Message
from agentcache.core.tools import ToolSpec
from agentcache.providers.base import ReasoningConfig


@dataclass(frozen=True)
class PromptStateSnapshot:
    system_hash: str
    tools_hash: str
    cache_control_hash: str
    model: str
    reasoning_hash: str
    effort_value: str | None
    extra_body_hash: str | None
    beta_flags: tuple[str, ...]
    system_char_count: int


@dataclass
class PromptStateDiff:
    model_changed: bool = False
    system_changed: bool = False
    tools_changed: bool = False
    cache_control_changed: bool = False
    reasoning_changed: bool = False
    effort_changed: bool = False
    extra_body_changed: bool = False
    betas_changed: bool = False
    system_char_delta: int = 0

    @property
    def has_changes(self) -> bool:
        return any([
            self.model_changed,
            self.system_changed,
            self.tools_changed,
            self.cache_control_changed,
            self.reasoning_changed,
            self.effort_changed,
            self.extra_body_changed,
            self.betas_changed,
        ])


class PromptStateSnapshotFactory:
    @staticmethod
    def from_request(
        *,
        system_prompt: str | list[dict[str, Any]],
        model: str,
        tools: list[ToolSpec] | None = None,
        reasoning: ReasoningConfig | None = None,
        cache_control: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
        beta_flags: list[str] | None = None,
    ) -> PromptStateSnapshot:
        sys_text = system_prompt if isinstance(system_prompt, str) else str(system_prompt)
        return PromptStateSnapshot(
            system_hash=stable_hash(system_prompt),
            tools_hash=stable_hash([t.parameters for t in tools] if tools else None),
            cache_control_hash=stable_hash(cache_control),
            model=model,
            reasoning_hash=stable_hash(reasoning),
            effort_value=reasoning.effort if reasoning else None,
            extra_body_hash=stable_hash(extra_body) if extra_body else None,
            beta_flags=tuple(sorted(beta_flags)) if beta_flags else (),
            system_char_count=len(sys_text),
        )

    @staticmethod
    def from_messages(
        *,
        system_prompt: str | list[dict[str, Any]],
        model: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        reasoning: ReasoningConfig | None = None,
        cache_control: dict[str, Any] | None = None,
    ) -> PromptStateSnapshot:
        sys_text = system_prompt if isinstance(system_prompt, str) else str(system_prompt)
        return PromptStateSnapshot(
            system_hash=stable_hash(system_prompt),
            tools_hash=stable_hash(
                [{"name": t.name, "params": t.parameters} for t in tools] if tools else None
            ),
            cache_control_hash=stable_hash(cache_control),
            model=model,
            reasoning_hash=stable_hash(reasoning),
            effort_value=reasoning.effort if reasoning else None,
            extra_body_hash=None,
            beta_flags=(),
            system_char_count=len(sys_text),
        )


def diff_prompt_states(
    prev: PromptStateSnapshot, curr: PromptStateSnapshot
) -> PromptStateDiff:
    return PromptStateDiff(
        model_changed=prev.model != curr.model,
        system_changed=prev.system_hash != curr.system_hash,
        tools_changed=prev.tools_hash != curr.tools_hash,
        cache_control_changed=prev.cache_control_hash != curr.cache_control_hash,
        reasoning_changed=prev.reasoning_hash != curr.reasoning_hash,
        effort_changed=prev.effort_value != curr.effort_value,
        extra_body_changed=prev.extra_body_hash != curr.extra_body_hash,
        betas_changed=prev.beta_flags != curr.beta_flags,
        system_char_delta=curr.system_char_count - prev.system_char_count,
    )
