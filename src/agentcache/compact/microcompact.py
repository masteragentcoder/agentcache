"""Pre-call context compaction: clear stale tool results and thinking blocks."""

from __future__ import annotations

from dataclasses import dataclass, field

from agentcache.compact.policy import CompactPolicy
from agentcache.core.messages import Message, TextBlock, ToolCallBlock, ToolResultBlock


def _estimate_tokens(messages: list[Message]) -> int:
    return sum(m.token_estimate() for m in messages)


@dataclass
class CompactResult:
    messages: list[Message]
    removed_tokens: int = 0
    actions: list[str] = field(default_factory=list)

    @property
    def preview(self) -> str:
        if not self.actions:
            return "No compaction needed."
        lines = ["Would remove:"]
        for a in self.actions:
            lines.append(f"  - {a}")
        return "\n".join(lines)


class MicroCompactor:
    def compact_if_needed(
        self,
        messages: list[Message],
        policy: CompactPolicy,
    ) -> CompactResult:
        est_tokens = _estimate_tokens(messages)
        if est_tokens < policy.max_input_tokens:
            return CompactResult(messages=messages, removed_tokens=0)

        actions: list[str] = []
        compacted = list(messages)

        compacted, removed_tool = _clear_stale_tool_results(
            compacted,
            allowed_tools=policy.clear_tool_results_for,
            preserve_last_turns=policy.preserve_last_turns,
        )
        if removed_tool > 0:
            actions.append(f"cleared stale tool results (~{removed_tool} est tokens)")

        if policy.clear_thinking:
            compacted, removed_think = _clear_stale_thinking(
                compacted,
                keep_recent_turns=policy.keep_recent_thinking_turns,
            )
            if removed_think > 0:
                actions.append(f"cleared stale thinking blocks (~{removed_think} est tokens)")

        compacted, removed_tool_uses = _clear_stale_tool_uses(
            compacted,
            tool_names=policy.clear_tool_uses_for,
            preserve_last_turns=policy.preserve_last_turns,
        )
        if removed_tool_uses > 0:
            actions.append(f"cleared stale tool-use blocks (~{removed_tool_uses} est tokens)")

        new_est = _estimate_tokens(compacted)
        return CompactResult(
            messages=compacted,
            removed_tokens=max(0, est_tokens - new_est),
            actions=actions,
        )

    def preview(
        self,
        messages: list[Message],
        policy: CompactPolicy,
    ) -> str:
        est_before = _estimate_tokens(messages)
        result = self.compact_if_needed(messages, policy)
        est_after = _estimate_tokens(result.messages)
        lines = [
            f"Estimated tokens before: {est_before:,}",
            f"Estimated tokens after:  {est_after:,}",
        ]
        if result.actions:
            lines.append("Actions:")
            for a in result.actions:
                lines.append(f"  - {a}")
        else:
            lines.append("No compaction needed.")
        return "\n".join(lines)


def _clear_stale_tool_results(
    messages: list[Message],
    allowed_tools: tuple[str, ...],
    preserve_last_turns: int,
) -> tuple[list[Message], int]:
    if not messages:
        return messages, 0

    boundary = max(0, len(messages) - preserve_last_turns)
    removed = 0
    result: list[Message] = []

    for i, msg in enumerate(messages):
        if i >= boundary or msg.role != "tool":
            result.append(msg)
            continue

        new_blocks = []
        for block in msg.blocks:
            if isinstance(block, ToolResultBlock):
                est = len(str(block.result)) // 4
                placeholder = f"[stale result cleared, ~{est} tokens]"
                new_blocks.append(
                    ToolResultBlock(
                        tool_call_id=block.tool_call_id,
                        result=placeholder,
                        is_error=block.is_error,
                    )
                )
                removed += est
            else:
                new_blocks.append(block)
        result.append(Message(role=msg.role, blocks=new_blocks, created_at=msg.created_at))

    return result, removed


def _clear_stale_thinking(
    messages: list[Message],
    keep_recent_turns: int,
) -> tuple[list[Message], int]:
    """Strip thinking content from assistant messages except the last N turns."""
    assistant_indices = [i for i, m in enumerate(messages) if m.role == "assistant"]
    if len(assistant_indices) <= keep_recent_turns:
        return messages, 0

    stale_indices = set(assistant_indices[:-keep_recent_turns])
    removed = 0
    result: list[Message] = []

    for i, msg in enumerate(messages):
        if i not in stale_indices:
            result.append(msg)
            continue

        new_blocks = []
        for block in msg.blocks:
            if isinstance(block, TextBlock) and block.text.startswith("<thinking>"):
                removed += len(block.text) // 4
            else:
                new_blocks.append(block)

        if new_blocks:
            result.append(Message(role=msg.role, blocks=new_blocks, created_at=msg.created_at))
        else:
            result.append(msg)

    return result, removed


def _clear_stale_tool_uses(
    messages: list[Message],
    tool_names: tuple[str, ...],
    preserve_last_turns: int,
) -> tuple[list[Message], int]:
    if not messages:
        return messages, 0

    boundary = max(0, len(messages) - preserve_last_turns)
    removed = 0
    result: list[Message] = []

    for i, msg in enumerate(messages):
        if i >= boundary or msg.role != "assistant":
            result.append(msg)
            continue

        new_blocks = []
        for block in msg.blocks:
            if isinstance(block, ToolCallBlock) and block.name in tool_names:
                removed += len(str(block.arguments)) // 4
            else:
                new_blocks.append(block)
        result.append(Message(role=msg.role, blocks=new_blocks, created_at=msg.created_at))

    return result, removed
