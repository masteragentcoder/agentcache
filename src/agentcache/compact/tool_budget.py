"""Tool-result budgeting and replacement state.

ReplacementState must be *cloned* (not reset) when forking, to preserve
deterministic replacement decisions on the inherited prefix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentcache.core.messages import Message, TextBlock, ToolResultBlock


@dataclass
class ReplacementRecord:
    tool_call_id: str
    original_token_estimate: int
    replacement_text: str


@dataclass
class ReplacementState:
    replacements: dict[str, ReplacementRecord] = field(default_factory=dict)

    def clone(self) -> ReplacementState:
        return ReplacementState(replacements=dict(self.replacements))


class ToolResultBudgeter:
    def __init__(self, per_turn_budget_tokens: int = 8000) -> None:
        self.per_turn_budget_tokens = per_turn_budget_tokens

    def enforce(
        self,
        messages: list[Message],
        state: ReplacementState | None,
    ) -> list[Message]:
        if state is None:
            return messages

        result: list[Message] = []
        for msg in messages:
            if msg.role != "tool":
                result.append(msg)
                continue

            new_blocks = []
            for block in msg.blocks:
                if not isinstance(block, ToolResultBlock):
                    new_blocks.append(block)
                    continue

                if block.tool_call_id in state.replacements:
                    rec = state.replacements[block.tool_call_id]
                    new_blocks.append(
                        ToolResultBlock(
                            tool_call_id=block.tool_call_id,
                            result=rec.replacement_text,
                            is_error=block.is_error,
                        )
                    )
                    continue

                est = len(str(block.result)) // 4
                if est > self.per_turn_budget_tokens:
                    replacement_text = (
                        f"[result truncated: ~{est} tokens, "
                        f"tool_call_id={block.tool_call_id}]"
                    )
                    state.replacements[block.tool_call_id] = ReplacementRecord(
                        tool_call_id=block.tool_call_id,
                        original_token_estimate=est,
                        replacement_text=replacement_text,
                    )
                    new_blocks.append(
                        ToolResultBlock(
                            tool_call_id=block.tool_call_id,
                            result=replacement_text,
                            is_error=block.is_error,
                        )
                    )
                else:
                    new_blocks.append(block)

            result.append(Message(role=msg.role, blocks=new_blocks, created_at=msg.created_at))
        return result
