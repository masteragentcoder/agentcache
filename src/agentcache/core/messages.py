from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class TextBlock:
    text: str


@dataclass(frozen=True)
class ToolCallBlock:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResultBlock:
    tool_call_id: str
    result: Any
    is_error: bool = False


ContentBlock = TextBlock | ToolCallBlock | ToolResultBlock


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    blocks: list[ContentBlock] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    meta: dict[str, Any] | None = None

    @classmethod
    def user(cls, text: str) -> Message:
        return cls(role="user", blocks=[TextBlock(text)])

    @classmethod
    def assistant(cls, text: str) -> Message:
        return cls(role="assistant", blocks=[TextBlock(text)])

    @classmethod
    def tool_result(
        cls, tool_call_id: str, result: Any, *, is_error: bool = False
    ) -> Message:
        return cls(
            role="tool",
            blocks=[ToolResultBlock(tool_call_id=tool_call_id, result=result, is_error=is_error)],
        )

    @property
    def text(self) -> str:
        parts = [b.text for b in self.blocks if isinstance(b, TextBlock)]
        return "\n".join(parts)

    @property
    def tool_calls(self) -> list[ToolCallBlock]:
        return [b for b in self.blocks if isinstance(b, ToolCallBlock)]

    @property
    def tool_results(self) -> list[ToolResultBlock]:
        return [b for b in self.blocks if isinstance(b, ToolResultBlock)]

    def token_estimate(self) -> int:
        """Rough ~4 chars/token estimate for budget decisions."""
        total_chars = 0
        for block in self.blocks:
            if isinstance(block, TextBlock):
                total_chars += len(block.text)
            elif isinstance(block, ToolCallBlock):
                total_chars += len(str(block.arguments)) + len(block.name)
            elif isinstance(block, ToolResultBlock):
                total_chars += len(str(block.result))
        return max(1, total_chars // 4)
