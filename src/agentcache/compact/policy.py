from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CompactPolicy:
    max_input_tokens: int = 180_000
    target_input_tokens: int = 40_000
    clear_tool_results_for: tuple[str, ...] = (
        "shell",
        "glob",
        "grep",
        "file_read",
        "web_fetch",
        "web_search",
    )
    clear_tool_uses_for: tuple[str, ...] = (
        "file_edit",
        "file_write",
        "notebook_edit",
    )
    clear_thinking: bool = True
    keep_recent_thinking_turns: int = 1
    preserve_last_turns: int = 8
    preserve_all_user_messages: bool = True
