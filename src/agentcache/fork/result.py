from __future__ import annotations

from dataclasses import dataclass, field

from agentcache.core.messages import Message
from agentcache.core.usage import Usage


@dataclass
class ForkResult:
    messages: list[Message] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    agent_id: str = ""
    purpose: str = ""
    turns_used: int = 0

    @property
    def final_text(self) -> str:
        for msg in reversed(self.messages):
            if msg.role == "assistant" and msg.text:
                return msg.text
        return ""
