from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class SessionMemory:
    preferences: list[str] = field(default_factory=list)
    project_facts: list[str] = field(default_factory=list)
    task_state: list[str] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)
    notable_artifacts: list[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)

    def to_markdown(self) -> str:
        sections: list[str] = ["# Session Memory\n"]
        for heading, items in [
            ("Preferences", self.preferences),
            ("Project Facts", self.project_facts),
            ("Task State", self.task_state),
            ("Unresolved Questions", self.unresolved_questions),
            ("Notable Artifacts", self.notable_artifacts),
        ]:
            if items:
                sections.append(f"## {heading}")
                for item in items:
                    sections.append(f"- {item}")
                sections.append("")
        return "\n".join(sections)


@dataclass
class MemoryUpdate:
    additions: SessionMemory = field(default_factory=SessionMemory)
    removals: dict[str, list[str]] | None = None
    rationale: str | None = None
