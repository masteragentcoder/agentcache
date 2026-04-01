from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Protocol

from agentcache.memory.models import MemoryUpdate, SessionMemory


class SessionMemoryStore(Protocol):
    def load(self, session_id: str) -> SessionMemory | None: ...
    def save(self, session_id: str, memory: SessionMemory) -> None: ...
    def merge(self, session_id: str, update: MemoryUpdate) -> SessionMemory: ...


class FileSessionMemoryStore:
    """Markdown-file backed session memory store."""

    def __init__(self, base_dir: str = ".agentcache/memory") -> None:
        self.base_dir = Path(base_dir)

    def _path(self, session_id: str) -> Path:
        return self.base_dir / f"{session_id}.md"

    def load(self, session_id: str) -> SessionMemory | None:
        path = self._path(session_id)
        if not path.exists():
            return None
        return _parse_memory_markdown(path.read_text(encoding="utf-8"))

    def save(self, session_id: str, memory: SessionMemory) -> None:
        path = self._path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(memory.to_markdown(), encoding="utf-8")

    def merge(self, session_id: str, update: MemoryUpdate) -> SessionMemory:
        existing = self.load(session_id) or SessionMemory()
        merged = _merge_memory(existing, update)
        self.save(session_id, merged)
        return merged


def _merge_memory(existing: SessionMemory, update: MemoryUpdate) -> SessionMemory:
    def _merge_list(current: list[str], additions: list[str], removals: list[str]) -> list[str]:
        result = [item for item in current if item not in removals]
        for item in additions:
            if item not in result:
                result.append(item)
        return result

    removal_map = update.removals or {}

    return SessionMemory(
        preferences=_merge_list(
            existing.preferences,
            update.additions.preferences,
            removal_map.get("preferences", []),
        ),
        project_facts=_merge_list(
            existing.project_facts,
            update.additions.project_facts,
            removal_map.get("project_facts", []),
        ),
        task_state=_merge_list(
            existing.task_state,
            update.additions.task_state,
            removal_map.get("task_state", []),
        ),
        unresolved_questions=_merge_list(
            existing.unresolved_questions,
            update.additions.unresolved_questions,
            removal_map.get("unresolved_questions", []),
        ),
        notable_artifacts=_merge_list(
            existing.notable_artifacts,
            update.additions.notable_artifacts,
            removal_map.get("notable_artifacts", []),
        ),
        updated_at=time.time(),
    )


def _parse_memory_markdown(text: str) -> SessionMemory:
    memory = SessionMemory()
    current_section: str | None = None

    section_map = {
        "preferences": "preferences",
        "project facts": "project_facts",
        "task state": "task_state",
        "unresolved questions": "unresolved_questions",
        "notable artifacts": "notable_artifacts",
    }

    for line in text.splitlines():
        heading_match = re.match(r"^##\s+(.+)$", line.strip())
        if heading_match:
            heading = heading_match.group(1).strip().lower()
            current_section = section_map.get(heading)
            continue

        if current_section and line.strip().startswith("- "):
            item = line.strip()[2:]
            getattr(memory, current_section).append(item)

    return memory
