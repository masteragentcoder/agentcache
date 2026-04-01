"""Memory extraction via cache-safe forked helper."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from agentcache.cache.cache_safe_params import CacheSafeParamsFactory
from agentcache.core.messages import Message
from agentcache.fork.policies import ForkPolicy
from agentcache.fork.runner import ForkRunner
from agentcache.memory.models import MemoryUpdate, SessionMemory
from agentcache.memory.session_store import SessionMemoryStore

if TYPE_CHECKING:
    from agentcache.core.session import AgentSession

logger = logging.getLogger(__name__)

_EXTRACTION_PROMPT = """\
Review the conversation so far and extract structured memory updates.

Current memory state:
{current_memory}

Return a JSON object with this schema:
{{
  "additions": {{
    "preferences": [...],
    "project_facts": [...],
    "task_state": [...],
    "unresolved_questions": [...],
    "notable_artifacts": [...]
  }},
  "removals": {{
    "preferences": [...],
    "project_facts": [...],
    "task_state": [...],
    "unresolved_questions": [...],
    "notable_artifacts": [...]
  }},
  "rationale": "short explanation of changes"
}}

Only include fields that have actual additions or removals. Be concise.
Respond ONLY with the JSON object, no other text.
"""


class MemoryExtractor:
    def __init__(self, fork_runner: ForkRunner, store: SessionMemoryStore) -> None:
        self.fork_runner = fork_runner
        self.store = store

    async def extract_from_session(self, session: AgentSession) -> MemoryUpdate:
        current = self.store.load(session.session_id)
        current_text = current.to_markdown() if current else "(empty)"

        prompt = _EXTRACTION_PROMPT.format(current_memory=current_text)
        cache_safe = (
            session.last_cache_safe_params
            or CacheSafeParamsFactory.from_session(session)
        )

        result = await self.fork_runner.run(
            parent=session,
            prompt_messages=[Message.user(prompt)],
            cache_safe=cache_safe,
            policy=ForkPolicy.session_memory_update(),
        )

        update = _parse_memory_update(result.final_text)
        self.store.merge(session.session_id, update)
        return update


def _parse_memory_update(text: str) -> MemoryUpdate:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse memory extraction response as JSON")
        return MemoryUpdate()

    additions = SessionMemory(
        preferences=data.get("additions", {}).get("preferences", []),
        project_facts=data.get("additions", {}).get("project_facts", []),
        task_state=data.get("additions", {}).get("task_state", []),
        unresolved_questions=data.get("additions", {}).get("unresolved_questions", []),
        notable_artifacts=data.get("additions", {}).get("notable_artifacts", []),
    )

    removals = data.get("removals")
    rationale = data.get("rationale")

    return MemoryUpdate(additions=additions, removals=removals, rationale=rationale)
