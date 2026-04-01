"""Progress summarization for coordinator workers. (Milestone 6)

TODO: Full implementation deferred until coordinator is built.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentcache.fork.policies import ForkPolicy

if TYPE_CHECKING:
    from agentcache.core.session import AgentSession


class ProgressSummarizer:
    async def summarize_worker(self, worker_session: AgentSession) -> str:
        """Summarize a worker's progress in 1-2 sentences via cache-safe fork."""
        result = await worker_session.fork(
            prompt="Summarize current progress in 2 sentences max.",
            policy=ForkPolicy.background_summary(),
        )
        return result.final_text.strip()
