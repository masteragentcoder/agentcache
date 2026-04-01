"""Session-level metrics accumulator.

TODO: OpenTelemetry integration planned for later milestones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentcache.core.usage import Usage


@dataclass
class SessionMetrics:
    total_usage: Usage = field(default_factory=Usage)
    fork_count: int = 0
    compaction_count: int = 0
    memory_extraction_count: int = 0

    def record_response(self, usage: Usage) -> None:
        self.total_usage = self.total_usage + usage

    def record_fork(self, usage: Usage) -> None:
        self.total_usage = self.total_usage + usage
        self.fork_count += 1

    def record_compaction(self) -> None:
        self.compaction_count += 1

    def record_memory_extraction(self) -> None:
        self.memory_extraction_count += 1
