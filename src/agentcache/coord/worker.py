"""Worker specifications and reports for coordinator mode. (Milestone 6)"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from agentcache.fork.policies import ForkPolicy


class ReportFormat(str, Enum):
    BULLETS = "bullets"
    PROSE = "prose"
    STRUCTURED = "structured"


@dataclass
class WorkerSpec:
    name: str
    instruction: str
    policy: ForkPolicy = field(default_factory=ForkPolicy.coord_worker)
    report_format: ReportFormat = ReportFormat.BULLETS
    persistent: bool = False


@dataclass
class WorkerReport:
    worker_id: str
    summary: str
    findings: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    suggested_actions: list[str] = field(default_factory=list)
    confidence: float = 0.0
