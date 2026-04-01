"""Simple event bus for telemetry hooks.

TODO: OpenTelemetry spans/traces planned for later milestones.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Event:
    name: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class EventBus:
    def __init__(self) -> None:
        self._listeners: list[Callable[[Event], None]] = []

    def subscribe(self, listener: Callable[[Event], None]) -> None:
        self._listeners.append(listener)

    def emit(self, event: Event) -> None:
        for listener in self._listeners:
            listener(event)
