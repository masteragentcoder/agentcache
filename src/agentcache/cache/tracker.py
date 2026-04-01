"""Prompt-state tracker: records pre/post call snapshots and detects cache breaks."""

from __future__ import annotations

from dataclasses import dataclass, field

from agentcache.cache.explain import CacheBreakExplanation, explain_break
from agentcache.cache.prompt_state import PromptStateDiff, PromptStateSnapshot, diff_prompt_states
from agentcache.core.usage import Usage


@dataclass
class CacheStatus:
    session_id: str
    cache_read_tokens_last: int = 0
    cache_write_tokens_last: int = 0
    hit_rate_recent: float = 0.0
    last_break: CacheBreakExplanation | None = None
    call_count: int = 0

    def pretty(self) -> str:
        lines = [
            f"Session: {self.session_id}",
            f"Cache read tokens (last call): {self.cache_read_tokens_last:,}",
            f"Cache write tokens (last call): {self.cache_write_tokens_last:,}",
            f"Hit rate (last 10 calls): {self.hit_rate_recent:.1%}",
            f"Last break: {self.last_break.causes[0] if self.last_break and self.last_break.causes else 'none'}",
        ]
        return "\n".join(lines)


class PromptStateTracker:
    def __init__(self, *, min_drop_tokens: int = 2000) -> None:
        self.prev_snapshot: PromptStateSnapshot | None = None
        self.pending_diff: PromptStateDiff | None = None
        self.prev_cache_read_tokens: int | None = None
        self.min_drop_tokens = min_drop_tokens
        self.last_explanation: CacheBreakExplanation | None = None
        self._recent_hit_rates: list[float] = []

    def record_pre_call(self, snapshot: PromptStateSnapshot) -> None:
        if self.prev_snapshot is None:
            self.prev_snapshot = snapshot
            self.pending_diff = None
            return
        self.pending_diff = diff_prompt_states(self.prev_snapshot, snapshot)
        self.prev_snapshot = snapshot

    def record_post_call(self, usage: Usage) -> None:
        current = usage.cache_read_input_tokens

        hit_rate = usage.cache_hit_rate
        self._recent_hit_rates.append(hit_rate)
        if len(self._recent_hit_rates) > 10:
            self._recent_hit_rates = self._recent_hit_rates[-10:]

        if self.prev_cache_read_tokens is None:
            self.prev_cache_read_tokens = current
            return

        drop = self.prev_cache_read_tokens - current
        if drop >= self.min_drop_tokens and current < self.prev_cache_read_tokens * 0.95:
            self.last_explanation = explain_break(
                diff=self.pending_diff,
                previous=self.prev_cache_read_tokens,
                current=current,
            )
        else:
            self.last_explanation = None

        self.prev_cache_read_tokens = current

    def notify_compaction(self) -> None:
        """Prevent false-positive break detection after compaction."""
        self.prev_cache_read_tokens = None

    def recent_hit_rate(self) -> float:
        if not self._recent_hit_rates:
            return 0.0
        return sum(self._recent_hit_rates) / len(self._recent_hit_rates)

    def status(self, session_id: str, last_usage: Usage | None = None) -> CacheStatus:
        return CacheStatus(
            session_id=session_id,
            cache_read_tokens_last=(
                last_usage.cache_read_input_tokens if last_usage else 0
            ),
            cache_write_tokens_last=(
                last_usage.cache_creation_input_tokens if last_usage else 0
            ),
            hit_rate_recent=self.recent_hit_rate(),
            last_break=self.last_explanation,
        )
