"""Human-readable cache-break explanations."""

from __future__ import annotations

from dataclasses import dataclass, field

from agentcache.cache.prompt_state import PromptStateDiff


@dataclass
class CacheBreakExplanation:
    broke: bool
    causes: list[str] = field(default_factory=list)
    token_drop: int = 0
    previous_cache_read_tokens: int | None = None
    current_cache_read_tokens: int | None = None
    notes: list[str] = field(default_factory=list)

    def pretty(self) -> str:
        if not self.broke:
            return "No cache break detected."
        lines = ["Cache break detected."]
        if self.causes:
            lines.append("Primary causes:")
            for c in self.causes:
                lines.append(f"  - {c}")
        if self.notes:
            lines.append("Notes:")
            for n in self.notes:
                lines.append(f"  - {n}")
        if self.previous_cache_read_tokens is not None:
            lines.append(f"Previous cache-read tokens: {self.previous_cache_read_tokens:,}")
        if self.current_cache_read_tokens is not None:
            lines.append(f"Current cache-read tokens: {self.current_cache_read_tokens:,}")
        return "\n".join(lines)


def explain_break(
    diff: PromptStateDiff | None,
    previous: int,
    current: int,
) -> CacheBreakExplanation:
    causes: list[str] = []
    notes: list[str] = []

    if diff is not None:
        if diff.model_changed:
            causes.append("model changed")
        if diff.system_changed:
            delta = diff.system_char_delta
            sign = "+" if delta > 0 else ""
            causes.append(f"system prompt changed ({sign}{delta} chars)")
        if diff.tools_changed:
            causes.append("tool schema changed")
        if diff.cache_control_changed:
            causes.append("cache_control changed (scope or TTL)")
        if diff.reasoning_changed:
            causes.append("reasoning config changed")
        if diff.effort_changed:
            causes.append("effort value changed")
        if diff.extra_body_changed:
            causes.append("extra request body changed")
        if diff.betas_changed:
            causes.append("beta flags changed")

    if not causes:
        notes.append("possible TTL expiry (5min or 1h) or server-side cache eviction")

    return CacheBreakExplanation(
        broke=True,
        causes=causes,
        token_drop=previous - current,
        previous_cache_read_tokens=previous,
        current_cache_read_tokens=current,
        notes=notes,
    )
