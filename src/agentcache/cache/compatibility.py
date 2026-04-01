"""Cache-safety compatibility checker for fork policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from agentcache.cache.cache_safe_params import CacheSafeParams

if TYPE_CHECKING:
    from agentcache.fork.policies import ForkPolicy


class CompatibilityLevel(str, Enum):
    SAFE = "safe"
    LIKELY_SAFE = "likely_safe"
    LIKELY_UNSAFE = "likely_unsafe"
    UNSAFE = "unsafe"


@dataclass
class CompatibilityResult:
    level: CompatibilityLevel
    warnings: list[str] = field(default_factory=list)

    @property
    def message(self) -> str:
        if not self.warnings:
            return f"Fork compatibility: {self.level.value}"
        return f"Fork compatibility: {self.level.value}\n" + "\n".join(
            f"  - {w}" for w in self.warnings
        )


class CacheCompatibilityChecker:
    def check(self, cache_safe: CacheSafeParams, policy: ForkPolicy) -> CompatibilityResult:
        warnings: list[str] = []

        if policy.max_output_tokens is not None:
            warnings.append(
                "max_output_tokens override may alter reasoning budget "
                "and invalidate cache sharing"
            )

        if not policy.cache_safe:
            return CompatibilityResult(
                level=CompatibilityLevel.UNSAFE,
                warnings=warnings or ["policy.cache_safe is False"],
            )

        if warnings:
            return CompatibilityResult(
                level=CompatibilityLevel.LIKELY_UNSAFE,
                warnings=warnings,
            )

        return CompatibilityResult(level=CompatibilityLevel.SAFE)
