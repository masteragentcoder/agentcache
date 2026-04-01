from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ForkPolicy:
    cache_safe: bool = True
    isolated: bool = True
    share_state: bool = False
    share_abort: bool = False
    skip_transcript: bool = False
    skip_cache_write: bool = False
    max_output_tokens: int | None = None
    max_turns: int | None = 1
    purpose: str = "helper"

    @classmethod
    def cache_safe_ephemeral(cls) -> ForkPolicy:
        return cls(
            cache_safe=True,
            isolated=True,
            share_state=False,
            share_abort=False,
            skip_transcript=True,
            skip_cache_write=True,
            max_turns=1,
            purpose="ephemeral_helper",
        )

    @classmethod
    def background_summary(cls) -> ForkPolicy:
        return cls(
            cache_safe=True,
            isolated=True,
            skip_transcript=True,
            skip_cache_write=True,
            max_turns=1,
            purpose="background_summary",
        )

    @classmethod
    def session_memory_update(cls) -> ForkPolicy:
        return cls(
            cache_safe=True,
            isolated=True,
            share_state=False,
            skip_transcript=False,
            skip_cache_write=False,
            max_turns=1,
            purpose="session_memory",
        )

    @classmethod
    def coord_worker(cls) -> ForkPolicy:
        return cls(
            cache_safe=True,
            isolated=True,
            share_state=False,
            skip_transcript=False,
            skip_cache_write=False,
            max_turns=None,
            purpose="coord_worker",
        )

    @classmethod
    def cache_unsafe_rewrite(cls) -> ForkPolicy:
        return cls(
            cache_safe=False,
            isolated=True,
            skip_transcript=False,
            skip_cache_write=False,
            purpose="cache_unsafe_rewrite",
        )
