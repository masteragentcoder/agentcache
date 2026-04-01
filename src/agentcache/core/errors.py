from __future__ import annotations


class AgentCacheError(Exception):
    """Base exception for all agentcache errors."""


class CacheUnsafeForkError(AgentCacheError):
    """Raised when a fork would violate cache-safety constraints."""


class ProviderError(AgentCacheError):
    """Raised on provider communication failures."""


class CompactionError(AgentCacheError):
    """Raised when compaction cannot satisfy the target."""


class MemoryStoreError(AgentCacheError):
    """Raised on memory store read/write failures."""
