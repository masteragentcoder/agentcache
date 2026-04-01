"""Deterministic hashing for cache-state comparison."""

from __future__ import annotations

import hashlib
from typing import Any

import orjson


def stable_hash(value: Any) -> str:
    """Produce a deterministic hex digest for any JSON-serialisable value.

    Uses orjson for canonical serialisation (sorted keys, deterministic floats)
    then SHA-256 truncated to 16 hex chars for fast comparison.
    """
    if value is None:
        raw = b"__none__"
    elif isinstance(value, str):
        raw = value.encode("utf-8")
    elif isinstance(value, bytes):
        raw = value
    else:
        raw = orjson.dumps(value, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(raw).hexdigest()[:16]
