from __future__ import annotations

import secrets


def new_session_id() -> str:
    return "sess_" + secrets.token_hex(8)


def new_agent_id() -> str:
    return "agent_" + secrets.token_hex(8)
