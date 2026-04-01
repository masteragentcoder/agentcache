"""CLI commands for cache diagnostics."""

from __future__ import annotations

import typer

cache_app = typer.Typer(no_args_is_help=True)


@cache_app.command("status")
def cache_status() -> None:
    """Show cache hit rate and last-call token stats.

    Requires an active session (not yet connected to a live session in v0.1).
    """
    typer.echo(
        "Cache status requires an active session.\n"
        "Use the Python API: session.cache_status()\n\n"
        "Example output:\n"
        "  Session: sess_f4a9c2\n"
        "  Cache read tokens (last call): 128,440\n"
        "  Cache write tokens (last call): 4,012\n"
        "  Hit rate (last 10 calls): 87.2%\n"
        "  Last break: none"
    )


@cache_app.command("explain-last-break")
def explain_last_break() -> None:
    """Explain the most recent cache break.

    Requires an active session (not yet connected to a live session in v0.1).
    """
    typer.echo(
        "Cache break explanation requires an active session.\n"
        "Use the Python API: session.explain_last_cache_break()\n\n"
        "Example output:\n"
        "  Cache break detected.\n"
        "  Primary causes:\n"
        "    - system prompt changed (+321 chars)\n"
        "    - tool schema changed\n"
        "  Previous cache-read tokens: 142,880\n"
        "  Current cache-read tokens: 22,110"
    )
