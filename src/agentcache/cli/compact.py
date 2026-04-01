"""CLI commands for context compaction."""

from __future__ import annotations

import typer

compact_app = typer.Typer(no_args_is_help=True)


@compact_app.command("preview")
def compact_preview() -> None:
    """Preview what compaction would remove.

    Requires an active session (not yet connected to a live session in v0.1).
    """
    typer.echo(
        "Compact preview requires an active session.\n"
        "Use the Python API: session.compact_preview()\n\n"
        "Example output:\n"
        "  Would remove:\n"
        "    - stale grep/file-read/web-fetch tool results\n"
        "    - old assistant thinking blocks\n"
        "  Estimated reduction: 96,000 tokens -> 38,000 tokens"
    )
