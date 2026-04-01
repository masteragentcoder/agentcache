"""CLI entry point for agentcache debugging and orchestration console."""

from __future__ import annotations

try:
    import typer
except ImportError:
    raise SystemExit(
        "CLI requires the 'cli' extra: pip install 'git+https://github.com/masterFoad/agentcache.git@main#egg=agentcache[cli]'"
    )

from agentcache.cli.cache import cache_app
from agentcache.cli.compact import compact_app

app = typer.Typer(
    name="agentcache",
    help="Cache-aware orchestration for LLM agents.",
    no_args_is_help=True,
)

app.add_typer(cache_app, name="cache", help="Cache diagnostics.")
app.add_typer(compact_app, name="compact", help="Context compaction tools.")


@app.command()
def doctor() -> None:
    """Show installed agentcache and LiteLLM versions."""
    from agentcache.version import __version__

    typer.echo(f"agentcache {__version__}")
    try:
        import litellm

        typer.echo(f"litellm   {litellm.version}")
    except Exception:
        typer.echo("litellm   (not installed or import error)")


def main() -> None:
    app()
