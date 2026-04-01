"""Side-question example: fork a cache-safe helper after a main turn."""

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from agentcache import AgentSession, ForkPolicy, LiteLLMSDKProvider


async def main():
    provider = LiteLLMSDKProvider()

    if os.getenv("OPENAI_API_KEY"):
        model = "gpt-4o-mini"
    elif os.getenv("GEMINI_API_KEY"):
        model = "gemini/gemini-2.5-flash"
    elif os.getenv("ANTHROPIC_API_KEY"):
        model = "anthropic/claude-sonnet-4-20250514"
    else:
        print("Set OPENAI_API_KEY, GEMINI_API_KEY, or ANTHROPIC_API_KEY in .env")
        return

    session = AgentSession(
        model=model,
        provider=provider,
        system_prompt="You are a careful code assistant.",
    )

    # Main conversation turn
    reply = await session.respond("Refactor the caching layer and explain the tradeoffs.")
    print("Main reply:", reply.text[:200], "...")
    print()

    # Cache-safe side helper: shares parent prefix, isolated state, skips cache write
    summary = await session.fork(
        prompt="Summarize the current state in 3 bullets.",
        policy=ForkPolicy.cache_safe_ephemeral(),
    )
    print("Side helper summary:")
    print(summary.final_text)
    print()
    print(f"Side helper used {summary.usage.total_tokens} tokens across {summary.turns_used} turn(s)")
    print()

    # Cache diagnostics after both calls
    status = session.cache_status()
    print(status.pretty())


if __name__ == "__main__":
    asyncio.run(main())
