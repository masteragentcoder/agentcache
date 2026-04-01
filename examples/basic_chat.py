"""Basic agentcache usage: create a session, send a message, check cache status."""

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from agentcache import AgentSession, LiteLLMSDKProvider


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

    reply = await session.respond("Analyze my cache layer and suggest improvements.")
    print("Reply:", reply.text)
    print()

    status = session.cache_status()
    print(status.pretty())
    print()

    explanation = session.explain_last_cache_break()
    if explanation:
        print(explanation.pretty())
    else:
        print("No cache break detected.")


if __name__ == "__main__":
    asyncio.run(main())
