"""Basic agentcache usage: create a session, send a message, check cache status."""

import asyncio

from agentcache import AgentSession, LiteLLMSDKProvider


async def main():
    provider = LiteLLMSDKProvider()

    session = AgentSession(
        model="anthropic/claude-sonnet-4-20250514",
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
