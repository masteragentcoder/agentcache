"""Quick smoke test with a real provider."""

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from agentcache import AgentSession, ForkPolicy, LiteLLMSDKProvider


async def main():
    provider = LiteLLMSDKProvider()

    # Pick model based on available keys
    if os.getenv("GEMINI_API_KEY"):
        model = "gemini/gemini-2.0-flash"
    elif os.getenv("OPENAI_API_KEY"):
        model = "gpt-4o-mini"
    else:
        print("No API key found in .env")
        return

    print(f"Using model: {model}")
    print("=" * 50)

    # 1. Basic respond
    session = AgentSession(
        model=model,
        provider=provider,
        system_prompt="You are a concise assistant. Reply in 1-2 sentences max.",
    )

    print("\n[1] Testing session.respond()...")
    reply = await session.respond("What is prompt caching and why does it matter?")
    print(f"Reply: {reply.text}")
    print(f"Usage: {reply.usage}")

    # 2. Cache status
    print("\n[2] Cache status:")
    status = session.cache_status()
    print(status.pretty())

    # 3. Cache-safe fork
    print("\n[3] Testing session.fork() (cache-safe ephemeral)...")
    summary = await session.fork(
        prompt="Repeat the key point from the conversation in one bullet.",
        policy=ForkPolicy.cache_safe_ephemeral(),
    )
    print(f"Fork result: {summary.final_text}")
    print(f"Fork usage: {summary.usage}")
    print(f"Fork turns: {summary.turns_used}")

    # 4. Second respond to see cache tracking
    print("\n[4] Second respond (cache tracking)...")
    reply2 = await session.respond("How does this relate to LLM agent orchestration?")
    print(f"Reply: {reply2.text}")

    status2 = session.cache_status()
    print(f"\nUpdated cache status:")
    print(status2.pretty())

    # 5. Check for cache break explanation
    explanation = session.explain_last_cache_break()
    if explanation:
        print(f"\nCache break: {explanation.pretty()}")
    else:
        print("\nNo cache break detected (good).")

    # 6. Compact preview
    print("\n[6] Compact preview:")
    print(session.compact_preview())

    print("\n" + "=" * 50)
    print("Smoke test complete!")


if __name__ == "__main__":
    asyncio.run(main())
