"""Test both providers and dump raw usage to see what cache fields come back."""

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from agentcache import AgentSession, ForkPolicy, LiteLLMSDKProvider
from agentcache.providers.adapters import build_litellm_payload, message_to_openai
from agentcache.core.messages import Message

import litellm


async def test_provider(model: str, label: str):
    print(f"\n{'='*60}")
    print(f"  {label}: {model}")
    print(f"{'='*60}")

    provider = LiteLLMSDKProvider()
    session = AgentSession(
        model=model,
        provider=provider,
        system_prompt="You are a concise assistant. Reply in 1 sentence max.",
    )

    # First call
    print("\n--- Call 1 ---")
    reply1 = await session.respond("What is prompt caching?")
    print(f"Reply: {reply1.text}")
    print(f"Raw usage type: {type(reply1.raw.usage)}")
    print(f"Raw usage dict: {dict(reply1.raw.usage)}" if hasattr(reply1.raw.usage, '__iter__') else f"Raw usage: {reply1.raw.usage}")

    # Print all attributes of raw usage
    raw_usage = reply1.raw.usage
    print(f"Usage attributes:")
    for attr in dir(raw_usage):
        if not attr.startswith('_'):
            try:
                val = getattr(raw_usage, attr)
                if not callable(val):
                    print(f"  {attr}: {val}")
            except Exception:
                pass

    print(f"\nNormalized: {reply1.usage}")

    # Second call (same prefix, should benefit from caching)
    print("\n--- Call 2 (same session, should cache) ---")
    reply2 = await session.respond("How does it save money?")
    print(f"Reply: {reply2.text}")

    raw_usage2 = reply2.raw.usage
    print(f"Usage attributes:")
    for attr in dir(raw_usage2):
        if not attr.startswith('_'):
            try:
                val = getattr(raw_usage2, attr)
                if not callable(val):
                    print(f"  {attr}: {val}")
            except Exception:
                pass

    print(f"\nNormalized: {reply2.usage}")
    print(f"Cache status: {session.cache_status().pretty()}")


async def main():
    if os.getenv("OPENAI_API_KEY"):
        await test_provider("gpt-4o-mini", "OpenAI")

    if os.getenv("GEMINI_API_KEY"):
        await test_provider("gemini/gemini-2.0-flash", "Gemini")

    print("\n\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
