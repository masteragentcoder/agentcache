"""Test with enough tokens to trigger automatic prompt caching.

OpenAI: needs >= 1,024 token prefix, caches in 128-token increments
Gemini 2.5+: needs >= 1,024 token prefix (Flash) or 4,096 (Pro), implicit caching
Gemini 2.0: no implicit caching
"""

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from agentcache import AgentSession, ForkPolicy, LiteLLMSDKProvider


_PADDING = "\n".join(
    f"- Architecture note #{i}: Cache-aware orchestration requires that every forked "
    f"subagent inherits the exact same system prompt, tool schema, model identifier, "
    f"reasoning configuration, and cache-control metadata as the parent session. "
    f"Deviation in any of these fields will cause a prompt-cache miss. (item {i})"
    for i in range(40)
)

LARGE_SYSTEM_PROMPT = (
    "You are a senior software architect specializing in distributed systems, "
    "caching strategies, and LLM infrastructure. You have deep expertise in: "
    "prompt caching mechanisms across providers (Anthropic, OpenAI, Google), "
    "cache-safe agent forking patterns, context window management, "
    "microcompaction strategies for long-running conversations, "
    "session memory extraction and consolidation, "
    "coordinator/worker orchestration with summarized handoffs, "
    "and supply-chain security for Python ML dependencies.\n\n"
    "When answering questions, follow these guidelines:\n"
    "1. Be precise about token counts and cache behavior\n"
    "2. Distinguish between provider-specific and universal patterns\n"
    "3. Always consider cost implications of cache misses\n"
    "4. Recommend cache-safe defaults over flexible-but-expensive options\n"
    "5. Explain trade-offs between isolation and cache sharing\n\n"
    "Context about the agentcache library:\n"
    "The agentcache library provides cache-aware orchestration for LLM agents. "
    "Its core insight is that helpers should reuse the parent's cacheable prefix "
    "while running in an isolated mutable context. The library tracks prompt-state "
    "snapshots, detects cache breaks, applies microcompaction before expensive calls, "
    "extracts session memory via forked helpers, and supports coordinator/worker "
    "orchestration with summarized handoffs.\n\n"
    f"Extended reference material:\n{_PADDING}\n\n"
    "Reply concisely in 1-2 sentences."
)


async def test_model(model: str, label: str):
    print(f"\n{'='*60}")
    print(f"  {label}: {model}")
    print(f"  System prompt: ~{len(LARGE_SYSTEM_PROMPT)} chars")
    print(f"{'='*60}")

    provider = LiteLLMSDKProvider()
    session = AgentSession(
        model=model,
        provider=provider,
        system_prompt=LARGE_SYSTEM_PROMPT,
    )

    # Call 1 -- should create cache
    print("\n--- Call 1 (cache creation) ---")
    r1 = await session.respond("What is the minimum prefix for caching to activate?")
    print(f"Reply: {r1.text}")
    print(f"Normalized usage: {r1.usage}")
    _dump_raw_cache(r1)

    # Call 2 -- same prefix, should hit cache
    print("\n--- Call 2 (should hit cache) ---")
    r2 = await session.respond("How does ReplacementState cloning prevent cache misses?")
    print(f"Reply: {r2.text}")
    print(f"Normalized usage: {r2.usage}")
    _dump_raw_cache(r2)

    # Call 3 -- fork (same prefix)
    print("\n--- Call 3 (fork, same prefix) ---")
    fork = await session.fork(
        "Summarize the conversation so far in 1 bullet.",
        policy=ForkPolicy.cache_safe_ephemeral(),
    )
    print(f"Fork: {fork.final_text}")
    print(f"Fork usage: {fork.usage}")

    print(f"\nFinal cache status:\n{session.cache_status().pretty()}")

    explanation = session.explain_last_cache_break()
    if explanation:
        print(f"\n{explanation.pretty()}")


def _dump_raw_cache(response):
    u = response.raw.usage
    details = getattr(u, "prompt_tokens_details", None)
    print(f"  raw.prompt_tokens_details: {details}")
    print(f"  raw.cache_read_input_tokens: {getattr(u, 'cache_read_input_tokens', 'N/A')}")
    print(f"  raw.cache_creation_input_tokens: {getattr(u, 'cache_creation_input_tokens', 'N/A')}")


async def main():
    if os.getenv("GEMINI_API_KEY"):
        await test_model("gemini/gemini-3-flash-preview", "Gemini 3 Flash Preview (implicit caching)")

    if os.getenv("OPENAI_API_KEY"):
        await test_model("gpt-4o-mini", "OpenAI gpt-4o-mini")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
