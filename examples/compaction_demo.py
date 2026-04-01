"""Microcompaction demo: shows how agentcache trims bloated context
while preserving cache-safe prefix bytes.

This example simulates a coding-assistant session where tool results
(file reads, shell output) accumulate and bloat the context window.
It demonstrates both:
  1. ToolResultBudgeter -- replaces oversized tool results with placeholders
  2. MicroCompactor -- policy-driven clearing of stale results and thinking blocks

No API key needed -- this runs entirely locally on synthetic messages.
"""

from agentcache.compact.microcompact import MicroCompactor
from agentcache.compact.policy import CompactPolicy
from agentcache.compact.tool_budget import ReplacementState, ToolResultBudgeter
from agentcache.core.messages import (
    Message,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
)


def estimate_tokens(messages: list[Message]) -> int:
    return sum(m.token_estimate() for m in messages)


def demo_tool_result_budgeter():
    print("=" * 60)
    print("DEMO 1: Tool-result budgeting")
    print("=" * 60)
    print()

    state = ReplacementState()
    budgeter = ToolResultBudgeter(per_turn_budget_tokens=2000)

    huge_file = "x" * 40_000  # ~10,000 tokens
    small_file = "y" * 2_000  # ~500 tokens

    messages = [
        Message.user("Read the big config file"),
        Message(role="assistant", blocks=[
            TextBlock("I'll read that file for you."),
            ToolCallBlock(id="tc_1", name="file_read", arguments={"path": "big_config.json"}),
        ]),
        Message.tool_result("tc_1", huge_file),
        Message.user("Now read the small helper"),
        Message(role="assistant", blocks=[
            TextBlock("Reading the small file."),
            ToolCallBlock(id="tc_2", name="file_read", arguments={"path": "helper.py"}),
        ]),
        Message.tool_result("tc_2", small_file),
    ]

    before = estimate_tokens(messages)
    trimmed = budgeter.enforce(messages, state)
    after = estimate_tokens(trimmed)

    print(f"  Before budgeting:  ~{before:,} est tokens")
    print(f"  After budgeting:   ~{after:,} est tokens")
    print(f"  Saved:             ~{before - after:,} est tokens")
    print()

    for msg in trimmed:
        for block in msg.blocks:
            if isinstance(block, ToolResultBlock):
                preview = str(block.result)[:80]
                print(f"  tool_result [{block.tool_call_id}]: {preview}...")
    print()

    print("  Key point: ReplacementState recorded which tool_call_ids were truncated.")
    print(f"  Tracked replacements: {list(state.replacements.keys())}")
    print()

    cloned = state.clone()
    print("  After fork (cloned state), the same replacements apply to the")
    print("  shared prefix -- so prefix bytes stay identical = cache hit.")
    print(f"  Cloned replacements: {list(cloned.replacements.keys())}")
    print()


def demo_microcompactor():
    print("=" * 60)
    print("DEMO 2: Policy-driven microcompaction")
    print("=" * 60)
    print()

    messages: list[Message] = []

    for i in range(12):
        messages.append(Message.user(f"Turn {i+1}: do something with the codebase"))

        blocks: list = [TextBlock(f"<thinking>Internal reasoning for turn {i+1}... " + "t" * 2000 + "</thinking>")]
        blocks.append(TextBlock(f"Here's what I found for turn {i+1}."))
        blocks.append(ToolCallBlock(id=f"tc_shell_{i}", name="shell", arguments={"cmd": f"grep -r pattern_{i}"}))
        messages.append(Message(role="assistant", blocks=blocks))

        messages.append(Message.tool_result(f"tc_shell_{i}", f"Shell output for turn {i+1}: " + "o" * 4000))

    before = estimate_tokens(messages)
    print(f"  Simulated session: 12 turns of coding-assistant work")
    print(f"  Estimated tokens:  ~{before:,}")
    print()

    policy = CompactPolicy(
        max_input_tokens=5_000,
        preserve_last_turns=6,
        clear_thinking=True,
        keep_recent_thinking_turns=1,
    )

    compactor = MicroCompactor()
    result = compactor.compact_if_needed(messages, policy)
    after = estimate_tokens(result.messages)

    print(f"  Policy: max_input_tokens=5,000 | preserve_last_turns=6 | clear_thinking=True")
    print()
    print(f"  Before compaction: ~{before:,} est tokens")
    print(f"  After compaction:  ~{after:,} est tokens")
    print(f"  Removed:           ~{result.removed_tokens:,} est tokens")
    print()

    if result.actions:
        print("  Actions taken:")
        for action in result.actions:
            print(f"    - {action}")
    print()

    thinking_count = 0
    for msg in result.messages:
        for block in msg.blocks:
            if isinstance(block, TextBlock) and block.text.startswith("<thinking>"):
                thinking_count += 1
    print(f"  Thinking blocks remaining: {thinking_count} (only the most recent)")
    print()


def demo_preview():
    print("=" * 60)
    print("DEMO 3: compact_preview() on a session")
    print("=" * 60)
    print()

    from agentcache.core.session import AgentSession
    from agentcache.providers.base import Provider

    class FakeProvider(Provider):
        async def complete(self, **kwargs):
            pass

    session = AgentSession(
        model="gpt-4o-mini",
        provider=FakeProvider(),
        system_prompt="You are a coding assistant.",
        compact_policy=CompactPolicy(
            max_input_tokens=3_000,
            preserve_last_turns=4,
            clear_thinking=True,
        ),
    )

    for i in range(8):
        session.messages.append(Message.user(f"Question {i+1}"))
        blocks = [
            TextBlock(f"<thinking>Reasoning step {i+1}... " + "r" * 1500 + "</thinking>"),
            TextBlock(f"Answer to question {i+1}."),
            ToolCallBlock(id=f"tc_{i}", name="grep", arguments={"pattern": f"term_{i}"}),
        ]
        session.messages.append(Message(role="assistant", blocks=blocks))
        session.messages.append(Message.tool_result(f"tc_{i}", "grep result: " + "g" * 3000))

    print("  Session with 8 turns of grep + thinking:")
    print()
    print(f"  {session.compact_preview()}")
    print()


if __name__ == "__main__":
    demo_tool_result_budgeter()
    demo_microcompactor()
    demo_preview()
    print("=" * 60)
    print("All demos complete.")
    print("=" * 60)
