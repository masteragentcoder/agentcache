"""Tests for MicroCompactor and ToolResultBudgeter."""

from __future__ import annotations

from agentcache.compact.microcompact import MicroCompactor
from agentcache.compact.policy import CompactPolicy
from agentcache.compact.tool_budget import ReplacementState, ToolResultBudgeter
from agentcache.core.messages import Message, TextBlock, ToolCallBlock, ToolResultBlock


def _make_big_messages(n: int, result_size: int = 1000) -> list[Message]:
    """Create n tool-result messages with large payloads."""
    msgs: list[Message] = []
    for i in range(n):
        msgs.append(Message.user(f"Do thing {i}"))
        msgs.append(
            Message(
                role="assistant",
                blocks=[
                    TextBlock(text=f"Running tool {i}"),
                    ToolCallBlock(id=f"tc_{i}", name="grep", arguments={"q": "x"}),
                ],
            )
        )
        msgs.append(
            Message.tool_result(f"tc_{i}", "x" * result_size)
        )
        msgs.append(Message.assistant(f"Result for {i}."))
    return msgs


class TestMicroCompactor:
    def test_no_compaction_under_threshold(self):
        msgs = [Message.user("hello"), Message.assistant("hi")]
        policy = CompactPolicy(max_input_tokens=100_000)
        compactor = MicroCompactor()

        result = compactor.compact_if_needed(msgs, policy)
        assert result.removed_tokens == 0
        assert len(result.messages) == 2

    def test_compaction_clears_old_tool_results(self):
        msgs = _make_big_messages(50, result_size=20000)
        policy = CompactPolicy(
            max_input_tokens=100,
            target_input_tokens=50,
            preserve_last_turns=4,
        )
        compactor = MicroCompactor()
        result = compactor.compact_if_needed(msgs, policy)

        assert result.removed_tokens > 0
        assert len(result.actions) > 0
        assert any("tool results" in a for a in result.actions)

    def test_preview_output(self):
        msgs = _make_big_messages(50, result_size=20000)
        policy = CompactPolicy(max_input_tokens=100, preserve_last_turns=4)
        compactor = MicroCompactor()

        preview_text = compactor.preview(msgs, policy)
        assert "Estimated tokens before:" in preview_text
        assert "Estimated tokens after:" in preview_text


class TestToolResultBudgeter:
    def test_small_results_unchanged(self):
        msgs = [
            Message.user("run grep"),
            Message.tool_result("tc_1", "small result"),
            Message.assistant("done"),
        ]
        state = ReplacementState()
        budgeter = ToolResultBudgeter(per_turn_budget_tokens=8000)

        result = budgeter.enforce(msgs, state)
        tool_msg = [m for m in result if m.role == "tool"][0]
        assert tool_msg.tool_results[0].result == "small result"
        assert len(state.replacements) == 0

    def test_large_results_truncated(self):
        big_content = "x" * 100_000
        msgs = [
            Message.user("run grep"),
            Message.tool_result("tc_big", big_content),
            Message.assistant("done"),
        ]
        state = ReplacementState()
        budgeter = ToolResultBudgeter(per_turn_budget_tokens=1000)

        result = budgeter.enforce(msgs, state)
        tool_msg = [m for m in result if m.role == "tool"][0]
        assert "[result truncated" in str(tool_msg.tool_results[0].result)
        assert "tc_big" in state.replacements

    def test_replacement_state_reapplied(self):
        """Once a result is replaced, future calls reuse the replacement."""
        big_content = "x" * 100_000
        msgs = [
            Message.tool_result("tc_re", big_content),
        ]
        state = ReplacementState()
        budgeter = ToolResultBudgeter(per_turn_budget_tokens=1000)

        budgeter.enforce(msgs, state)
        assert "tc_re" in state.replacements

        msgs2 = [
            Message.tool_result("tc_re", big_content),
        ]
        result2 = budgeter.enforce(msgs2, state)
        tool_msg = [m for m in result2 if m.role == "tool"][0]
        assert "[result truncated" in str(tool_msg.tool_results[0].result)

    def test_none_state_passthrough(self):
        msgs = [Message.user("hello")]
        budgeter = ToolResultBudgeter()
        result = budgeter.enforce(msgs, None)
        assert result is msgs
