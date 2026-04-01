"""Tests for prompt-state tracking and cache-break detection."""

from __future__ import annotations

from agentcache.cache.explain import CacheBreakExplanation, explain_break
from agentcache.cache.prompt_state import (
    PromptStateDiff,
    PromptStateSnapshot,
    PromptStateSnapshotFactory,
    diff_prompt_states,
)
from agentcache.cache.tracker import PromptStateTracker
from agentcache.core.usage import Usage


def _make_snapshot(**overrides) -> PromptStateSnapshot:
    defaults = {
        "system_hash": "abc123",
        "tools_hash": "def456",
        "cache_control_hash": "ghi789",
        "model": "test-model",
        "reasoning_hash": "jkl012",
        "effort_value": None,
        "extra_body_hash": None,
        "beta_flags": (),
        "system_char_count": 100,
    }
    defaults.update(overrides)
    return PromptStateSnapshot(**defaults)


class TestPromptStateDiff:
    def test_identical_snapshots_no_diff(self):
        a = _make_snapshot()
        b = _make_snapshot()
        diff = diff_prompt_states(a, b)
        assert not diff.has_changes

    def test_system_change_detected(self):
        a = _make_snapshot(system_hash="aaa")
        b = _make_snapshot(system_hash="bbb", system_char_count=200)
        diff = diff_prompt_states(a, b)
        assert diff.system_changed
        assert diff.system_char_delta == 100

    def test_model_change_detected(self):
        a = _make_snapshot(model="claude-3")
        b = _make_snapshot(model="claude-4")
        diff = diff_prompt_states(a, b)
        assert diff.model_changed

    def test_tools_change_detected(self):
        a = _make_snapshot(tools_hash="t1")
        b = _make_snapshot(tools_hash="t2")
        diff = diff_prompt_states(a, b)
        assert diff.tools_changed

    def test_cache_control_change_detected(self):
        a = _make_snapshot(cache_control_hash="cc1")
        b = _make_snapshot(cache_control_hash="cc2")
        diff = diff_prompt_states(a, b)
        assert diff.cache_control_changed

    def test_effort_change_detected(self):
        a = _make_snapshot(effort_value="high")
        b = _make_snapshot(effort_value="low")
        diff = diff_prompt_states(a, b)
        assert diff.effort_changed


class TestExplainBreak:
    def test_system_prompt_cause(self):
        diff = PromptStateDiff(system_changed=True, system_char_delta=321)
        explanation = explain_break(diff, previous=142880, current=22110)
        assert explanation.broke
        assert any("system prompt" in c for c in explanation.causes)
        assert "+321 chars" in explanation.causes[0]
        assert explanation.previous_cache_read_tokens == 142880
        assert explanation.current_cache_read_tokens == 22110

    def test_tool_schema_cause(self):
        diff = PromptStateDiff(tools_changed=True)
        explanation = explain_break(diff, previous=100000, current=10000)
        assert any("tool schema" in c for c in explanation.causes)

    def test_no_diff_suggests_ttl(self):
        diff = PromptStateDiff()
        explanation = explain_break(diff, previous=100000, current=10000)
        assert not explanation.causes
        assert any("TTL" in n for n in explanation.notes)

    def test_no_diff_no_causes(self):
        explanation = explain_break(diff=None, previous=50000, current=5000)
        assert explanation.broke
        assert not explanation.causes
        assert any("TTL" in n for n in explanation.notes)


class TestPromptStateTracker:
    def test_no_break_on_first_call(self):
        tracker = PromptStateTracker()
        snap = _make_snapshot()
        tracker.record_pre_call(snap)
        tracker.record_post_call(Usage(cache_read_input_tokens=80000))
        assert tracker.last_explanation is None

    def test_break_detected_on_significant_drop(self):
        tracker = PromptStateTracker(min_drop_tokens=1000)
        snap1 = _make_snapshot(system_hash="v1")
        tracker.record_pre_call(snap1)
        tracker.record_post_call(Usage(cache_read_input_tokens=100000))

        snap2 = _make_snapshot(system_hash="v2", system_char_count=200)
        tracker.record_pre_call(snap2)
        tracker.record_post_call(Usage(cache_read_input_tokens=5000))

        assert tracker.last_explanation is not None
        assert tracker.last_explanation.broke
        assert any("system prompt" in c for c in tracker.last_explanation.causes)

    def test_no_false_positive_on_small_drop(self):
        tracker = PromptStateTracker(min_drop_tokens=5000)
        snap = _make_snapshot()
        tracker.record_pre_call(snap)
        tracker.record_post_call(Usage(cache_read_input_tokens=100000))

        tracker.record_pre_call(snap)
        tracker.record_post_call(Usage(cache_read_input_tokens=99000))

        assert tracker.last_explanation is None

    def test_compaction_resets_baseline(self):
        tracker = PromptStateTracker(min_drop_tokens=1000)
        snap = _make_snapshot()
        tracker.record_pre_call(snap)
        tracker.record_post_call(Usage(cache_read_input_tokens=100000))

        tracker.notify_compaction()
        tracker.record_pre_call(snap)
        tracker.record_post_call(Usage(cache_read_input_tokens=5000))

        assert tracker.last_explanation is None


class TestSnapshotFactory:
    def test_from_request_basic(self):
        snap = PromptStateSnapshotFactory.from_request(
            system_prompt="Hello",
            model="test",
        )
        assert snap.model == "test"
        assert snap.system_char_count == 5
        assert snap.effort_value is None
