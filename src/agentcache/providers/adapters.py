"""Conversion between agentcache internal types and LiteLLM/OpenAI wire format."""

from __future__ import annotations

from typing import Any

from agentcache.core.messages import (
    Message,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
)
from agentcache.core.tools import ToolSpec
from agentcache.core.usage import Usage
from agentcache.providers.base import ProviderResponse, ReasoningConfig


def message_to_openai(msg: Message) -> dict[str, Any]:
    if msg.role == "tool":
        results = msg.tool_results
        if results:
            return {
                "role": "tool",
                "tool_call_id": results[0].tool_call_id,
                "content": str(results[0].result),
            }

    content: list[dict[str, Any]] | str
    tool_calls: list[dict[str, Any]] = []

    text_parts = [b.text for b in msg.blocks if isinstance(b, TextBlock)]
    for b in msg.blocks:
        if isinstance(b, ToolCallBlock):
            tool_calls.append({
                "id": b.id,
                "type": "function",
                "function": {"name": b.name, "arguments": _serialize_args(b.arguments)},
            })

    content = "\n".join(text_parts) if text_parts else ""

    result: dict[str, Any] = {"role": msg.role, "content": content}
    if tool_calls:
        result["tool_calls"] = tool_calls
    return result


def _serialize_args(args: dict[str, Any]) -> str:
    import orjson

    return orjson.dumps(args).decode()


def build_litellm_payload(
    *,
    model: str,
    system_prompt: str | list[dict[str, Any]],
    messages: list[Message],
    tools: list[ToolSpec] | None = None,
    reasoning: ReasoningConfig | None = None,
    metadata: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    wire_messages: list[dict[str, Any]] = []

    if isinstance(system_prompt, str) and system_prompt:
        wire_messages.append({"role": "system", "content": system_prompt})
    elif isinstance(system_prompt, list):
        wire_messages.append({"role": "system", "content": system_prompt})

    for msg in messages:
        wire_messages.append(message_to_openai(msg))

    payload: dict[str, Any] = {"model": model, "messages": wire_messages}

    if tools:
        payload["tools"] = [t.to_openai_tool() for t in tools]

    if reasoning and reasoning.enabled:
        if reasoning.effort:
            payload["reasoning_effort"] = reasoning.effort
        if reasoning.budget_tokens:
            payload.setdefault("extra_body", {})
            payload["extra_body"]["reasoning"] = {"budget_tokens": reasoning.budget_tokens}

    if metadata:
        payload["metadata"] = metadata

    if extra_body:
        payload.setdefault("extra_body", {})
        payload["extra_body"].update(extra_body)

    return payload


def _extract_cache_read_tokens(raw_usage: Any) -> int:
    """Extract cache-read tokens from whichever field the provider uses.

    - Anthropic / LiteLLM native: raw_usage.cache_read_input_tokens
    - OpenAI: raw_usage.prompt_tokens_details.cached_tokens
    - Gemini via LiteLLM: raw_usage.cache_read_input_tokens (extra field)
    """
    # Anthropic-style / Gemini-style top-level field
    direct = getattr(raw_usage, "cache_read_input_tokens", None)
    if direct is not None and direct > 0:
        return direct

    # OpenAI-style nested field
    details = getattr(raw_usage, "prompt_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", None)
        if cached is not None and cached > 0:
            return cached

    return 0


def normalize_litellm_response(raw: Any) -> ProviderResponse:
    choice = raw.choices[0]
    raw_msg = choice.message

    blocks: list[TextBlock | ToolCallBlock | ToolResultBlock] = []
    if raw_msg.content:
        blocks.append(TextBlock(text=raw_msg.content))

    if hasattr(raw_msg, "tool_calls") and raw_msg.tool_calls:
        import orjson

        for tc in raw_msg.tool_calls:
            args = tc.function.arguments
            parsed = orjson.loads(args) if isinstance(args, str) else args
            blocks.append(ToolCallBlock(id=tc.id, name=tc.function.name, arguments=parsed))

    message = Message(role="assistant", blocks=blocks)

    raw_usage = raw.usage
    cache_read = _extract_cache_read_tokens(raw_usage)
    cache_create = getattr(raw_usage, "cache_creation_input_tokens", 0) or 0

    usage = Usage(
        input_tokens=getattr(raw_usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(raw_usage, "completion_tokens", 0) or 0,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_create,
        total_tokens=getattr(raw_usage, "total_tokens", 0) or 0,
    )

    return ProviderResponse(
        message=message,
        usage=usage,
        model=getattr(raw, "model", ""),
        request_id=getattr(raw, "id", None),
        raw=raw,
        stop_reason=getattr(choice, "finish_reason", None),
    )
