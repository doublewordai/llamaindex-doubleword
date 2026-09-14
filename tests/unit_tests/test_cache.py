import copy
import json
from typing import Any

import httpx
import pytest
from llama_index.core.base.llms.types import ChatMessage
from pydantic import ValidationError

from llamaindex_doubleword import DoublewordLLM, DoublewordLLMAsync, DoublewordLLMBatch
from llamaindex_doubleword._cache import apply_cache_control

EPHEMERAL = {"type": "ephemeral"}
ONE_HOUR = {"type": "ephemeral", "ttl": "1h"}
FIVE_MINUTES = {"type": "ephemeral", "ttl": "5m"}
COMPLETION = {
    "id": "c",
    "object": "chat.completion",
    "created": 0,
    "model": "m",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
    ],
}
CHUNK = {
    "id": "c",
    "object": "chat.completion.chunk",
    "created": 0,
    "model": "m",
    "choices": [
        {"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
    ],
}


def _marked(text: str, cache_control: dict[str, str]) -> list[dict[str, Any]]:
    return [{"type": "text", "text": text, "cache_control": cache_control}]


def _contents(messages: list[dict[str, Any]]) -> list[Any]:
    return [message["content"] for message in messages]


def _llm(bodies: list[dict[str, Any]], **kwargs: Any) -> DoublewordLLM:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if body.get("stream"):
            sse = f"data: {json.dumps(CHUNK)}\n\ndata: [DONE]\n\n".encode()
            return httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=COMPLETION)

    transport = httpx.MockTransport(handler)
    return DoublewordLLM(
        model="m",
        api_key="x",
        api_base="https://test/v1",
        max_retries=0,
        http_client=httpx.Client(transport=transport),
        async_http_client=httpx.AsyncClient(transport=transport),
        **kwargs,
    )


async def _call(llm: DoublewordLLM, method: str, **kwargs: Any) -> None:
    prompt: Any = [ChatMessage(role="user", content="q")] if "chat" in method else "q"
    result = getattr(llm, method)(prompt, **kwargs)
    if method.startswith("a"):
        result = await result
    if method.startswith("astream"):
        async for _ in result:
            pass
    elif method.startswith("stream"):
        for _ in result:
            pass


def test_marks_last_system_and_latest_message() -> None:
    messages = [
        {"role": "system", "content": "old"},
        {"role": "system", "content": "stable"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    apply_cache_control({"messages": messages}, EPHEMERAL)
    assert _contents(messages) == [
        "old",
        _marked("stable", EPHEMERAL),
        "q1",
        "a1",
        _marked("q2", EPHEMERAL),
    ]


def test_system_that_is_also_latest_is_marked_once() -> None:
    messages = [{"role": "system", "content": "stable"}]
    apply_cache_control({"messages": messages}, EPHEMERAL)
    assert _contents(messages) == [_marked("stable", EPHEMERAL)]


def test_without_system_only_latest_is_marked() -> None:
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    apply_cache_control({"messages": messages}, EPHEMERAL)
    assert _contents(messages) == ["q1", "a1", _marked("q2", EPHEMERAL)]


def test_omitted_ttl_sends_no_ttl_key() -> None:
    messages = [{"role": "user", "content": "q"}]
    apply_cache_control({"messages": messages}, {"type": "ephemeral"})
    assert messages[0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_ttl_passes_through() -> None:
    messages = [{"role": "user", "content": "q"}]
    apply_cache_control({"messages": messages}, {"type": "ephemeral", "ttl": "1h"})
    assert messages[0]["content"] == _marked("q", ONE_HOUR)


def test_marks_last_text_block() -> None:
    image = {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}, image],
        }
    ]
    apply_cache_control({"messages": messages}, EPHEMERAL)
    assert messages[0]["content"] == [{"type": "text", "text": "a"}, *_marked("b", EPHEMERAL), image]


def test_skips_target_without_text() -> None:
    call = {"id": "1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    messages = [
        {"role": "system", "content": "stable"},
        {"role": "assistant", "content": None, "tool_calls": [call]},
    ]
    apply_cache_control({"messages": messages}, EPHEMERAL)
    assert _contents(messages) == [_marked("stable", EPHEMERAL), None]


def test_existing_markers_are_untouched() -> None:
    messages = [
        {"role": "system", "content": _marked("stable", FIVE_MINUTES)},
        {
            "role": "user",
            "content": [*_marked("a", FIVE_MINUTES), {"type": "text", "text": "b"}],
        },
    ]
    before = copy.deepcopy(messages)
    apply_cache_control({"messages": messages}, ONE_HOUR)
    assert messages == before


@pytest.mark.parametrize(
    ("existing", "expected"),
    [(2, [True, True]), (3, [True, False]), (4, [False, False])],
)
def test_never_exceeds_four_breakpoints(existing: int, expected: list[bool]) -> None:
    history = [
        {"type": "text", "text": str(i), "cache_control": FIVE_MINUTES} for i in range(existing)
    ]
    messages = [
        {"role": "system", "content": "stable"},
        {"role": "user", "content": history},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "q"},
    ]
    apply_cache_control({"messages": messages}, EPHEMERAL)
    assert [isinstance(messages[i]["content"], list) for i in (0, 3)] == expected


def test_tool_markers_count_toward_the_limit() -> None:
    bodies: list[dict[str, Any]] = []
    llm = _llm(bodies, cache_control=EPHEMERAL)
    tools = [
        {"type": "function", "function": {"name": f"t{i}"}, "cache_control": FIVE_MINUTES}
        for i in range(3)
    ]
    llm.chat(
        [ChatMessage(role="system", content="stable"), ChatMessage(role="user", content="q")],
        tools=tools,
    )
    assert bodies[0]["tools"] == tools
    assert _contents(bodies[0]["messages"]) == [_marked("stable", EPHEMERAL), "q"]


@pytest.mark.parametrize(
    "method",
    [
        "chat",
        "complete",
        "stream_chat",
        "stream_complete",
        "achat",
        "acomplete",
        "astream_chat",
        "astream_complete",
    ],
)
async def test_per_call_value_overrides_the_field(method: str) -> None:
    bodies: list[dict[str, Any]] = []
    llm = _llm(bodies, cache_control=EPHEMERAL)
    await _call(llm, method)
    await _call(llm, method, cache_control=ONE_HOUR)
    await _call(llm, method, cache_control=None)
    assert [body["messages"][0]["content"] for body in bodies] == [
        _marked("q", EPHEMERAL),
        _marked("q", ONE_HOUR),
        "q",
    ]
    assert not any("cache_control" in body for body in bodies)


def test_unset_field_sends_messages_untouched() -> None:
    bodies: list[dict[str, Any]] = []
    llm = _llm(bodies)
    llm.chat([ChatMessage(role="user", content="q")])
    llm.chat([ChatMessage(role="user", content="q")], cache_control=EPHEMERAL)
    assert [body["messages"][0]["content"] for body in bodies] == ["q", _marked("q", EPHEMERAL)]
    assert not any("cache_control" in body for body in bodies)


def test_extra_body_composes_with_additional_kwargs() -> None:
    bodies: list[dict[str, Any]] = []
    llm = _llm(bodies, cache_control=EPHEMERAL, additional_kwargs={"extra_body": {"top_k": 5}})
    llm.chat([ChatMessage(role="user", content="q")])
    assert bodies[0]["top_k"] == 5
    assert bodies[0]["messages"][0]["content"] == _marked("q", EPHEMERAL)


@pytest.mark.parametrize("model_class", [DoublewordLLMBatch, DoublewordLLMAsync])
def test_batch_variants_inherit_the_hook(model_class: type[DoublewordLLMBatch]) -> None:
    llm = model_class(model="m", api_key="x", cache_control=EPHEMERAL)
    kwargs = llm._cache_kwargs([ChatMessage(role="user", content="q")], {})
    assert kwargs["extra_body"]["messages"][0]["content"] == _marked("q", EPHEMERAL)


def test_rejects_an_unknown_ttl() -> None:
    with pytest.raises(ValidationError):
        DoublewordLLM(model="m", api_key="x", cache_control={"type": "ephemeral", "ttl": "2h"})
