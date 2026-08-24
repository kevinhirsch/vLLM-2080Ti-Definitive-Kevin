# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    RequestResponseMetadata,
)
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.tokenizers.mistral import MistralTokenizer
from vllm.tool_parsers.mistral_tool_parser import MistralToolCall

pytestmark = pytest.mark.skip_global_cleanup


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }
]
NAMED_CHOICE = {"type": "function", "function": {"name": "get_weather"}}
ARGUMENTS = '{"city": "Shanghai"}'


def _request(stream: bool) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": TOOLS,
            "tool_choice": NAMED_CHOICE,
            "stream": stream,
        }
    )


def _result(text: str, finish_reason: str | None) -> RequestOutput:
    return RequestOutput(
        request_id="named-tool",
        prompt="prompt",
        prompt_token_ids=[1, 2],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text=text,
                token_ids=[3],
                cumulative_logprob=None,
                logprobs=None,
                finish_reason=finish_reason,
                stop_reason=None,
            )
        ],
        finished=finish_reason is not None,
        num_cached_tokens=0,
    )


async def _results(texts: list[tuple[str, str | None]]):
    for text, finish_reason in texts:
        yield _result(text, finish_reason)


class _PlainContentParser:
    def __init__(self, *_args, **_kwargs):
        self.tool_parser = None
        self._stream_state = type("StreamState", (), {})()

    def parse(self, model_output, _request, enable_auto_tools=False):
        return None, model_output, []

    def parse_delta(self, delta_text, **_kwargs):
        return DeltaMessage(content=delta_text)


class _FakeMistralTokenizer(MistralTokenizer):
    pass


def _serving(parser_cls=None):
    instance = OpenAIServingChat.__new__(OpenAIServingChat)
    instance.use_harmony = False
    instance.tool_call_id_type = "random_uuid"
    instance.parser_cls = parser_cls
    instance.tool_parser = None
    instance.enable_auto_tools = False
    instance.enable_force_include_usage = False
    instance.response_role = "assistant"
    instance.enable_log_outputs = False
    instance.request_logger = None
    instance.enable_prompt_tokens_details = False
    instance.system_fingerprint = None
    instance.return_tokens_as_token_ids = False
    return instance


async def _full(text: str, parser=None, tokenizer=object(), finish_reason="stop"):
    request = _request(False)
    return await _serving().chat_completion_full_generator(
        request,
        _results([(text, finish_reason)]),
        "named-full",
        "test-model",
        request.messages,
        tokenizer,
        RequestResponseMetadata(request_id="named-full"),
        parser,
    )


async def _stream(
    texts: list[tuple[str, str | None]],
    tokenizer=object(),
):
    request = _request(True)
    chunks = []
    async for item in _serving(_PlainContentParser).chat_completion_stream_generator(
        request,
        _results(texts),
        "named-stream",
        "test-model",
        request.messages,
        tokenizer,
        RequestResponseMetadata(request_id="named-stream"),
    ):
        if item.startswith("data: ") and item.strip() != "data: [DONE]":
            chunks.append(json.loads(item[len("data: ") :].strip()))
    return [
        choice
        for chunk in chunks
        for choice in chunk.get("choices", [])
    ]


def test_named_tool_choice_full_wraps_plain_content():
    response = asyncio.run(_full(ARGUMENTS, _PlainContentParser()))
    choice = response.choices[0]

    assert choice.message.content is None
    assert choice.finish_reason == "tool_calls"
    assert choice.message.tool_calls is not None
    assert choice.message.tool_calls[0].function.name == "get_weather"
    assert choice.message.tool_calls[0].function.arguments == ARGUMENTS


def test_named_tool_choice_stream_wraps_plain_content():
    choices = asyncio.run(
        _stream([("{", None), ('"city": "Shanghai"}', "stop")])
    )
    tool_deltas = [
        choice["delta"]["tool_calls"][0]
        for choice in choices
        if choice["delta"].get("tool_calls")
    ]

    assert tool_deltas[0]["id"]
    assert tool_deltas[0]["type"] == "function"
    assert tool_deltas[0]["function"] == {
        "name": "get_weather",
        "arguments": "{",
    }
    assert tool_deltas[1]["function"]["arguments"] == '"city": "Shanghai"}'
    assert choices[-1]["finish_reason"] == "tool_calls"


def test_named_tool_choice_empty_full_does_not_create_tool_call():
    response = asyncio.run(_full("", None))
    choice = response.choices[0]

    assert choice.message.content is None
    assert not choice.message.tool_calls
    assert choice.finish_reason == "stop"


def test_named_tool_choice_empty_stream_does_not_create_tool_call():
    choices = asyncio.run(_stream([("", "stop")]))

    assert choices[-1]["finish_reason"] == "stop"
    assert not choices[-1]["delta"].get("tool_calls")


def test_named_tool_choice_mistral_stream_uses_mistral_id():
    tokenizer = _FakeMistralTokenizer.__new__(_FakeMistralTokenizer)
    choices = asyncio.run(
        _stream([("{", None), ('"city": "Shanghai"}', "stop")], tokenizer)
    )
    tool_delta = next(
        choice["delta"]["tool_calls"][0]
        for choice in choices
        if choice["delta"].get("tool_calls")
    )

    assert MistralToolCall.is_valid_id(tool_delta["id"])


def test_named_tool_choice_stream_truncated_preserves_length():
    # Truncated by max_tokens mid-arguments -> finish_reason "length". A
    # streamed named tool call must not flip the final finish_reason to
    # "tool_calls", or clients would execute a truncated argument blob.
    choices = asyncio.run(_stream([("{", None), ('"city": "Shang', "length")]))
    tool_deltas = [
        choice["delta"]["tool_calls"][0]
        for choice in choices
        if choice["delta"].get("tool_calls")
    ]

    assert tool_deltas[0]["function"] == {
        "name": "get_weather",
        "arguments": "{",
    }
    # continuation chunk streams arguments only; id/type/name are omitted
    assert "id" not in tool_deltas[1]
    assert "type" not in tool_deltas[1]
    assert "name" not in tool_deltas[1]["function"]
    assert tool_deltas[1]["function"]["arguments"] == '"city": "Shang'
    assert choices[-1]["finish_reason"] == "length"


def test_named_tool_choice_full_truncated_preserves_length():
    response = asyncio.run(
        _full('{"city": "Shang', _PlainContentParser(), finish_reason="length")
    )
    choice = response.choices[0]

    # A truncated named tool call keeps its real finish_reason.
    assert choice.finish_reason == "length"
