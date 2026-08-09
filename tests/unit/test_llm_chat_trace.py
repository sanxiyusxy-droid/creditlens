"""OpenAI-compatible chat tracing stays useful without retaining model data."""

import hashlib

import httpx
import pytest
from pydantic import BaseModel, Field

from creditlens.infrastructure.llm.chat import LLMCallError, OpenAICompatChat


class _Answer(BaseModel):
    statement: str = Field(min_length=3)


async def _mocked_chat(handler, api_key: str = "api-key-must-not-leak") -> OpenAICompatChat:
    chat = OpenAICompatChat(
        base_url="https://provider.example/v1",
        api_key=api_key,
        model="test-model",
    )
    # Replace the real transport only after closing the client allocated by the
    # production constructor, so the test does not leak a connection pool.
    await chat.aclose()
    chat._client = httpx.AsyncClient(
        base_url="https://provider.example/v1",
        headers={"Authorization": f"Bearer {api_key}"},
        transport=httpx.MockTransport(handler),
    )
    return chat


def _completion(content: str, usage: dict | None = None, status: int = 200) -> httpx.Response:
    payload: dict = {"choices": [{"message": {"content": content}}]}
    if usage is not None:
        payload["usage"] = usage
    return httpx.Response(status, json=payload)


async def test_generate_structured_traced_success_is_redacted():
    system = "system prompt secret"
    user = "user evidence secret"
    api_key = "api-key-must-not-leak"
    response_text = '{"statement":"validated response secret"}'

    def handler(_request: httpx.Request) -> httpx.Response:
        return _completion(
            response_text,
            {"prompt_tokens": 17, "completion_tokens": 5, "total_tokens": 22},
        )

    chat = await _mocked_chat(handler, api_key)
    try:
        result = await chat.generate_structured_traced(
            system=system,
            user=user,
            output_schema=_Answer,
            prompt_version="grounded_qa_v1",
        )
    finally:
        await chat.aclose()

    assert result.output.statement == "validated response secret"
    trace = result.trace
    assert trace.provider == "openai_compatible"
    assert trace.model == "test-model"
    assert trace.prompt_version == "grounded_qa_v1"
    assert trace.status == "SUCCESS"
    assert trace.error_type is None
    assert trace.attempts == 1
    assert trace.input_tokens == 17
    assert trace.output_tokens == 5
    assert trace.total_tokens == 22
    assert trace.latency_ms >= 0
    for digest in (trace.prompt_sha256, trace.request_sha256, trace.response_sha256):
        assert digest is not None
        assert len(digest) == 64
        int(digest, 16)

    persisted = trace.model_dump_json()
    for secret in (system, user, response_text, "validated response secret", api_key):
        assert secret not in persisted


async def test_schema_retry_aggregates_usage_and_hashes_last_response():
    responses = [
        _completion(
            '{"wrong":"first invalid response"}',
            {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        ),
        _completion(
            '{"statement":"second response wins"}',
            {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
        ),
    ]
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        response = responses[calls]
        calls += 1
        return response

    chat = await _mocked_chat(handler)
    try:
        result = await chat.generate_structured_traced(
            system="system",
            user="user",
            output_schema=_Answer,
            prompt_version="prompt-v2",
        )
    finally:
        await chat.aclose()

    assert calls == 2
    assert result.output.statement == "second response wins"
    assert result.trace.attempts == 2
    assert result.trace.input_tokens == 21
    assert result.trace.output_tokens == 5
    assert result.trace.total_tokens == 26
    assert result.trace.response_sha256 == hashlib.sha256(responses[-1].content).hexdigest()


async def test_usage_is_nullable_when_provider_omits_it():
    def handler(_request: httpx.Request) -> httpx.Response:
        return _completion('{"statement":"valid without usage"}')

    chat = await _mocked_chat(handler)
    try:
        result = await chat.generate_structured_traced(
            system="system",
            user="user",
            output_schema=_Answer,
        )
    finally:
        await chat.aclose()

    assert result.trace.input_tokens is None
    assert result.trace.output_tokens is None
    assert result.trace.total_tokens is None


async def test_two_schema_failures_raise_with_redacted_failed_trace():
    invalid_response = '{"wrong":"response body must not leak"}'

    def handler(_request: httpx.Request) -> httpx.Response:
        return _completion(
            invalid_response,
            {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
        )

    chat = await _mocked_chat(handler)
    try:
        with pytest.raises(LLMCallError) as captured:
            await chat.generate_structured_traced(
                system="private system prompt",
                user="private user prompt",
                output_schema=_Answer,
                prompt_version="prompt-failed",
            )
    finally:
        await chat.aclose()

    error = captured.value
    assert str(error) == "结构化输出两次校验失败"
    assert error.trace.status == "FAILED"
    assert error.trace.error_type == "ValidationError"
    assert error.trace.attempts == 2
    assert error.trace.input_tokens == 14
    assert error.trace.output_tokens == 4
    assert error.trace.total_tokens == 18
    persisted = error.trace.model_dump_json()
    for secret in (invalid_response, "response body must not leak", "private system prompt"):
        assert secret not in persisted


async def test_http_failure_has_single_attempt_and_no_response_body_in_trace():
    provider_error = "sensitive upstream response"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text=provider_error)

    chat = await _mocked_chat(handler)
    try:
        with pytest.raises(LLMCallError) as captured:
            await chat.generate_structured_traced(
                system="system",
                user="user",
                output_schema=_Answer,
            )
    finally:
        await chat.aclose()

    trace = captured.value.trace
    assert trace.status == "FAILED"
    assert trace.error_type == "HTTPStatusError"
    assert trace.attempts == 1
    assert trace.response_sha256 is not None
    assert trace.input_tokens is None
    assert provider_error not in trace.model_dump_json()
    assert provider_error not in str(captured.value)


async def test_legacy_generate_structured_returns_only_validated_output_and_aclose_works():
    def handler(_request: httpx.Request) -> httpx.Response:
        return _completion('{"statement":"legacy caller remains compatible"}')

    chat = await _mocked_chat(handler)
    output = await chat.generate_structured(
        system="system",
        user="user",
        output_schema=_Answer,
    )
    assert isinstance(output, _Answer)
    assert output.statement == "legacy caller remains compatible"

    client = chat._client
    await chat.aclose()
    assert client.is_closed
