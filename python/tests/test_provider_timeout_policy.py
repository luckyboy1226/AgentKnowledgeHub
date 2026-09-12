"""Fake-only contracts for bounded extraction provider behavior."""

from __future__ import annotations

import asyncio
import json
import ssl
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException

import api.main as api
from agents.doc_parser_agent import DocType, DocumentChunk
from agents.knowledge_extract_agent import (
    ExtractionDeadlineError,
    ExtractionInvocationError,
    KnowledgeExtractAgent,
)
from config.settings import ExtractionTimeoutPolicy, Settings
from providers.chat import ChatRequestOptions, OpenAICompatibleChatProvider
from services.document_processor import KnowledgeExtractionError
from services.processing_errors import classify_safe_processing_error


VALID_RESPONSE = json.dumps({"entities": [], "relations": [], "events": []})


class FakeChat:
    provider_name = "fake"
    model_name = "fake-model"

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[ChatRequestOptions | None] = []

    async def ainvoke(self, _messages, *, request_options=None, **_kwargs):
        self.calls.append(request_options)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, float):
            await asyncio.sleep(outcome)
            return SimpleNamespace(content=VALID_RESPONSE)
        return SimpleNamespace(content=outcome)


def chunk(index=0):
    return DocumentChunk("safe fixture text", "doc", index, DocType.TEXT, {})


def policy(**overrides):
    values = {
        "request_timeout_seconds": 0.03,
        "chunk_deadline_seconds": 0.08,
        "document_deadline_seconds": 0.2,
        "max_attempts": 2,
        "retry_backoff_seconds": 0,
    }
    values.update(overrides)
    return ExtractionTimeoutPolicy(**values)


@pytest.mark.asyncio
async def test_extraction_uses_separate_request_options_from_ordinary_chat():
    fake = FakeChat([VALID_RESPONSE])
    await KnowledgeExtractAgent(fake, timeout_policy=policy(request_timeout_seconds=0.04)).extract([chunk()])
    assert fake.calls[0] == ChatRequestOptions(timeout_seconds=0.04, sdk_max_retries=0)


@pytest.mark.asyncio
async def test_first_chunk_timeout_has_safe_index_and_bounded_attempts():
    fake = FakeChat([asyncio.TimeoutError(), asyncio.TimeoutError(), VALID_RESPONSE])
    with pytest.raises(ExtractionInvocationError) as caught:
        await KnowledgeExtractAgent(fake, timeout_policy=policy(max_attempts=2)).extract([chunk(0)])
    failure = classify_safe_processing_error(caught.value, phase="extract")
    assert len(fake.calls) == 2
    assert (failure.error_category, failure.chunk_index, failure.attempt, failure.max_attempts) == (
        "provider_timeout", 0, 2, 2
    )
    assert failure.timeout_kind == "request_deadline"


@pytest.mark.asyncio
async def test_middle_chunk_failure_keeps_its_index():
    fake = FakeChat([VALID_RESPONSE, httpx.ReadTimeout("secret=never-store"), httpx.ReadTimeout("secret=never-store")])
    with pytest.raises(ExtractionInvocationError) as caught:
        await KnowledgeExtractAgent(fake, timeout_policy=policy()).extract([chunk(0), chunk(1)])
    failure = classify_safe_processing_error(caught.value, phase="extract")
    assert failure.chunk_index == 1 and failure.timeout_kind == "read"
    assert "never-store" not in str(failure.as_dict())


@pytest.mark.asyncio
async def test_chunk_deadline_stops_retry_without_long_sleep():
    fake = FakeChat([0.2])
    with pytest.raises(ExtractionDeadlineError) as caught:
        await KnowledgeExtractAgent(fake, timeout_policy=policy(request_timeout_seconds=0.2, chunk_deadline_seconds=0.01)).extract([chunk(0)])
    assert caught.value.timeout_kind == "chunk_deadline"
    assert caught.value.chunk_index == 0 and len(fake.calls) == 1


@pytest.mark.asyncio
async def test_document_deadline_cannot_be_attributed_to_a_chunk():
    fake = FakeChat([0.03, 0.03])
    with pytest.raises(ExtractionDeadlineError) as caught:
        await KnowledgeExtractAgent(fake, timeout_policy=policy(chunk_deadline_seconds=0.08, document_deadline_seconds=0.04)).extract([chunk(0), chunk(1)])
    assert caught.value.timeout_kind == "document_deadline" and caught.value.chunk_index is None


def test_timeout_classifier_distinguishes_connect_and_read():
    assert classify_safe_processing_error(httpx.ConnectTimeout("x"), phase="extract").timeout_kind == "connect"
    assert classify_safe_processing_error(httpx.ReadTimeout("x"), phase="extract").timeout_kind == "read"


def test_tls_read_wait_is_timeout_only_when_in_timeout_chain():
    timeout = TimeoutError()
    timeout.__cause__ = ssl.SSLWantReadError()
    failure = classify_safe_processing_error(timeout, phase="extract")
    assert (failure.error_category, failure.timeout_kind, failure.error_type) == (
        "provider_timeout", "tls_read_wait", "SSLWantReadError"
    )


@pytest.mark.parametrize(
    ("error", "category"),
    [
        (httpx.ConnectError("x"), "provider_connection"),
        (json.JSONDecodeError("bad", "x", 0), "provider_invalid_response"),
    ],
)
def test_safe_error_categories_without_message_matching(error, category):
    assert classify_safe_processing_error(error, phase="extract").error_category == category


def test_http_status_categories_are_safe():
    request = httpx.Request("POST", "https://provider.invalid")
    rate = httpx.HTTPStatusError("secret=never", request=request, response=httpx.Response(429, request=request))
    auth = httpx.HTTPStatusError("secret=never", request=request, response=httpx.Response(401, request=request))
    upstream = httpx.HTTPStatusError("secret=never", request=request, response=httpx.Response(502, request=request))
    assert classify_safe_processing_error(rate, phase="extract").error_category == "provider_rate_limit"
    assert classify_safe_processing_error(auth, phase="extract").error_category == "provider_auth"
    assert classify_safe_processing_error(upstream, phase="extract").error_category == "provider_http"


def test_api_maps_wrapped_provider_errors_without_leaking_text():
    error = KnowledgeExtractionError("safe wrapper")
    error.__cause__ = httpx.ReadTimeout("api_key=never-show")
    with pytest.raises(HTTPException) as caught:
        api._raise_document_error(error)
    assert caught.value.status_code == 504 and "never-show" not in str(caught.value.detail)


def test_api_maps_connection_and_invalid_provider_responses():
    connection = KnowledgeExtractionError("safe")
    connection.__cause__ = httpx.ConnectError("secret")
    invalid = KnowledgeExtractionError("safe")
    invalid.__cause__ = json.JSONDecodeError("bad", "x", 0)
    with pytest.raises(HTTPException) as first:
        api._raise_document_error(connection)
    with pytest.raises(HTTPException) as second:
        api._raise_document_error(invalid)
    assert first.value.status_code == 503 and second.value.status_code == 502


def test_timeout_configuration_is_bounded_and_hierarchical():
    configured = Settings(
        _env_file=None,
        extraction_request_timeout_seconds=120,
        extraction_chunk_deadline_seconds=180,
        document_processing_timeout_seconds=900,
        extraction_max_attempts=2,
    )
    assert configured.extraction_timeout_policy.max_attempts == 2
    with pytest.raises(ValueError):
        Settings(_env_file=None, extraction_request_timeout_seconds=200, extraction_chunk_deadline_seconds=100)
    with pytest.raises(ValueError):
        Settings(_env_file=None, extraction_max_attempts=4)


def test_chat_provider_keeps_one_provider_and_uses_override_transport_options():
    created = []

    class Client:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created.append(self)

        async def ainvoke(self, _messages, **_kwargs):
            return SimpleNamespace(content=VALID_RESPONSE)

    provider = OpenAICompatibleChatProvider(
        "fake", "key", "https://provider.invalid", "model", client_factory=Client
    )
    selected = provider._client_for(ChatRequestOptions(timeout_seconds=120, sdk_max_retries=0))
    assert provider._client_for(ChatRequestOptions(timeout_seconds=120, sdk_max_retries=0)) is selected
    assert len(created) == 2
    assert selected.kwargs["timeout"] == 120 and selected.kwargs["max_retries"] == 0
