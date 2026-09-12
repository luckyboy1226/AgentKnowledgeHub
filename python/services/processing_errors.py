"""Safe, structured ingestion-failure classification.

Only stable exception types, explicitly exposed HTTP status codes, and the
exception cause chain are inspected.  Provider messages and response bodies
are intentionally never copied into the returned diagnostic.
"""

from __future__ import annotations

import json
import ssl
from dataclasses import dataclass
from typing import Any, Iterator

import httpx


@dataclass(frozen=True)
class SafeProcessingFailure:
    """A journal-safe description of one ingestion failure."""

    phase: str
    error_category: str
    error_type: str
    chunk_index: int | None = None
    timeout_kind: str | None = None
    attempt: int | None = None
    max_attempts: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "error_phase": self.phase,
            "error_category": self.error_category,
            "error_type": self.error_type,
            "chunk_index": self.chunk_index,
            "timeout_kind": self.timeout_kind,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
        }


def _exception_chain(error: BaseException) -> Iterator[BaseException]:
    """Yield a finite cause/context chain without rendering exception text."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        next_error = current.__cause__ or current.__context__
        current = next_error if isinstance(next_error, BaseException) else None


def _status_code(error: BaseException) -> int | None:
    """Read only an explicit numeric status field; never parse error text."""
    value = getattr(error, "status_code", None)
    if isinstance(value, int):
        return value
    response = getattr(error, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _safe_int_from_chain(chain: tuple[BaseException, ...], field: str) -> int | None:
    for item in chain:
        value = getattr(item, field, None)
        if isinstance(value, int) and value >= 0:
            return value
    return None


def _safe_timeout_kind(chain: tuple[BaseException, ...]) -> str | None:
    for item in chain:
        value = getattr(item, "timeout_kind", None)
        if value in {"connect", "read", "request_deadline", "chunk_deadline", "document_deadline", "tls_read_wait"}:
            return value
    if any(isinstance(item, ssl.SSLWantReadError) for item in chain):
        return "tls_read_wait"
    if any(isinstance(item, httpx.ConnectTimeout) for item in chain):
        return "connect"
    if any(isinstance(item, httpx.ReadTimeout) for item in chain):
        return "read"
    if any(isinstance(item, httpx.TimeoutException) for item in chain):
        return "request_deadline"
    return None


def _failure(
    phase: str, category: str, error_type: str, chain: tuple[BaseException, ...], *, timeout_kind: str | None = None
) -> SafeProcessingFailure:
    return SafeProcessingFailure(
        phase,
        category,
        error_type,
        chunk_index=_safe_int_from_chain(chain, "chunk_index"),
        timeout_kind=timeout_kind,
        attempt=_safe_int_from_chain(chain, "attempt"),
        max_attempts=_safe_int_from_chain(chain, "max_attempts"),
    )


def classify_safe_processing_error(error: BaseException, *, phase: str) -> SafeProcessingFailure:
    """Classify a processing error without retaining sensitive provider detail.

    ``phase`` is supplied by the boundary that owns the operation.  Unknown
    types deliberately remain ``unknown`` rather than using message matching.
    """
    if phase not in {"parse", "extract", "normalize", "unknown"}:
        phase = "unknown"

    chain = tuple(_exception_chain(error))
    error_type = type(chain[-1]).__name__ if chain else type(error).__name__
    explicit_categories = tuple(
        getattr(type(item), "safe_processing_category", None)
        for item in chain
    )
    if "validation" in explicit_categories or "extraction_validation_error" in explicit_categories:
        return _failure(phase, "validation", error_type, chain)
    statuses = tuple(status for item in chain if (status := _status_code(item)) is not None)
    if any(status == 429 for status in statuses):
        return _failure(phase, "provider_rate_limit", error_type, chain)
    if any(status in {401, 403} for status in statuses):
        return _failure(phase, "provider_auth", error_type, chain)
    if any(status >= 400 for status in statuses):
        return _failure(phase, "provider_http", error_type, chain)

    if any(isinstance(item, (httpx.TimeoutException, TimeoutError)) for item in chain):
        return _failure(phase, "provider_timeout", error_type, chain, timeout_kind=_safe_timeout_kind(chain))
    if any(isinstance(item, httpx.NetworkError) for item in chain):
        return _failure(phase, "provider_connection", error_type, chain)
    if any(isinstance(item, json.JSONDecodeError) for item in chain):
        return _failure(phase, "provider_invalid_response", error_type, chain)

    # Imported lazily so classification remains usable in minimal offline tests.
    try:
        from openai import APIConnectionError, APITimeoutError, APIStatusError, AuthenticationError, RateLimitError
    except ImportError:  # pragma: no cover - production dependency is present
        pass
    else:
        if any(isinstance(item, RateLimitError) for item in chain):
            return _failure(phase, "provider_rate_limit", error_type, chain)
        if any(isinstance(item, AuthenticationError) for item in chain):
            return _failure(phase, "provider_auth", error_type, chain)
        if any(isinstance(item, APITimeoutError) for item in chain):
            return _failure(phase, "provider_timeout", error_type, chain, timeout_kind=_safe_timeout_kind(chain) or "request_deadline")
        if any(isinstance(item, APIConnectionError) for item in chain):
            return _failure(phase, "provider_connection", error_type, chain)
        if any(isinstance(item, APIStatusError) for item in chain):
            return _failure(phase, "provider_http", error_type, chain)
    return _failure(phase, "unknown", error_type, chain)
