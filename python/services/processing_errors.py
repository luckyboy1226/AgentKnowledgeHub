"""Safe, structured ingestion-failure classification.

Only stable exception types, explicitly exposed HTTP status codes, and the
exception cause chain are inspected.  Provider messages and response bodies
are intentionally never copied into the returned diagnostic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator

import httpx


@dataclass(frozen=True)
class SafeProcessingFailure:
    """A journal-safe description of one ingestion failure."""

    phase: str
    error_category: str
    error_type: str
    chunk_index: int | None = None

    def as_dict(self) -> dict[str, str]:
        return {
            "error_phase": self.phase,
            "error_category": self.error_category,
            "error_type": self.error_type,
            "chunk_index": self.chunk_index,
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
    if "extraction_validation_error" in explicit_categories:
        return SafeProcessingFailure(phase, "extraction_validation_error", error_type)
    statuses = tuple(status for item in chain if (status := _status_code(item)) is not None)
    if any(status == 429 for status in statuses):
        return SafeProcessingFailure(phase, "provider_rate_limit", error_type)
    if any(status in {401, 403} for status in statuses):
        return SafeProcessingFailure(phase, "provider_auth_error", error_type)
    if any(status >= 400 for status in statuses):
        return SafeProcessingFailure(phase, "provider_http_error", error_type)

    if any(isinstance(item, (httpx.TimeoutException, TimeoutError)) for item in chain):
        return SafeProcessingFailure(phase, "provider_timeout", error_type)
    if any(isinstance(item, httpx.NetworkError) for item in chain):
        return SafeProcessingFailure(phase, "provider_connection_error", error_type)
    if any(isinstance(item, json.JSONDecodeError) for item in chain):
        return SafeProcessingFailure(phase, "provider_invalid_response", error_type)

    # Imported lazily so classification remains usable in minimal offline tests.
    try:
        from openai import APIConnectionError, APITimeoutError, APIStatusError, AuthenticationError, RateLimitError
    except ImportError:  # pragma: no cover - production dependency is present
        pass
    else:
        if any(isinstance(item, RateLimitError) for item in chain):
            return SafeProcessingFailure(phase, "provider_rate_limit", error_type)
        if any(isinstance(item, AuthenticationError) for item in chain):
            return SafeProcessingFailure(phase, "provider_auth_error", error_type)
        if any(isinstance(item, APITimeoutError) for item in chain):
            return SafeProcessingFailure(phase, "provider_timeout", error_type)
        if any(isinstance(item, APIConnectionError) for item in chain):
            return SafeProcessingFailure(phase, "provider_connection_error", error_type)
        if any(isinstance(item, APIStatusError) for item in chain):
            return SafeProcessingFailure(phase, "provider_http_error", error_type)
    return SafeProcessingFailure(phase, "unknown", error_type)
