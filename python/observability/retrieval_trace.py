"""Safe request-scoped observability for the internal Retrieval V2 pipeline.

This is intentionally independent from graph evidence tracing: it records
pipeline facts and safe candidate identities, never document content, prompts,
answers, credentials, or full queries.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from retrieval.candidates import safe_source


STAGES = frozenset({
    "query_received", "query_rewritten", "bm25_retrieved", "vector_retrieved",
    "graph_retrieved", "rrf_fused", "rerank_started", "rerank_completed",
    "rerank_fallback", "parent_expand_started", "parent_expand_completed",
    "budget_applied", "final_context_built",
})
FAILURE_REASONS = frozenset({
    "invalid_candidate", "provenance_conflict", "scope_rejected", "rerank_timeout",
    "rerank_provider_error", "rerank_malformed", "rerank_candidate_mismatch",
    "parent_missing", "parent_invalid", "legacy_parent_unavailable", "budget_exceeded",
    "top_k_truncated", "parent_to_child_fallback",
})
_DENIED_DETAIL_TOKENS = frozenset({
    "content", "query", "prompt", "answer", "secret", "password", "credential",
    "api_key", "authorization", "url", "path", "exception",
})


def _finite(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _safe_scalar(value: object) -> object | None:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return None


def _safe_details(details: Mapping[str, object] | None) -> dict[str, object]:
    safe: dict[str, object] = {}
    explicitly_safe = {
        "query_fingerprint", "query_length", "rewrite_query_count", "rewritten_query_fingerprints",
    }
    for key, value in (details or {}).items():
        rendered = str(key).lower()
        if rendered not in explicitly_safe and any(token in rendered for token in _DENIED_DETAIL_TOKENS):
            continue
        if isinstance(value, Mapping):
            nested = _safe_details({str(nested_key): nested_value for nested_key, nested_value in value.items()})
            safe[str(key)] = nested
        elif isinstance(value, (tuple, list)):
            safe_values: list[object] = []
            for item in value:
                if isinstance(item, Mapping):
                    safe_values.append(_safe_details({str(nested_key): nested_value for nested_key, nested_value in item.items()}))
                else:
                    scalar = _safe_scalar(item)
                    if scalar is not None:
                        safe_values.append(scalar)
            safe[str(key)] = safe_values
        else:
            scalar = _safe_scalar(value)
            if scalar is not None:
                safe[str(key)] = scalar
    return safe


def _metadata_value(candidate: object, name: str) -> object:
    metadata = getattr(candidate, "metadata", {})
    return metadata.get(name) if isinstance(metadata, Mapping) else None


def _candidate_record(candidate: object) -> dict[str, object]:
    """Extract the finite, source-safe candidate identity contract only."""
    retrieval_types = getattr(candidate, "retrieval_types", None)
    if retrieval_types is None:
        retrieval_type = getattr(candidate, "retrieval_type", None)
        retrieval_types = (retrieval_type,) if retrieval_type else ()
    raw_scores = getattr(candidate, "raw_scores", None)
    if not isinstance(raw_scores, Mapping):
        raw_score = _finite(getattr(candidate, "raw_score", None))
        raw_scores = {str(retrieval_types[0]): raw_score} if retrieval_types else {}
    safe_scores = {str(key): _finite(value) for key, value in raw_scores.items()}
    candidate_id = str(getattr(candidate, "candidate_id", getattr(candidate, "context_id", "")))
    context_id = str(getattr(candidate, "context_id", "")) or None
    return {
        "candidate_id": candidate_id,
        "context_id": context_id,
        "kind": _safe_scalar(getattr(candidate, "kind", None)),
        "document_id": str(getattr(candidate, "document_id", "")) or None,
        "document_version": _safe_scalar(getattr(candidate, "document_version", None)),
        "chunk_id": str(getattr(candidate, "chunk_id", "")) or None,
        "parent_chunk_id": str(getattr(candidate, "parent_chunk_id", "")) or None,
        "retrieval_types": [str(item) for item in retrieval_types],
        "rank": _safe_scalar(getattr(candidate, "rank", getattr(candidate, "post_rerank_rank", getattr(candidate, "final_rank", None)))),
        "pre_rerank_rank": _safe_scalar(getattr(candidate, "pre_rerank_rank", None)),
        "post_rerank_rank": _safe_scalar(getattr(candidate, "post_rerank_rank", None)),
        "source_ranks": {
            str(key): value for key, value in (getattr(candidate, "source_ranks", {}) or {}).items()
            if type(value) is int and value >= 1
        },
        "raw_scores": safe_scores,
        "rrf_score": _finite(getattr(candidate, "rrf_score", None)),
        "rerank_score": _finite(getattr(candidate, "rerank_score", None)),
        "source_basename": safe_source(getattr(candidate, "source", "")),
        "status": _safe_scalar(_metadata_value(candidate, "status")),
        "is_current": _safe_scalar(_metadata_value(candidate, "is_current")),
        "supporting_candidate_ids": [str(item) for item in getattr(candidate, "supporting_candidate_ids", ())],
        "estimated_token_count": _safe_scalar(getattr(candidate, "estimated_token_count", None)),
    }


@dataclass(frozen=True)
class TraceCandidate:
    """Public schema documentation for the serialised safe candidate record."""

    candidate_id: str
    document_id: str | None
    document_version: int | None
    chunk_id: str | None
    parent_chunk_id: str | None
    retrieval_types: tuple[str, ...]
    source_basename: str


class RetrievalTrace:
    """Request-owned trace collector with bounded candidate snapshots."""

    def __init__(
        self,
        request_id: str,
        *,
        max_candidates_per_stage: int = 50,
        monotonic_clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        if not str(request_id).strip() or max_candidates_per_stage < 1:
            raise ValueError("trace identity and candidate limit are required")
        self.request_id = str(request_id)
        self.max_candidates_per_stage = int(max_candidates_per_stage)
        self._monotonic_clock = monotonic_clock
        self._wall_clock = wall_clock
        self.created_at = float(wall_clock())
        self.created_monotonic = float(monotonic_clock())
        self._stages: list[dict[str, object]] = []
        self._lock = threading.RLock()

    def monotonic(self) -> float:
        return float(self._monotonic_clock())

    def elapsed_ms(self) -> float:
        return max(0.0, self.monotonic() - self.created_monotonic) * 1000

    def query_received(self, query: str) -> None:
        self.record_stage("query_received", details={
            "query_fingerprint": hashlib.sha256(str(query).encode("utf-8")).hexdigest(),
            "query_length": len(str(query)),
        })

    def query_rewritten(self, queries: Iterable[str], *, entity_count: int = 0, keyword_count: int = 0) -> None:
        normalized = tuple(str(query) for query in queries)
        self.record_stage("query_rewritten", details={
            "rewrite_query_count": len(normalized),
            "entity_count": int(entity_count),
            "keyword_count": int(keyword_count),
            "rewritten_query_fingerprints": [hashlib.sha256(item.encode("utf-8")).hexdigest() for item in normalized],
            "rewrite_ms": 0.0,
        }, latency_ms=0.0)

    def record_stage(
        self,
        stage: str,
        *,
        candidates: Iterable[object] = (),
        details: Mapping[str, object] | None = None,
        latency_ms: float | None = None,
    ) -> None:
        if stage not in STAGES:
            raise ValueError("unknown retrieval trace stage")
        rows = [_candidate_record(candidate) for candidate in candidates]
        stored = rows[:self.max_candidates_per_stage]
        entry: dict[str, object] = {
            "stage": stage,
            "monotonic_ms": max(0.0, self.monotonic() - self.created_monotonic) * 1000,
            "details": _safe_details(details),
            "candidate_summary": {
                "original_count": len(rows), "stored_count": len(stored), "truncated": len(rows) > len(stored),
            },
            "candidates": stored,
        }
        safe_latency = _finite(latency_ms)
        if safe_latency is not None:
            entry["latency_ms"] = max(0.0, safe_latency)
        with self._lock:
            self._stages.append(entry)

    def to_dict(self) -> dict[str, object]:
        with self._lock:
            payload = {
                "request_id": self.request_id,
                "created_at": self.created_at,
                "stages": [dict(stage) for stage in self._stages],
            }
        # A JSON round-trip produces a detached snapshot and verifies the
        # public contract without returning mutable internal stage objects.
        return json.loads(json.dumps(payload))


class RetrievalTraceStore:
    """Small thread-safe TTL/LRU-ish trace store; no external persistence."""

    def __init__(
        self, *, max_entries: int = 500, ttl_seconds: int = 1800, clock: Callable[[], float] = time.monotonic
    ) -> None:
        if max_entries < 1 or ttl_seconds < 1:
            raise ValueError("trace store limits must be positive")
        self.max_entries, self.ttl_seconds, self._clock = int(max_entries), int(ttl_seconds), clock
        self._entries: dict[str, tuple[float, int, RetrievalTrace]] = {}
        self._sequence = 0
        self._lock = threading.RLock()

    def _purge(self, now: float) -> None:
        for request_id, (created, _sequence, _trace) in list(self._entries.items()):
            if now - created >= self.ttl_seconds:
                del self._entries[request_id]

    def put(self, trace: RetrievalTrace) -> None:
        now = float(self._clock())
        with self._lock:
            self._purge(now)
            self._sequence += 1
            self._entries[trace.request_id] = (now, self._sequence, trace)
            while len(self._entries) > self.max_entries:
                oldest = min(self._entries.items(), key=lambda item: (item[1][0], item[1][1]))[0]
                del self._entries[oldest]

    def get(self, request_id: str) -> dict[str, object] | None:
        now = float(self._clock())
        with self._lock:
            self._purge(now)
            entry = self._entries.get(str(request_id))
            return entry[2].to_dict() if entry else None
