"""Provider-neutral second-stage reranking for isolated Retrieval V2.

No provider is constructed here.  A future integration must inject a provider
that implements ``ModelRerankProvider``; this keeps the current phase fully
offline and makes reranking an optional quality layer rather than an
availability dependency.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from retrieval.fusion import FusedCandidate
from retrieval.trace_support import emit


@dataclass(frozen=True)
class RerankedCandidate:
    candidate_id: str
    content: str
    source: str
    document_id: str | None
    document_version: int | None
    chunk_id: str | None
    parent_chunk_id: str | None
    retrieval_types: tuple[str, ...]
    source_ranks: dict[str, int]
    raw_scores: dict[str, float | None]
    rrf_score: float
    rerank_score: float | None
    pre_rerank_rank: int
    post_rerank_rank: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RerankDiagnostics:
    rerank_input_count: int
    rerank_output_count: int
    rerank_used: bool
    rerank_fallback_reason: str | None


@dataclass(frozen=True)
class RerankResult:
    candidates: tuple[RerankedCandidate, ...]
    diagnostics: RerankDiagnostics


@dataclass(frozen=True)
class RerankDocument:
    candidate_id: str
    text: str


@dataclass(frozen=True)
class RerankRequest:
    query: str
    documents: tuple[RerankDocument, ...]


@dataclass(frozen=True)
class RerankScore:
    candidate_id: str
    score: float


class Reranker(Protocol):
    async def rerank(
        self, query: str, candidates: list[FusedCandidate], top_k: int, *, trace: Any | None = None
    ) -> RerankResult: ...


class ModelRerankProvider(Protocol):
    """Future model boundary; no endpoint, model name, or network is assumed."""

    async def score(self, request: RerankRequest) -> Sequence[RerankScore]: ...


class RerankerError(RuntimeError):
    """Base class deliberately carries no provider response/body."""


class RerankerTimeoutError(RerankerError):
    pass


class RerankerUnavailableError(RerankerError):
    pass


class RerankerMalformedResponse(RerankerError):
    pass


class RerankerCandidateMismatch(RerankerMalformedResponse):
    pass


def _graph_text(candidate: FusedCandidate) -> str:
    """Render graph evidence as directed triples, never ``str(dict)``."""
    sources = candidate.metadata.get("retrieval_metadata", {})
    graph_metadata = sources.get("graph", {}) if isinstance(sources, dict) else {}
    edges = graph_metadata.get("evidence_edges", []) if isinstance(graph_metadata, dict) else []
    rendered: list[str] = []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        subject, predicate, obj = edge.get("subject"), edge.get("predicate"), edge.get("object")
        direction = edge.get("direction")
        if subject and predicate and obj:
            rendered.append(f"{subject} --{predicate}--> {obj} [direction: {direction or 'forward'}]")
    return "\n".join(rendered) if rendered else str(candidate.content)


def candidate_rerank_text(candidate: FusedCandidate) -> str:
    """Return one stable text per fused candidate for a future provider."""
    if "graph" in candidate.retrieval_types:
        return _graph_text(candidate)
    return str(candidate.content)


def _to_reranked(candidate: FusedCandidate, *, pre_rank: int, post_rank: int, score: float | None) -> RerankedCandidate:
    return RerankedCandidate(
        candidate_id=candidate.candidate_id,
        content=candidate.content,
        source=candidate.source,
        document_id=candidate.document_id,
        document_version=candidate.document_version,
        chunk_id=candidate.chunk_id,
        parent_chunk_id=candidate.parent_chunk_id,
        retrieval_types=candidate.retrieval_types,
        source_ranks=dict(candidate.source_ranks),
        raw_scores=dict(candidate.raw_scores),
        rrf_score=candidate.rrf_score,
        rerank_score=score,
        pre_rerank_rank=pre_rank,
        post_rerank_rank=post_rank,
        metadata=dict(candidate.metadata),
    )


def _fallback_result(candidates: list[FusedCandidate], top_k: int, reason: str | None) -> RerankResult:
    output = tuple(
        _to_reranked(candidate, pre_rank=index, post_rank=index, score=None)
        for index, candidate in enumerate(candidates[:top_k], start=1)
    )
    return RerankResult(
        candidates=output,
        diagnostics=RerankDiagnostics(
            rerank_input_count=len(candidates),
            rerank_output_count=len(output),
            rerank_used=False,
            rerank_fallback_reason=reason,
        ),
    )


def _rank_with_scores(candidates: list[FusedCandidate], scores: dict[str, float], top_k: int) -> RerankResult:
    expected = {candidate.candidate_id for candidate in candidates}
    if set(scores) != expected or len(scores) != len(candidates):
        raise RerankerCandidateMismatch("candidate identity mismatch")
    if any(not math.isfinite(score) for score in scores.values()):
        raise RerankerMalformedResponse("non-finite rerank score")
    indexed = list(enumerate(candidates, start=1))
    indexed.sort(key=lambda item: (
        -scores[item[1].candidate_id],
        -item[1].rrf_score,
        item[0],
        item[1].candidate_id,
    ))
    output = tuple(
        _to_reranked(candidate, pre_rank=pre_rank, post_rank=post_rank, score=scores[candidate.candidate_id])
        for post_rank, (pre_rank, candidate) in enumerate(indexed[:top_k], start=1)
    )
    return RerankResult(
        candidates=output,
        diagnostics=RerankDiagnostics(
            rerank_input_count=len(candidates),
            rerank_output_count=len(output),
            rerank_used=True,
            rerank_fallback_reason=None,
        ),
    )


class DisabledReranker:
    """Explicitly preserve RRF ordering when reranking is off or unavailable."""

    async def rerank(self, query: str, candidates: list[FusedCandidate], top_k: int, *, trace: Any | None = None) -> RerankResult:
        del query
        started = time.monotonic()
        emit(trace, "record_stage", "rerank_started", details={
            "input_count": len(candidates), "provider_kind": "disabled", "attempt_count": 0,
        })
        result = _fallback_result(candidates, top_k, "disabled")
        emit(trace, "record_stage", "rerank_completed", candidates=result.candidates, details={
            "output_count": len(result.candidates), "provider_kind": "disabled",
        }, latency_ms=(time.monotonic() - started) * 1000)
        return result


class FakeReranker:
    """Deterministic fake for tests; it never accesses a model or network."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = {str(key): float(value) for key, value in scores.items()}
        self.requests: list[tuple[str, tuple[str, ...]]] = []

    async def rerank(self, query: str, candidates: list[FusedCandidate], top_k: int, *, trace: Any | None = None) -> RerankResult:
        started = time.monotonic()
        emit(trace, "record_stage", "rerank_started", details={
            "input_count": len(candidates), "provider_kind": "fake", "attempt_count": 1,
        })
        self.requests.append((str(query), tuple(candidate.candidate_id for candidate in candidates)))
        result = _rank_with_scores(candidates, self.scores, top_k)
        emit(trace, "record_stage", "rerank_completed", candidates=result.candidates, details={
            "output_count": len(result.candidates), "provider_kind": "fake",
        }, latency_ms=(time.monotonic() - started) * 1000)
        return result


class ConfigurableModelReranker:
    """Injected provider adapter with bounded attempts and strict parsing."""

    def __init__(self, provider: ModelRerankProvider, *, timeout_seconds: float = 10.0, max_attempts: int = 2) -> None:
        if timeout_seconds <= 0 or max_attempts < 1:
            raise ValueError("reranker timeout and attempts must be bounded positive values")
        self.provider = provider
        self.timeout_seconds = float(timeout_seconds)
        self.max_attempts = int(max_attempts)

    async def rerank(self, query: str, candidates: list[FusedCandidate], top_k: int, *, trace: Any | None = None) -> RerankResult:
        started = time.monotonic()
        emit(trace, "record_stage", "rerank_started", details={
            "input_count": len(candidates), "provider_kind": "configured", "attempt_count": 0,
        })
        request = RerankRequest(
            query=str(query),
            documents=tuple(RerankDocument(candidate.candidate_id, candidate_rerank_text(candidate)) for candidate in candidates),
        )
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                response = await asyncio.wait_for(self.provider.score(request), timeout=self.timeout_seconds)
                if not isinstance(response, Sequence):
                    raise RerankerMalformedResponse("response is not a sequence")
                scores: dict[str, float] = {}
                for item in response:
                    if not isinstance(item, RerankScore) or item.candidate_id in scores:
                        raise RerankerMalformedResponse("malformed score entry")
                    scores[item.candidate_id] = float(item.score)
                result = _rank_with_scores(candidates, scores, top_k)
                emit(trace, "record_stage", "rerank_completed", candidates=result.candidates, details={
                    "output_count": len(result.candidates), "provider_kind": "configured", "attempt_count": attempt + 1,
                }, latency_ms=(time.monotonic() - started) * 1000)
                return result
            except asyncio.TimeoutError as exc:
                last_error = RerankerTimeoutError("timeout")
            except RerankerError as exc:
                last_error = exc
            except Exception as exc:  # Provider exceptions are never exposed as response bodies.
                last_error = RerankerUnavailableError(type(exc).__name__)
            if attempt + 1 >= self.max_attempts:
                break
        assert last_error is not None
        raise last_error
