"""Deterministic reciprocal-rank fusion for the isolated Retrieval V2 path.

The service accepts already-retrieved candidates only.  It deliberately does
not retrieve, expand parents, rerank, or generate an answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

from retrieval.candidates import RetrievalCandidate, RetrievalType, safe_source
from retrieval.trace_support import emit


_RETRIEVAL_TYPES: tuple[RetrievalType, ...] = ("bm25", "vector", "graph")


@dataclass(frozen=True)
class FusedCandidate:
    """One candidate with all rank and provenance contributions retained."""

    candidate_id: str
    content: str
    source: str
    document_id: str | None
    document_version: int | None
    chunk_id: str | None
    parent_chunk_id: str | None
    retrieval_types: tuple[RetrievalType, ...]
    source_ranks: dict[str, int]
    raw_scores: dict[str, float | None]
    rrf_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RRFFusionDiagnostics:
    """Safe counters for a fusion run; no query, prompt, or content is kept."""

    input_counts: dict[str, int]
    unique_candidate_count: int
    multi_source_candidate_count: int
    invalid_candidate_count: int
    provenance_conflict_count: int
    output_count: int


@dataclass(frozen=True)
class RRFFusionResult:
    candidates: tuple[FusedCandidate, ...]
    diagnostics: RRFFusionDiagnostics


def _false_value(value: object) -> bool:
    return value is False or (isinstance(value, str) and value.strip().lower() == "false")


def _eligible(candidate: RetrievalCandidate) -> bool:
    """Keep V2 lifecycle/scope boundaries defensive even after retrieval."""
    metadata = candidate.metadata
    if "status" in metadata and metadata["status"] != "ready":
        return False
    if "is_current" in metadata and metadata["is_current"] is not True:
        return False
    return not any(
        _false_value(metadata.get(key))
        for key in ("scope_verified", "scope_accepted")
    ) and metadata.get("scope_conflict") is not True


def _identity(candidate: RetrievalCandidate) -> tuple[object, ...]:
    return (
        candidate.document_id,
        candidate.document_version,
        candidate.chunk_id,
        candidate.parent_chunk_id,
        safe_source(candidate.source),
    )


class RRFFusion:
    """Unweighted RRF with strict rank/provenance validation.

    A malformed candidate is skipped.  A candidate ID with contradictory
    document provenance is removed entirely rather than silently choosing one
    retriever's version of the record.
    """

    def __init__(self, rrf_k: int = 60, fusion_top_k: int = 30) -> None:
        if rrf_k < 1:
            raise ValueError("rrf_k must be positive")
        if fusion_top_k < 1:
            raise ValueError("fusion_top_k must be positive")
        self.rrf_k = int(rrf_k)
        self.fusion_top_k = int(fusion_top_k)

    def fuse(
        self,
        retrieval_results: dict[str, list[RetrievalCandidate]],
        top_k: int | None = None,
        *,
        trace: Any | None = None,
    ) -> RRFFusionResult:
        """Fuse supplied rank lists without comparing raw score spaces."""
        started = time.monotonic()
        effective_top_k = self.fusion_top_k if top_k is None else int(top_k)
        if effective_top_k < 1:
            raise ValueError("top_k must be positive")

        input_counts = {
            retrieval_type: len(retrieval_results.get(retrieval_type, []))
            if isinstance(retrieval_results.get(retrieval_type, []), list) else 0
            for retrieval_type in _RETRIEVAL_TYPES
        }
        invalid_count = 0
        conflict_count = 0
        grouped: dict[str, list[RetrievalCandidate]] = {}
        rejected_ids: set[str] = set()

        # Unknown list names have no RRF semantics.  Count their supplied
        # entries as rejected rather than treating them as a fourth retriever.
        for retrieval_type, candidates in retrieval_results.items():
            if retrieval_type not in _RETRIEVAL_TYPES and isinstance(candidates, list):
                invalid_count += len(candidates)

        for retrieval_type in _RETRIEVAL_TYPES:
            source_candidates = retrieval_results.get(retrieval_type, [])
            if not isinstance(source_candidates, list):
                continue
            rank_counts: dict[int, int] = {}
            for candidate in source_candidates:
                if isinstance(candidate, RetrievalCandidate) and type(candidate.rank) is int and candidate.rank >= 1:
                    rank_counts[candidate.rank] = rank_counts.get(candidate.rank, 0) + 1

            for candidate in source_candidates:
                if not isinstance(candidate, RetrievalCandidate):
                    invalid_count += 1
                    continue
                if (
                    candidate.retrieval_type != retrieval_type
                    or type(candidate.rank) is not int
                    or candidate.rank < 1
                    or rank_counts.get(candidate.rank, 0) != 1
                    or not _eligible(candidate)
                ):
                    invalid_count += 1
                    continue
                if candidate.candidate_id in rejected_ids:
                    invalid_count += 1
                    continue
                existing = grouped.get(candidate.candidate_id)
                if existing and _identity(existing[0]) != _identity(candidate):
                    # Reject the complete inconsistent identity, including the
                    # earlier contribution, instead of silently preferring it.
                    del grouped[candidate.candidate_id]
                    rejected_ids.add(candidate.candidate_id)
                    conflict_count += 1
                    continue
                grouped.setdefault(candidate.candidate_id, []).append(candidate)

        fused: list[FusedCandidate] = []
        for candidate_id, candidates in grouped.items():
            canonical = candidates[0]
            contributions = {candidate.retrieval_type: candidate for candidate in candidates}
            if len(contributions) != len(candidates):
                invalid_count += len(candidates)
                continue
            retrieval_types = tuple(kind for kind in _RETRIEVAL_TYPES if kind in contributions)
            source_ranks = {kind: contributions[kind].rank for kind in retrieval_types}
            raw_scores = {kind: contributions[kind].raw_score for kind in retrieval_types}
            rrf_score = sum(1.0 / (self.rrf_k + rank) for rank in source_ranks.values())
            metadata = {
                "retrieval_metadata": {kind: dict(contributions[kind].metadata) for kind in retrieval_types},
                "document_id": canonical.document_id,
                "document_version": canonical.document_version,
                "chunk_id": canonical.chunk_id,
                "parent_chunk_id": canonical.parent_chunk_id,
                "source": safe_source(canonical.source),
            }
            fused.append(FusedCandidate(
                candidate_id=candidate_id,
                content=canonical.content,
                source=safe_source(canonical.source),
                document_id=canonical.document_id,
                document_version=canonical.document_version,
                chunk_id=canonical.chunk_id,
                parent_chunk_id=canonical.parent_chunk_id,
                retrieval_types=retrieval_types,
                source_ranks=source_ranks,
                raw_scores=raw_scores,
                rrf_score=rrf_score,
                metadata=metadata,
            ))

        fused.sort(key=lambda item: (
            -item.rrf_score,
            -len(item.retrieval_types),
            min(item.source_ranks.values()),
            item.candidate_id,
        ))
        output = tuple(fused[:effective_top_k])
        diagnostics = RRFFusionDiagnostics(
            input_counts=input_counts,
            unique_candidate_count=len(grouped),
            multi_source_candidate_count=sum(1 for item in fused if len(item.retrieval_types) > 1),
            invalid_candidate_count=invalid_count,
            provenance_conflict_count=conflict_count,
            output_count=len(output),
        )
        result = RRFFusionResult(candidates=output, diagnostics=diagnostics)
        emit(trace, "record_stage", "rrf_fused", candidates=output, details={
            "input_counts": diagnostics.input_counts,
            "unique_candidate_count": diagnostics.unique_candidate_count,
            "multi_source_candidate_count": diagnostics.multi_source_candidate_count,
            "invalid_candidate_count": diagnostics.invalid_candidate_count,
            "provenance_conflict_count": diagnostics.provenance_conflict_count,
            "output_count": diagnostics.output_count,
        }, latency_ms=(time.monotonic() - started) * 1000)
        return result
