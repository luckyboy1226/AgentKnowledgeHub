"""Turn reranked child evidence into bounded, provenance-safe final contexts."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from retrieval.reranker import RerankedCandidate
from services.chunk_models import UnicodeTokenEstimator


FinalContextKind = Literal["parent", "child_fallback", "graph_evidence"]


@dataclass(frozen=True)
class FinalContext:
    context_id: str
    kind: FinalContextKind
    content: str
    source: str
    document_id: str | None
    document_version: int | None
    parent_chunk_id: str | None
    supporting_candidate_ids: tuple[str, ...]
    supporting_child_ids: tuple[str, ...]
    retrieval_types: tuple[str, ...]
    rrf_score: float
    rerank_score: float | None
    final_rank: int
    estimated_token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParentExpansionDiagnostics:
    parent_expand_attempt_count: int
    parent_expand_success_count: int
    parent_missing_count: int
    parent_scope_rejected_count: int
    legacy_child_fallback_count: int
    graph_context_count: int


@dataclass(frozen=True)
class ParentExpansionResult:
    contexts: tuple[FinalContext, ...]
    fallback_contexts: dict[str, FinalContext]
    diagnostics: ParentExpansionDiagnostics


def _safe_source(value: object) -> str:
    return str(value or "").replace("\\", "/").rsplit("/", 1)[-1][:180]


def _token_count(value: object, estimator: UnicodeTokenEstimator) -> int:
    if type(value) is int and value > 0:
        return value
    return max(1, estimator.count(str(value or "")))


def _candidate_fallback(candidate: RerankedCandidate, *, estimator: UnicodeTokenEstimator) -> FinalContext:
    chunk_identity = candidate.chunk_id or candidate.candidate_id
    token_value = candidate.metadata.get("estimated_token_count") if isinstance(candidate.metadata, dict) else None
    return FinalContext(
        context_id=f"child:{chunk_identity}",
        kind="child_fallback",
        content=str(candidate.content),
        source=_safe_source(candidate.source),
        document_id=candidate.document_id,
        document_version=candidate.document_version,
        parent_chunk_id=candidate.parent_chunk_id,
        supporting_candidate_ids=(candidate.candidate_id,),
        supporting_child_ids=(chunk_identity,),
        retrieval_types=tuple(candidate.retrieval_types),
        rrf_score=float(candidate.rrf_score),
        rerank_score=candidate.rerank_score,
        final_rank=candidate.post_rerank_rank,
        estimated_token_count=_token_count(token_value, estimator)
        if token_value is not None else max(1, estimator.count(candidate.content)),
        metadata={"supporting_child_provenance": {"chunk_id": chunk_identity}},
    )


def _graph_content(candidate: RerankedCandidate) -> str:
    retrieval_metadata = candidate.metadata.get("retrieval_metadata", {}) if isinstance(candidate.metadata, dict) else {}
    graph_metadata = retrieval_metadata.get("graph", {}) if isinstance(retrieval_metadata, dict) else {}
    edges = graph_metadata.get("evidence_edges", []) if isinstance(graph_metadata, dict) else []
    parts: list[str] = []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        subject, predicate, obj = edge.get("subject"), edge.get("predicate"), edge.get("object")
        if subject and predicate and obj:
            parts.append(f"{subject} --{predicate}--> {obj} [direction: {edge.get('direction') or 'forward'}]")
    return "\n".join(parts) if parts else str(candidate.content)


def _graph_context(candidate: RerankedCandidate, *, estimator: UnicodeTokenEstimator) -> FinalContext:
    content = _graph_content(candidate)
    metadata = dict(candidate.metadata)
    return FinalContext(
        context_id=f"graph:{candidate.candidate_id}",
        kind="graph_evidence",
        content=content,
        source=_safe_source(candidate.source),
        document_id=candidate.document_id,
        document_version=candidate.document_version,
        parent_chunk_id=None,
        supporting_candidate_ids=(candidate.candidate_id,),
        supporting_child_ids=(),
        retrieval_types=tuple(candidate.retrieval_types),
        rrf_score=float(candidate.rrf_score),
        rerank_score=candidate.rerank_score,
        final_rank=candidate.post_rerank_rank,
        estimated_token_count=max(1, estimator.count(content)),
        metadata=metadata,
    )


class ParentExpander:
    """Expand only exact ready/current parents after child reranking."""

    def __init__(self, repository: Any, *, estimator: UnicodeTokenEstimator | None = None) -> None:
        self.repository = repository
        self.estimator = estimator or UnicodeTokenEstimator()

    async def expand(
        self,
        candidates: list[RerankedCandidate],
        *,
        allowed_document_ids: frozenset[str] | None = None,
        enabled: bool = True,
    ) -> ParentExpansionResult:
        # Explicitly empty scopes cannot turn into unscoped parent reads.
        if allowed_document_ids is not None and not allowed_document_ids:
            return ParentExpansionResult(
                contexts=(), fallback_contexts={},
                diagnostics=ParentExpansionDiagnostics(0, 0, 0, len(candidates), 0, 0),
            )

        attempts = successes = missing = scope_rejected = legacy_fallbacks = graph_count = 0
        parent_groups: dict[tuple[str, int, str], dict[str, Any]] = {}
        standalone: dict[str, FinalContext] = {}

        for candidate in candidates:
            if allowed_document_ids is not None and candidate.document_id not in allowed_document_ids:
                scope_rejected += 1
                continue
            if "graph" in candidate.retrieval_types:
                context = _graph_context(candidate, estimator=self.estimator)
                standalone.setdefault(context.context_id, context)
                graph_count += 1
                continue
            if not enabled:
                context = _candidate_fallback(candidate, estimator=self.estimator)
                standalone.setdefault(context.context_id, context)
                legacy_fallbacks += 1
                continue
            if not candidate.parent_chunk_id or not candidate.document_id or candidate.document_version is None:
                context = _candidate_fallback(candidate, estimator=self.estimator)
                standalone.setdefault(context.context_id, context)
                legacy_fallbacks += 1
                continue

            attempts += 1
            parent = self.repository.get_parent(
                candidate.document_id, int(candidate.document_version), candidate.parent_chunk_id
            )
            if inspect.isawaitable(parent):
                parent = await parent
            if not self._eligible_parent(parent, candidate, allowed_document_ids):
                context = _candidate_fallback(candidate, estimator=self.estimator)
                standalone.setdefault(context.context_id, context)
                missing += 1
                continue

            key = (candidate.document_id, int(candidate.document_version), candidate.parent_chunk_id)
            group = parent_groups.setdefault(key, {"parent": parent, "candidates": []})
            group["candidates"].append(candidate)

        parent_contexts: list[FinalContext] = []
        fallbacks: dict[str, FinalContext] = {}
        for (document_id, version, parent_chunk_id), group in parent_groups.items():
            group_candidates = sorted(group["candidates"], key=lambda item: (item.post_rerank_rank, item.candidate_id))
            best = group_candidates[0]
            parent = group["parent"]
            child_ids = tuple(dict.fromkeys((item.chunk_id or item.candidate_id) for item in group_candidates))
            candidate_ids = tuple(dict.fromkeys(item.candidate_id for item in group_candidates))
            retrieval_types = tuple(dict.fromkeys(kind for item in group_candidates for kind in item.retrieval_types))
            context_id = f"parent:{document_id}:v{version}:{parent_chunk_id}"
            parent_context = FinalContext(
                context_id=context_id,
                kind="parent",
                content=str(parent["content"]),
                source=_safe_source(parent.get("source") or best.source),
                document_id=document_id,
                document_version=version,
                parent_chunk_id=parent_chunk_id,
                supporting_candidate_ids=candidate_ids,
                supporting_child_ids=child_ids,
                retrieval_types=retrieval_types,
                rrf_score=float(best.rrf_score),
                rerank_score=best.rerank_score,
                final_rank=best.post_rerank_rank,
                estimated_token_count=_token_count(parent.get("estimated_token_count"), self.estimator),
                metadata={
                    "supporting_child_provenance": {"child_ids": child_ids},
                    "best_post_rerank_rank": best.post_rerank_rank,
                    "best_rerank_score": best.rerank_score,
                },
            )
            parent_contexts.append(parent_context)
            fallbacks[context_id] = _candidate_fallback(best, estimator=self.estimator)
            successes += 1

        ordered = sorted(
            [*parent_contexts, *standalone.values()],
            key=lambda item: (item.final_rank, item.context_id),
        )
        return ParentExpansionResult(
            contexts=tuple(ordered),
            fallback_contexts=fallbacks,
            diagnostics=ParentExpansionDiagnostics(
                attempts, successes, missing, scope_rejected, legacy_fallbacks, graph_count
            ),
        )

    @staticmethod
    def _eligible_parent(
        parent: Any,
        child: RerankedCandidate,
        allowed_document_ids: frozenset[str] | None,
    ) -> bool:
        if not isinstance(parent, dict):
            return False
        if (
            parent.get("kind") != "parent"
            or parent.get("status") != "ready"
            or parent.get("is_current") is not True
            or not isinstance(parent.get("content"), str)
        ):
            return False
        if (
            str(parent.get("document_id")) != str(child.document_id)
            or int(parent.get("document_version", -1)) != int(child.document_version)
            or str(parent.get("parent_chunk_id")) != str(child.parent_chunk_id)
        ):
            return False
        return allowed_document_ids is None or str(parent.get("document_id")) in allowed_document_ids


def apply_context_budget(
    contexts: tuple[FinalContext, ...] | list[FinalContext],
    fallback_contexts: dict[str, FinalContext],
    *,
    top_k: int = 8,
    token_budget: int = 6000,
) -> tuple[tuple[FinalContext, ...], dict[str, int]]:
    """Select complete context units; use a child only when its parent cannot fit."""
    if top_k < 1 or token_budget < 1:
        raise ValueError("final context limits must be positive")
    selected: list[FinalContext] = []
    tokens_used = 0
    for context in sorted(contexts, key=lambda item: (item.final_rank, item.context_id)):
        if len(selected) >= top_k:
            break
        chosen = context
        remaining = token_budget - tokens_used
        if context.estimated_token_count > remaining:
            fallback = fallback_contexts.get(context.context_id)
            if fallback is None or fallback.estimated_token_count > remaining:
                continue
            chosen = fallback
        if chosen.estimated_token_count > token_budget - tokens_used:
            continue
        selected.append(replace(chosen, final_rank=len(selected) + 1))
        tokens_used += chosen.estimated_token_count
    return tuple(selected), {
        "budget_input_count": len(contexts),
        "budget_output_count": len(selected),
        "budget_dropped_count": len(contexts) - len(selected),
        "estimated_tokens_used": tokens_used,
    }
