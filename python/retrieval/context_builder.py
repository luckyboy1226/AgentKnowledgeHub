"""Internal V2 context assembly: RRF -> rerank -> parent expansion -> budget.

This module intentionally has no QAAgent/API integration. It stops at
``FinalContext`` and never constructs a prompt or invokes an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass

from retrieval.fusion import FusedCandidate
from retrieval.parent_expander import FinalContext, ParentExpander, apply_context_budget
from retrieval.reranker import DisabledReranker, RerankDiagnostics, RerankResult, Reranker, RerankerError
from retrieval.trace_support import emit


@dataclass(frozen=True)
class ContextBuilderDiagnostics:
    rerank_input_count: int
    rerank_output_count: int
    rerank_used: bool
    rerank_fallback_reason: str | None
    parent_expand_attempt_count: int
    parent_expand_success_count: int
    parent_missing_count: int
    parent_invalid_count: int
    parent_scope_rejected_count: int
    legacy_child_fallback_count: int
    graph_context_count: int
    budget_input_count: int
    budget_output_count: int
    budget_dropped_count: int
    estimated_tokens_used: int
    parent_to_child_fallback_count: int


@dataclass(frozen=True)
class ContextBuilderResult:
    contexts: tuple[FinalContext, ...]
    diagnostics: ContextBuilderDiagnostics


class ContextBuilderV2:
    """Composable, opt-in V2 context pipeline for future evaluation work."""

    def __init__(
        self,
        *,
        parent_expander: ParentExpander,
        reranker: Reranker | None = None,
        rerank_enabled: bool = False,
        parent_expansion_enabled: bool = False,
        rerank_input_top_k: int = 30,
        rerank_output_top_k: int = 12,
        final_context_top_k: int = 8,
        final_context_token_budget: int = 6000,
    ) -> None:
        if min(rerank_input_top_k, rerank_output_top_k, final_context_top_k, final_context_token_budget) < 1:
            raise ValueError("context builder limits must be positive")
        self.parent_expander = parent_expander
        self.reranker = reranker or DisabledReranker()
        self.rerank_enabled = bool(rerank_enabled)
        self.parent_expansion_enabled = bool(parent_expansion_enabled)
        self.rerank_input_top_k = int(rerank_input_top_k)
        self.rerank_output_top_k = int(rerank_output_top_k)
        self.final_context_top_k = int(final_context_top_k)
        self.final_context_token_budget = int(final_context_token_budget)

    async def build(
        self,
        query: str,
        candidates: tuple[FusedCandidate, ...] | list[FusedCandidate],
        *,
        allowed_document_ids: frozenset[str] | None = None,
        trace: object | None = None,
    ) -> ContextBuilderResult:
        rrf_top = list(candidates)[:self.rerank_input_top_k]
        if self.rerank_enabled:
            try:
                rerank_result = await self.reranker.rerank(query, rrf_top, self.rerank_output_top_k, trace=trace)
            except RerankerError as exc:
                reason_map = {
                    "RerankerTimeoutError": "rerank_timeout",
                    "RerankerMalformedResponse": "rerank_malformed",
                    "RerankerCandidateMismatch": "rerank_candidate_mismatch",
                }
                reason = reason_map.get(type(exc).__name__, "rerank_provider_error")
                emit(trace, "record_stage", "rerank_fallback", details={"reason_code": reason})
                fallback = await DisabledReranker().rerank(query, rrf_top, self.rerank_output_top_k, trace=None)
                rerank_result = RerankResult(
                    candidates=fallback.candidates,
                    diagnostics=RerankDiagnostics(
                        rerank_input_count=fallback.diagnostics.rerank_input_count,
                        rerank_output_count=fallback.diagnostics.rerank_output_count,
                        rerank_used=False,
                        rerank_fallback_reason=type(exc).__name__,
                    ),
                )
        else:
            rerank_result = await DisabledReranker().rerank(query, rrf_top, self.rerank_output_top_k, trace=trace)

        expansion = await self.parent_expander.expand(
            list(rerank_result.candidates),
            allowed_document_ids=allowed_document_ids,
            enabled=self.parent_expansion_enabled,
            trace=trace,
        )
        contexts, budget = apply_context_budget(
            expansion.contexts,
            expansion.fallback_contexts,
            top_k=self.final_context_top_k,
            token_budget=self.final_context_token_budget,
            trace=trace,
        )
        rerank = rerank_result.diagnostics
        parent = expansion.diagnostics
        total_elapsed = None
        try:
            elapsed = getattr(trace, "elapsed_ms", None)
            total_elapsed = elapsed() if callable(elapsed) else None
        except Exception:
            total_elapsed = None
        emit(trace, "record_stage", "final_context_built", candidates=contexts, details={
            "output_count": len(contexts), "retrieval_total_ms": total_elapsed,
        })
        return ContextBuilderResult(
            contexts=contexts,
            diagnostics=ContextBuilderDiagnostics(
                rerank_input_count=rerank.rerank_input_count,
                rerank_output_count=rerank.rerank_output_count,
                rerank_used=rerank.rerank_used,
                rerank_fallback_reason=rerank.rerank_fallback_reason,
                parent_expand_attempt_count=parent.parent_expand_attempt_count,
                parent_expand_success_count=parent.parent_expand_success_count,
                parent_missing_count=parent.parent_missing_count,
                parent_invalid_count=parent.parent_invalid_count,
                parent_scope_rejected_count=parent.parent_scope_rejected_count,
                legacy_child_fallback_count=parent.legacy_child_fallback_count,
                graph_context_count=parent.graph_context_count,
                budget_input_count=budget["budget_input_count"],
                budget_output_count=budget["budget_output_count"],
                budget_dropped_count=budget["budget_dropped_count"],
                estimated_tokens_used=budget["estimated_tokens_used"],
                parent_to_child_fallback_count=budget["parent_to_child_fallback_count"],
            ),
        )
