"""Fake-only tests for Phase D provider-neutral reranking."""

from __future__ import annotations

import asyncio
import inspect

import pytest

from retrieval.context_builder import ContextBuilderV2
from retrieval.fusion import FusedCandidate
from retrieval.parent_expander import ParentExpander
from retrieval.reranker import (
    ConfigurableModelReranker,
    DisabledReranker,
    FakeReranker,
    RerankScore,
    RerankerMalformedResponse,
    candidate_rerank_text,
)
from agents.qa_agent import QAAgent
from config.settings import Settings


def fused(candidate_id: str, rank: int, *, graph: bool = False) -> FusedCandidate:
    kinds = ("graph",) if graph else ("bm25", "vector")
    metadata = {"retrieval_metadata": {}}
    if graph:
        metadata["retrieval_metadata"]["graph"] = {"evidence_edges": [{
            "subject": "A", "predicate": "PROVIDES_INDEX", "object": "B",
            "direction": "forward", "evidence_key": "edge-1",
        }]}
    return FusedCandidate(
        candidate_id=candidate_id, content=f"content:{candidate_id}", source="safe.txt",
        document_id="doc-a", document_version=1,
        chunk_id=None if graph else candidate_id,
        parent_chunk_id=None if graph else "doc-a:v1:p0",
        retrieval_types=kinds, source_ranks={kinds[0]: rank}, raw_scores={kinds[0]: 0.5},
        rrf_score=1 / (60 + rank), metadata=metadata,
    )


class EmptyParents:
    def get_parent(self, *_args):
        return None


@pytest.mark.asyncio
async def test_fake_reranker_reorders_by_score_and_preserves_rrf_provenance():
    items = [fused("A", 1), fused("B", 2), fused("C", 3)]
    result = await FakeReranker({"A": 0.40, "B": 0.95, "C": 0.80}).rerank("normalized", items, 3)
    assert [item.candidate_id for item in result.candidates] == ["B", "C", "A"]
    assert [(item.pre_rerank_rank, item.post_rerank_rank, item.rerank_score) for item in result.candidates] == [
        (2, 1, 0.95), (3, 2, 0.80), (1, 3, 0.40),
    ]
    assert result.candidates[0].source_ranks == {"bm25": 2}
    assert result.diagnostics.rerank_used is True


@pytest.mark.asyncio
async def test_disabled_reranker_keeps_rrf_order_with_safe_diagnostic():
    result = await DisabledReranker().rerank("query", [fused("A", 1), fused("B", 2)], 2)
    assert [item.candidate_id for item in result.candidates] == ["A", "B"]
    assert result.diagnostics.rerank_used is False
    assert result.diagnostics.rerank_fallback_reason == "disabled"


@pytest.mark.asyncio
async def test_unknown_or_incomplete_score_set_is_rejected_not_created():
    with pytest.raises(RerankerMalformedResponse):
        await FakeReranker({"A": 0.9, "unknown": 0.8}).rerank("query", [fused("A", 1)], 1)


class TimeoutProvider:
    async def score(self, _request):
        await asyncio.sleep(1)


class BrokenProvider:
    async def score(self, _request):
        raise RuntimeError("provider body must never escape")


class MalformedProvider:
    async def score(self, _request):
        return [RerankScore("unknown", 0.9)]


class FlakyProvider:
    def __init__(self): self.calls = 0
    async def score(self, request):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary")
        return [RerankScore(item.candidate_id, 1.0) for item in request.documents]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", [TimeoutProvider(), BrokenProvider(), MalformedProvider()])
async def test_context_builder_falls_back_to_rrf_for_timeout_provider_or_malformed_response(provider):
    builder = ContextBuilderV2(
        parent_expander=ParentExpander(EmptyParents()),
        reranker=ConfigurableModelReranker(provider, timeout_seconds=0.001, max_attempts=1),
        rerank_enabled=True, parent_expansion_enabled=False,
    )
    result = await builder.build("normalized", [fused("A", 1), fused("B", 2)])
    assert [context.supporting_candidate_ids[0] for context in result.contexts] == ["A", "B"]
    assert result.diagnostics.rerank_used is False
    assert result.diagnostics.rerank_fallback_reason in {
        "RerankerTimeoutError", "RerankerUnavailableError", "RerankerMalformedResponse",
    }


@pytest.mark.asyncio
async def test_configurable_reranker_retries_only_within_its_bounded_attempt_limit():
    provider = FlakyProvider()
    result = await ConfigurableModelReranker(provider, max_attempts=2).rerank("query", [fused("A", 1)], 1)
    assert provider.calls == 2
    assert result.diagnostics.rerank_used is True


def test_graph_rerank_text_is_directed_structured_evidence_not_dictionary_rendering():
    rendered = candidate_rerank_text(fused("graph:edge-1", 1, graph=True))
    assert "A --PROVIDES_INDEX--> B" in rendered
    assert "direction: forward" in rendered
    assert "{" not in rendered and "DEPENDS_ON" not in rendered


@pytest.mark.asyncio
async def test_default_v2_builder_does_not_expand_parent_and_v1_qa_is_unchanged():
    builder = ContextBuilderV2(parent_expander=ParentExpander(EmptyParents()))
    result = await builder.build("query", [fused("A", 1)])
    assert result.contexts[0].kind == "child_fallback"
    settings = Settings()
    assert settings.hybrid_retrieval_v2_enabled is False
    assert settings.rerank_enabled is False and settings.parent_expansion_enabled is False
    answer_source = inspect.getsource(QAAgent.answer)
    assert "_hybrid_rerank" in answer_source and "ContextBuilderV2" not in answer_source
