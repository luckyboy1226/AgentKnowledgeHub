"""Fake-only coverage for request-scoped V2 RetrievalTrace."""

from __future__ import annotations

import asyncio
import inspect
import json
import math
from types import SimpleNamespace

import pytest

from agents.qa_agent import QAAgent
from config.settings import Settings
from observability.retrieval_trace import RetrievalTrace, RetrievalTraceStore
from retrieval.context_builder import ContextBuilderV2
from retrieval.fusion import RRFFusion
from retrieval.hybrid_retriever import HybridRetrieverV2
from retrieval.parent_expander import ParentExpander
from retrieval.reranker import ConfigurableModelReranker, FakeReranker, RerankScore


class BM25:
    async def search(self, _query, _top_k, *, allowed_document_ids=None):
        from retrieval.candidates import RetrievalCandidate
        return [RetrievalCandidate(
            candidate_id="doc-a:v1:c0", content="TOP SECRET DOCUMENT TEXT", source=r"C:\private\company\secret.pdf",
            document_id="doc-a", document_version=1, chunk_id="doc-a:v1:c0", parent_chunk_id="doc-a:v1:p0",
            retrieval_type="bm25", raw_score=99.0, rank=1, metadata={"status": "ready", "is_current": True},
        )]


class Vector:
    async def search(self, _query, *, top_k, allowed_document_ids=None):
        return [({
            "content": "TOP SECRET DOCUMENT TEXT", "source": r"C:\private\company\secret.pdf",
            "metadata": {"document_id": "doc-a", "document_version": 1, "chunk_id": "doc-a:v1:c0",
                         "parent_chunk_id": "doc-a:v1:p0", "status": "ready", "is_current": True},
        }, 0.9)]


class Graph:
    async def get_neighbors(self, _entity, **_kwargs):
        return [{"evidence_edges": [{
            "subject": "A", "predicate": "PROVIDES_INDEX", "object": "B", "direction": "forward",
            "evidence_key": "edge-1", "document_id": "doc-a", "document_version": 1, "source": "secret.pdf",
        }]}]


class Parents:
    def get_parent(self, document_id, version, parent_chunk_id):
        if (document_id, version, parent_chunk_id) != ("doc-a", 1, "doc-a:v1:p0"):
            return None
        return {"kind": "parent", "document_id": "doc-a", "document_version": 1,
                "parent_chunk_id": "doc-a:v1:p0", "content": "TOP SECRET PARENT CONTENT",
                "source": r"C:\private\company\secret.pdf", "estimated_token_count": 100,
                "status": "ready", "is_current": True}


async def run_pipeline(trace=None, *, parent_repository=None, reranker=None, budget=6000):
    recall = HybridRetrieverV2(bm25=BM25(), vector_store=Vector(), knowledge_graph=Graph())
    candidates = await recall.retrieve(
        "CEO薪资是多少？内部密钥XYZ", entities=["A"], bm25_enabled=True,
        allowed_document_ids=frozenset({"doc-a"}), trace=trace,
    )
    fusion = RRFFusion().fuse(candidates, trace=trace)
    builder = ContextBuilderV2(
        parent_expander=ParentExpander(parent_repository or Parents()),
        reranker=reranker or FakeReranker({item.candidate_id: 1.0 for item in fusion.candidates}),
        rerank_enabled=True, parent_expansion_enabled=True,
        final_context_token_budget=budget,
    )
    return await builder.build("CEO薪资是多少？内部密钥XYZ", fusion.candidates, allowed_document_ids=frozenset({"doc-a"}), trace=trace)


def stages(trace: RetrievalTrace):
    return trace.to_dict()["stages"]


@pytest.mark.asyncio
async def test_full_fake_pipeline_records_documented_stages_without_query_or_content():
    trace = RetrievalTrace("request-a")
    result = await run_pipeline(trace)
    assert [stage["stage"] for stage in stages(trace)] == [
        "query_received", "query_rewritten", "bm25_retrieved", "vector_retrieved", "graph_retrieved",
        "rrf_fused", "rerank_started", "rerank_completed", "parent_expand_started",
        "parent_expand_completed", "budget_applied", "final_context_built",
    ]
    assert result.contexts
    rendered = json.dumps(trace.to_dict(), ensure_ascii=False)
    assert "CEO薪资是多少" not in rendered and "内部密钥XYZ" not in rendered
    assert "TOP SECRET" not in rendered and r"C:\private" not in rendered
    assert "secret.pdf" in rendered
    assert json.dumps(trace.to_dict())


@pytest.mark.asyncio
async def test_latency_values_are_finite_non_negative_and_candidate_fields_are_bounded():
    trace = RetrievalTrace("request-latency")
    await run_pipeline(trace)
    for stage in stages(trace):
        if "latency_ms" in stage:
            assert math.isfinite(stage["latency_ms"]) and stage["latency_ms"] >= 0
        assert stage["candidate_summary"]["stored_count"] <= 50


def test_candidate_limit_and_non_finite_scores_are_safe():
    from retrieval.candidates import RetrievalCandidate
    trace = RetrievalTrace("request-limit", max_candidates_per_stage=50)
    rows = [RetrievalCandidate(
        candidate_id=f"c{index}", content="private", source=r"C:\private\secret.pdf", document_id="a",
        document_version=1, chunk_id=f"c{index}", parent_chunk_id="p", retrieval_type="bm25",
        raw_score=float("nan") if index == 0 else 1.0, rank=index + 1, metadata={},
    ) for index in range(100)]
    trace.record_stage("bm25_retrieved", candidates=rows)
    stage = stages(trace)[0]
    assert stage["candidate_summary"] == {"original_count": 100, "stored_count": 50, "truncated": True}
    assert stage["candidates"][0]["raw_scores"]["bm25"] is None


def test_store_ttl_and_max_entries_are_bounded_with_injected_clock():
    now = [0.0]
    store = RetrievalTraceStore(max_entries=3, ttl_seconds=10, clock=lambda: now[0])
    for request_id in ("A", "B", "C", "D"):
        store.put(RetrievalTrace(request_id))
        now[0] += 1
    assert store.get("A") is None
    assert store.get("B") is not None
    now[0] = 20
    assert store.get("B") is None


@pytest.mark.asyncio
async def test_request_traces_remain_isolated_under_parallel_fake_requests():
    first, second = RetrievalTrace("request-first"), RetrievalTrace("request-second")
    await asyncio.gather(run_pipeline(first), run_pipeline(second))
    assert first.to_dict()["request_id"] == "request-first" and second.to_dict()["request_id"] == "request-second"
    first.record_stage("rrf_fused", candidates=[SimpleNamespace(candidate_id="only-first", source="first.txt")])
    second.record_stage("rrf_fused", candidates=[SimpleNamespace(candidate_id="only-second", source="second.txt")])
    first_ids = {row["candidate_id"] for stage in stages(first) for row in stage["candidates"] if row["candidate_id"]}
    second_ids = {row["candidate_id"] for stage in stages(second) for row in stage["candidates"] if row["candidate_id"]}
    assert "only-first" in first_ids and "only-first" not in second_ids
    assert "only-second" in second_ids and "only-second" not in first_ids


@pytest.mark.asyncio
async def test_trace_on_off_has_identical_final_context_ids_and_order():
    disabled = await run_pipeline(None)
    enabled = await run_pipeline(RetrievalTrace("request-on"))
    assert [(item.context_id, item.final_rank) for item in disabled.contexts] == [
        (item.context_id, item.final_rank) for item in enabled.contexts
    ]


class TimeoutProvider:
    async def score(self, _request):
        await asyncio.sleep(1)


@pytest.mark.asyncio
async def test_rerank_timeout_records_fallback_and_preserves_rrf_final_context():
    trace = RetrievalTrace("request-timeout")
    result = await run_pipeline(trace, reranker=ConfigurableModelReranker(TimeoutProvider(), timeout_seconds=0.001, max_attempts=1))
    stage_names = [stage["stage"] for stage in stages(trace)]
    fallback = next(stage for stage in stages(trace) if stage["stage"] == "rerank_fallback")
    assert "rerank_started" in stage_names and "rerank_completed" not in stage_names
    assert fallback["details"]["reason_code"] == "rerank_timeout"
    assert result.contexts and result.diagnostics.rerank_used is False


@pytest.mark.asyncio
async def test_parent_missing_and_budget_drop_are_reflected_in_trace_facts():
    class MissingParents:
        def get_parent(self, *_args): return None
    missing_trace = RetrievalTrace("request-missing")
    missing = await run_pipeline(missing_trace, parent_repository=MissingParents())
    complete = next(stage for stage in stages(missing_trace) if stage["stage"] == "parent_expand_completed")
    assert complete["details"]["parent_missing_count"] == 1
    assert any(context.kind == "child_fallback" for context in missing.contexts)

    budget_trace = RetrievalTrace("request-budget")
    budget = await run_pipeline(budget_trace, budget=1)
    budget_stage = next(stage for stage in stages(budget_trace) if stage["stage"] == "budget_applied")
    assert budget.contexts == ()
    assert any(row["reason_code"] == "budget_exceeded" for row in budget_stage["details"]["decisions"])


def test_v1_defaults_remain_trace_free_and_qa_agent_has_no_trace_dependency():
    settings = Settings()
    assert settings.hybrid_retrieval_v2_enabled is False and settings.retrieval_trace_enabled is False
    source = inspect.getsource(QAAgent.answer)
    assert "RetrievalTrace" not in source and "TraceStore" not in source
