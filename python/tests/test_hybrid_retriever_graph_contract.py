"""Strict production graph-service contract tests for the V2 orchestrator."""

from __future__ import annotations

import pytest

from observability.retrieval_trace import RetrievalTrace
from retrieval.hybrid_retriever import HybridRetrieverV2
from services.graph_evidence_trace import GraphEvidenceTrace


class Vector:
    def __init__(self): self.calls = 0
    async def search(self, *_args, **_kwargs):
        self.calls += 1
        return []


class BM25:
    def __init__(self): self.calls = 0
    async def search(self, *_args, **_kwargs):
        self.calls += 1
        return []


class StrictProductionGraph:
    """No ``**kwargs``: an unsupported graph argument is a test failure."""
    def __init__(self): self.calls = []
    async def get_neighbors(self, entity_name, hops=2, *, allowed_document_ids=None,
                            scope_diagnostics=None, graph_trace=None):
        self.calls.append({"entity_name": entity_name, "hops": hops,
                           "allowed_document_ids": allowed_document_ids,
                           "scope_diagnostics": scope_diagnostics,
                           "graph_trace": graph_trace})
        record={"evidence_edges": [{
            "subject": "Atlas", "predicate": "USES", "object": "Queue",
            "direction": "forward", "evidence_key": "edge-1", "document_id": "doc-a",
            "document_version": 1, "source": "safe.txt", "status": "ready", "is_current": True,
        }]}
        if graph_trace is not None:
            graph_trace.record_record("retrieved_raw", record)
        return [record]


@pytest.mark.asyncio
@pytest.mark.parametrize("variant,selection,bm25_enabled,expected_graph_calls", [
    ("vector_only", {"vector": True, "bm25": False, "graph": False}, False, 0),
    ("bm25_vector_rrf", {"vector": True, "bm25": True, "graph": False}, True, 0),
    ("vector_graph_rrf", {"vector": True, "bm25": False, "graph": True}, False, 1),
    ("hybrid_v2_no_rerank", {"vector": True, "bm25": True, "graph": True}, True, 1),
])
async def test_variant_selection_uses_supported_graph_contract(
    variant, selection, bm25_enabled, expected_graph_calls,
):
    vector, bm25, graph = Vector(), BM25(), StrictProductionGraph()
    retriever = HybridRetrieverV2(bm25=bm25, vector_store=vector, knowledge_graph=graph)
    scope = frozenset({"doc-a"})
    await retriever.retrieve("Atlas", entities=["Atlas"], allowed_document_ids=scope,
                             bm25_enabled=bm25_enabled, selection=selection)
    assert len(graph.calls) == expected_graph_calls
    if expected_graph_calls:
        assert graph.calls == [{"entity_name": "Atlas", "hops": 2,
                                "allowed_document_ids": scope, "scope_diagnostics": None,
                                "graph_trace": None}]


@pytest.mark.asyncio
async def test_trace_off_and_on_use_the_same_strict_graph_contract():
    vector, bm25, graph = Vector(), BM25(), StrictProductionGraph()
    retriever = HybridRetrieverV2(bm25=bm25, vector_store=vector, knowledge_graph=graph)
    scope = frozenset({"doc-a"})
    await retriever.retrieve("Atlas", entities=["Atlas"], allowed_document_ids=scope,
                             selection={"vector": True, "bm25": False, "graph": True},
                             trace=None, graph_trace=None)
    retrieval_trace = RetrievalTrace("g4-q01")
    evidence_trace = GraphEvidenceTrace(run_id="g4", question_id="Q01",
                                        scope_verified=True, allowed_document_ids_count=1)
    await retriever.retrieve("Atlas", entities=["Atlas"], allowed_document_ids=scope,
                             selection={"vector": True, "bm25": False, "graph": True},
                             trace=retrieval_trace, graph_trace=evidence_trace)
    assert [call["hops"] for call in graph.calls] == [2, 2]
    assert graph.calls[0]["graph_trace"] is None
    assert graph.calls[1]["graph_trace"] is evidence_trace
    assert "retrieved_raw" in evidence_trace.to_dict()["stages"]
