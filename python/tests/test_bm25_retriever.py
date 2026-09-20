"""Fake-only tests for Phase B derived BM25 and unified candidates."""

from __future__ import annotations

import pytest

from retrieval.bm25_retriever import BM25Retriever, DeterministicChineseTokenizer
from retrieval.candidates import from_graph_record, from_vector_result, graph_candidate_id
from retrieval.hybrid_retriever import HybridRetrieverV2


def child(document_id, version, chunk_id, content, *, status="ready", current=True, parent="p0", source="safe.txt"):
    return {
        "kind": "child", "document_id": document_id, "document_version": version,
        "chunk_id": chunk_id, "parent_chunk_id": parent, "content": content,
        "source": source, "status": status, "is_current": current,
        "section_title": "section", "page_number": 1, "table_id": None,
        "estimated_token_count": 10, "chunk_index": 0, "metadata": {},
    }


class Catalog:
    def __init__(self, rows): self.rows = rows
    def list_current_children(self): return list(self.rows)


def test_tokenizer_preserves_identifiers_and_is_unicode_deterministic():
    tokenizer = DeterministicChineseTokenizer()
    value = "液压泵检修周期 E1007 Redis-7.2 POST /api/qa/ask Qwen2.5"
    expected = tokenizer.tokenize(value)
    assert expected == tokenizer.tokenize(value)
    assert {"e1007", "redis-7.2", "post", "/api/qa/ask", "qwen2.5"}.issubset(expected)
    assert "液压泵检修周期" in expected and "液压" in expected and "周期" in expected


@pytest.mark.asyncio
async def test_exact_identifier_and_chinese_bm25_recall_prefer_the_matching_child():
    retriever = BM25Retriever(Catalog([
        child("A", 1, "A:v1:c0", "液压泵维护规范"),
        child("B", 1, "B:v1:c0", "错误代码 E1007 表示电机过热"),
    ]))
    assert (await retriever.search("E1007", 5))[0].document_id == "B"
    assert (await retriever.search("电机过热", 5))[0].document_id == "B"


@pytest.mark.asyncio
async def test_version_stage_scope_and_delete_are_strictly_filtered_on_rebuild():
    catalog = Catalog([
        child("A", 1, "A:v1:c0", "检修周期 1000 小时", current=False),
        child("A", 2, "A:v2:c0", "检修周期 800 小时"),
        child("B", 1, "B:v1:c0", "检修周期 700 小时"),
        child("A", 3, "A:v3:c0", "检修周期 600 小时", status="processing", current=False),
    ])
    retriever = BM25Retriever(catalog)
    found = await retriever.search("检修周期", 10, allowed_document_ids=frozenset({"A"}))
    assert [(item.document_id, item.document_version) for item in found] == [("A", 2)]
    catalog.rows = []  # precise source-of-truth deletion followed by derived rebuild
    retriever.mark_stale()
    assert await retriever.search("检修周期", 10) == []


@pytest.mark.asyncio
async def test_empty_scope_query_index_and_over_limit_fail_safely():
    empty = BM25Retriever(Catalog([]))
    assert await empty.search("anything", 5) == []
    assert await empty.search("", 5) == []
    indexed = BM25Retriever(Catalog([child("A", 1, "A:v1:c0", "E1007")]))
    assert await indexed.search("E1007", 5, allowed_document_ids=frozenset()) == []
    limited = BM25Retriever(Catalog([child("A", 1, "A:v1:c0", "one"), child("B", 1, "B:v1:c0", "two")]), max_indexed_children=1)
    status = await limited.rebuild()
    assert status["available"] is False and status["stale"] is True and status["last_error"] == "max_indexed_children_exceeded"


def test_vector_and_graph_normalization_have_stable_candidate_identity_and_safe_provenance():
    vector = from_vector_result({"content": "body", "source": r"C:\private\safe.txt", "metadata": {
        "document_id": "A", "document_version": 2, "chunk_id": "A:v2:c0", "parent_chunk_id": "A:v2:p0",
    }}, 0.87, 1)
    first_record = {"evidence_edges": [{"subject": "A", "predicate": "USES", "object": "B", "direction": "forward", "evidence_key": "edge-1", "document_id": "A", "document_version": 2, "source": "safe.txt"}]}
    graph = from_graph_record(first_record, 1, 0.8)
    changed = dict(first_record) | {"evidence_edges": [dict(first_record["evidence_edges"][0]) | {"evidence_key": "edge-2"}]}
    assert (vector.candidate_id, vector.retrieval_type, vector.raw_score, vector.rank) == ("A:v2:c0", "vector", 0.87, 1)
    assert vector.source == "safe.txt" and vector.parent_chunk_id == "A:v2:p0"
    assert graph.candidate_id == graph_candidate_id(first_record) and graph.retrieval_type == "graph"
    assert graph.candidate_id != graph_candidate_id(changed)


class Vector:
    async def search(self, query, top_k, allowed_document_ids=None):
        assert query == "E1007" and top_k == 2
        return [({"content": "E1007", "source": "safe.txt", "metadata": {"document_id": "A", "document_version": 1, "chunk_id": "A:v1:c0", "parent_chunk_id": "A:v1:p0"}}, 0.9)]


class Graph:
    def __init__(self): self.calls = 0
    async def get_neighbors(self, entity, **kwargs):
        self.calls += 1
        assert entity == "Atlas" and kwargs["allowed_document_ids"] == frozenset({"A"})
        return [{"evidence_edges": [{"subject": "Atlas", "predicate": "USES", "object": "E1007", "direction": "forward", "evidence_key": "e1", "document_id": "A", "document_version": 1, "source": "safe.txt"}]}]


@pytest.mark.asyncio
async def test_v2_orchestrator_keeps_three_ranked_lists_separate_without_fusion():
    bm25 = BM25Retriever(Catalog([child("A", 1, "A:v1:c0", "E1007")]))
    graph = Graph()
    retriever = HybridRetrieverV2(bm25=bm25, vector_store=Vector(), knowledge_graph=graph)
    output = await retriever.retrieve("E1007", entities=["Atlas"], bm25_top_k=2, vector_top_k=2, graph_top_k=2, allowed_document_ids=frozenset({"A"}), bm25_enabled=True)
    assert list(output) == ["bm25", "vector", "graph"]
    assert [candidate.retrieval_type for candidate in output["bm25"] + output["vector"] + output["graph"]] == ["bm25", "vector", "graph"]
    assert graph.calls == 1


@pytest.mark.asyncio
async def test_v2_orchestrator_empty_scope_fails_closed_without_vector_or_graph_calls():
    graph = Graph()
    retriever = HybridRetrieverV2(bm25=BM25Retriever(Catalog([child("A", 1, "A:v1:c0", "E1007")])), vector_store=Vector(), knowledge_graph=graph)
    assert await retriever.retrieve("E1007", entities=["Atlas"], allowed_document_ids=frozenset(), bm25_enabled=True) == {"bm25": [], "vector": [], "graph": []}
    assert graph.calls == 0
