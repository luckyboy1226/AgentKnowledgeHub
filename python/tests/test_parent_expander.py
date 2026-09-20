"""Fake-only tests for exact parent expansion and final context budgeting."""

from __future__ import annotations

import asyncio

import pytest

from retrieval.parent_expander import ParentExpander, apply_context_budget
from retrieval.reranker import RerankedCandidate


def child(candidate_id: str, rank: int, *, document_id="doc-a", version=2, parent_id="doc-a:v2:p0", chunk_id=None) -> RerankedCandidate:
    return RerankedCandidate(
        candidate_id=candidate_id, content=f"child:{candidate_id}", source="safe.txt",
        document_id=document_id, document_version=version, chunk_id=chunk_id or candidate_id,
        parent_chunk_id=parent_id, retrieval_types=("bm25", "vector"),
        source_ranks={"bm25": rank}, raw_scores={"bm25": 0.5}, rrf_score=1 / (60 + rank),
        rerank_score=1 - rank / 100, pre_rerank_rank=rank, post_rerank_rank=rank,
        metadata={"estimated_token_count": 200},
    )


def graph(rank: int) -> RerankedCandidate:
    return RerankedCandidate(
        candidate_id="graph:e1", content="incorrect fallback", source="safe.txt",
        document_id="doc-a", document_version=2, chunk_id=None, parent_chunk_id=None,
        retrieval_types=("graph",), source_ranks={"graph": rank}, raw_scores={"graph": 0.8},
        rrf_score=1 / (60 + rank), rerank_score=0.9, pre_rerank_rank=rank, post_rerank_rank=rank,
        metadata={"retrieval_metadata": {"graph": {"evidence_edges": [{
            "subject": "A", "predicate": "PROVIDES_INDEX", "object": "B", "direction": "forward",
            "evidence_key": "e1",
        }]}}},
    )


class Catalog:
    def __init__(self, parents):
        self.parents = parents
        self.calls = []

    def get_parent(self, document_id, version, parent_chunk_id):
        self.calls.append((document_id, version, parent_chunk_id))
        return self.parents.get((document_id, version, parent_chunk_id))


def parent(document_id="doc-a", version=2, parent_id="doc-a:v2:p0", *, content="parent body", tokens=600, status="ready", current=True):
    return {
        "kind": "parent", "document_id": document_id, "document_version": version,
        "parent_chunk_id": parent_id, "content": content, "source": "safe.txt",
        "estimated_token_count": tokens, "status": status, "is_current": current,
    }


@pytest.mark.asyncio
async def test_children_for_same_parent_dedupe_and_retain_supporting_children():
    repository = Catalog({("doc-a", 2, "doc-a:v2:p0"): parent()})
    result = await ParentExpander(repository).expand([child("A1", 1), child("A2", 5)])
    assert len(result.contexts) == 1 and result.contexts[0].kind == "parent"
    assert result.contexts[0].supporting_candidate_ids == ("A1", "A2")
    assert result.contexts[0].supporting_child_ids == ("A1", "A2")
    assert result.contexts[0].final_rank == 1
    assert result.diagnostics.parent_expand_success_count == 1


@pytest.mark.asyncio
async def test_parent_lookup_is_exactly_versioned_and_never_uses_old_version():
    repository = Catalog({
        ("doc-a", 1, "doc-a:v2:p0"): parent(version=1),
        ("doc-a", 2, "doc-a:v2:p0"): parent(version=2, content="v2 parent"),
    })
    result = await ParentExpander(repository).expand([child("A", 1)])
    assert result.contexts[0].content == "v2 parent"
    assert repository.calls == [("doc-a", 2, "doc-a:v2:p0")]


@pytest.mark.asyncio
async def test_missing_or_legacy_parent_falls_back_to_exact_child_without_guessing():
    missing = await ParentExpander(Catalog({})).expand([child("A", 1)])
    legacy = await ParentExpander(Catalog({})).expand([child("legacy", 2, parent_id=None)])
    assert missing.contexts[0].kind == "child_fallback" and missing.diagnostics.parent_missing_count == 1
    assert legacy.contexts[0].kind == "child_fallback" and legacy.diagnostics.legacy_child_fallback_count == 1


@pytest.mark.asyncio
async def test_parent_scope_is_fail_closed_and_cannot_fallback_to_unscoped_data():
    repository = Catalog({("doc-a", 2, "doc-a:v2:p0"): parent()})
    rejected = await ParentExpander(repository).expand([child("A", 1)], allowed_document_ids=frozenset({"doc-b"}))
    empty = await ParentExpander(repository).expand([child("A", 1)], allowed_document_ids=frozenset())
    assert rejected.contexts == () and rejected.diagnostics.parent_scope_rejected_count == 1
    assert empty.contexts == () and empty.diagnostics.parent_scope_rejected_count == 1
    assert repository.calls == []


@pytest.mark.asyncio
async def test_scope_also_rejects_graph_evidence_outside_the_exact_document_allowlist():
    result = await ParentExpander(Catalog({})).expand([graph(1)], allowed_document_ids=frozenset({"doc-b"}))
    assert result.contexts == ()
    assert result.diagnostics.parent_scope_rejected_count == 1


@pytest.mark.asyncio
async def test_graph_evidence_remains_directed_and_never_expands_to_parent():
    result = await ParentExpander(Catalog({})).expand([graph(1)])
    context = result.contexts[0]
    assert context.kind == "graph_evidence"
    assert "A --PROVIDES_INDEX--> B" in context.content and "direction: forward" in context.content
    assert "DEPENDS_ON" not in context.content
    assert result.diagnostics.graph_context_count == 1


@pytest.mark.asyncio
async def test_ineligible_parent_status_or_current_state_falls_back_safely():
    repository = Catalog({("doc-a", 2, "doc-a:v2:p0"): parent(status="processing", current=False)})
    result = await ParentExpander(repository).expand([child("A", 1)])
    assert result.contexts[0].kind == "child_fallback"


def test_budget_keeps_complete_parent_then_uses_child_fallback_without_truncation():
    repository = Catalog({
        ("doc-a", 2, "p-a"): parent(parent_id="p-a", content="A" * 600, tokens=600),
        ("doc-a", 2, "p-b"): parent(parent_id="p-b", content="B" * 600, tokens=600),
    })
    result = asyncio.run(ParentExpander(repository).expand([
        child("A", 1, parent_id="p-a"), child("B", 2, parent_id="p-b"),
    ]))
    contexts, diagnostics = apply_context_budget(result.contexts, result.fallback_contexts, top_k=8, token_budget=1000)
    assert [(context.kind, context.supporting_candidate_ids) for context in contexts] == [
        ("parent", ("A",)), ("child_fallback", ("B",)),
    ]
    assert diagnostics["estimated_tokens_used"] == 800
    assert all(context.estimated_token_count <= 600 for context in contexts)


def test_final_top_k_and_order_are_deterministic_across_repeated_runs():
    contexts = []
    for index in range(10):
        item = child(f"c{index}", index + 1, parent_id=None)
        contexts.append(asyncio.run(ParentExpander(Catalog({})).expand([item])).contexts[0])
    expected = ["child:c0", "child:c1", "child:c2"]
    assert all(
        [context.context_id for context in apply_context_budget(contexts, {}, top_k=3, token_budget=9999)[0]] == expected
        for _ in range(100)
    )
