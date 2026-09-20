"""Fake-only deterministic tests for the isolated Phase C RRF service."""

from __future__ import annotations

import inspect
import math

import pytest

from agents.qa_agent import QAAgent
from config.settings import Settings
from retrieval.candidates import RetrievalCandidate
from retrieval.fusion import RRFFusion


def candidate(
    candidate_id: str,
    retrieval_type: str,
    rank: int | None,
    raw_score: float | None = 0.5,
    *,
    document_id: str = "doc-a",
    version: int = 1,
    chunk_id: str | None = None,
    parent_chunk_id: str | None = "doc-a:v1:p0",
    source: str = "safe.txt",
    metadata: dict | None = None,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        candidate_id=candidate_id,
        content=f"content:{candidate_id}",
        source=source,
        document_id=document_id,
        document_version=version,
        chunk_id=chunk_id or candidate_id,
        parent_chunk_id=parent_chunk_id,
        retrieval_type=retrieval_type,  # type: ignore[arg-type]
        raw_score=raw_score,
        rank=rank,
        metadata=dict(metadata or {}),
    )


def test_rrf_accumulates_exact_rank_contributions_and_keeps_provenance():
    fused = RRFFusion(rrf_k=60).fuse({
        "bm25": [candidate("A", "bm25", 1), candidate("B", "bm25", 2), candidate("C", "bm25", 3)],
        "vector": [candidate("B", "vector", 1), candidate("D", "vector", 2), candidate("A", "vector", 3)],
        "graph": [candidate("graph:e", "graph", 1, chunk_id=None, parent_chunk_id=None)],
    }, top_k=10)
    by_id = {item.candidate_id: item for item in fused.candidates}
    assert math.isclose(by_id["A"].rrf_score, 1 / 61 + 1 / 63)
    assert math.isclose(by_id["B"].rrf_score, 1 / 62 + 1 / 61)
    assert by_id["A"].retrieval_types == ("bm25", "vector")
    assert by_id["A"].source_ranks == {"bm25": 1, "vector": 3}
    assert by_id["A"].raw_scores == {"bm25": 0.5, "vector": 0.5}
    assert set(by_id["A"].metadata["retrieval_metadata"]) == {"bm25", "vector"}
    assert fused.diagnostics.input_counts == {"bm25": 3, "vector": 3, "graph": 1}
    assert fused.diagnostics.multi_source_candidate_count == 2


def test_same_child_merges_but_graph_evidence_with_different_identity_does_not():
    result = RRFFusion().fuse({
        "bm25": [candidate("doc-a:v1:c0", "bm25", 2)],
        "vector": [candidate("doc-a:v1:c0", "vector", 1)],
        "graph": [candidate("graph:edge-a", "graph", 1, chunk_id=None, parent_chunk_id=None)],
    })
    assert [item.candidate_id for item in result.candidates] == ["doc-a:v1:c0", "graph:edge-a"]
    assert result.candidates[0].retrieval_types == ("bm25", "vector")
    assert result.candidates[1].retrieval_types == ("graph",)


def test_raw_scores_do_not_cross_retriever_score_spaces():
    result = RRFFusion().fuse({
        "bm25": [candidate("A", "bm25", 2, raw_score=1000.0)],
        "vector": [candidate("B", "vector", 1, raw_score=0.01)],
        "graph": [],
    })
    assert [item.candidate_id for item in result.candidates] == ["B", "A"]
    assert result.candidates[0].rrf_score == pytest.approx(1 / 61)


def test_tie_break_is_stable_across_repeated_runs():
    fusion = RRFFusion()
    inputs = {
        "bm25": [candidate("z", "bm25", 1), candidate("a", "bm25", 2)],
        "vector": [candidate("a", "vector", 1), candidate("z", "vector", 2)],
        "graph": [],
    }
    expected = ["a", "z"]  # equal score/count/rank, candidate ID is final tie-break.
    assert all([item.candidate_id for item in fusion.fuse(inputs).candidates] == expected for _ in range(100))


def test_provenance_conflict_rejects_the_entire_identity_instead_of_merging():
    result = RRFFusion().fuse({
        "bm25": [candidate("same", "bm25", 1, version=1)],
        "vector": [candidate("same", "vector", 1, version=2)],
        "graph": [],
    })
    assert result.candidates == ()
    assert result.diagnostics.provenance_conflict_count == 1


@pytest.mark.parametrize("rank", [None, 0, -1])
def test_invalid_rank_is_skipped_and_recorded(rank):
    result = RRFFusion().fuse({
        "bm25": [candidate("bad", "bm25", rank)], "vector": [], "graph": [],
    })
    assert result.candidates == ()
    assert result.diagnostics.invalid_candidate_count == 1


def test_duplicate_source_rank_wrong_bucket_and_unknown_retriever_are_rejected():
    result = RRFFusion().fuse({
        "bm25": [candidate("a", "bm25", 1), candidate("b", "bm25", 1)],
        "vector": [candidate("wrong", "bm25", 2)],
        "graph": [], "unknown": [candidate("unknown", "bm25", 1)],
    })
    assert result.candidates == ()
    assert result.diagnostics.invalid_candidate_count == 4


def test_top_k_is_applied_after_stable_rrf_ordering():
    rows = [candidate(f"c{index:02d}", "bm25", index + 1) for index in range(50)]
    result = RRFFusion(fusion_top_k=30).fuse({"bm25": rows, "vector": [], "graph": []})
    assert len(result.candidates) == 30
    assert [item.candidate_id for item in result.candidates[:3]] == ["c00", "c01", "c02"]
    assert result.diagnostics.unique_candidate_count == 50


def test_defensive_lifecycle_scope_filter_never_revives_invalid_candidates():
    result = RRFFusion().fuse({
        "bm25": [
            candidate("old", "bm25", 1, metadata={"status": "ready", "is_current": False}),
            candidate("foreign", "bm25", 2, metadata={"scope_accepted": False}),
            candidate("safe", "bm25", 3, metadata={"status": "ready", "is_current": True}),
        ],
        "vector": [], "graph": [],
    })
    assert [item.candidate_id for item in result.candidates] == ["safe"]
    assert result.diagnostics.invalid_candidate_count == 2
    assert set(item.candidate_id for item in result.candidates) <= {"old", "foreign", "safe"}


def test_v1_default_remains_enabled_and_does_not_consume_rrf():
    assert Settings().hybrid_retrieval_v2_enabled is False
    answer_source = inspect.getsource(QAAgent.answer)
    assert "_hybrid_rerank" in answer_source
    assert "RRFFusion" not in answer_source
