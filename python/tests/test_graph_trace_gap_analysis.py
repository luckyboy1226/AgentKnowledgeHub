"""Fake-only contracts for S4.8a relation-gap attribution."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from services.graph_trace_gap_analysis import (
    GraphTraceGapAnalysisError,
    classify_extraction_gap,
    classify_retrieval_gap,
    endpoint_alias_match,
    immutable_input_hashes,
)


TARGET = {"subject": "北极星检索平台", "predicate": "DEPENDS_ON", "object": "Atlas事件服务", "direction": "forward"}


def _edge(**changes):
    base = {
        "subject": "北极星检索平台", "predicate": "DEPENDS_ON", "raw_predicate": "依赖",
        "object": "Atlas事件服务", "direction": "forward", "document_id": "a" * 36,
        "document_version": 1, "source": "fixture.txt", "evidence_key": "e-1", "stage": "extracted",
    }
    base.update(changes)
    return base


def test_exact_edge_absent_but_synonym_raw_predicate_is_not_called_no_relation():
    result = classify_extraction_gap(
        {**TARGET, "predicate": "MONITORS", "subject": "流光监控平台"},
        [_edge(subject="流光", predicate="RELATED_TO", raw_predicate="监控", object="Atlas", direction="forward")],
        stage_observed=True,
    )
    assert result["classification"] == "canonicalization_gap"


def test_raw_predicate_semantic_gap_remains_distinct_from_exact_edge_absence():
    target = {"subject": "陈航", "predicate": "HAS_ROLE", "object": "数据工程师", "direction": "forward"}
    result = classify_extraction_gap(
        target,
        [_edge(subject="陈航", predicate="WORKS_AT", raw_predicate="是数据工程师", object="数据平台部")],
        stage_observed=True,
    )
    assert result["classification"] == "raw_predicate_semantic_gap"


def test_endpoint_alias_mismatch_is_explicit():
    result = classify_extraction_gap(TARGET, [_edge(object="Atlas 事件服务")], stage_observed=True)
    assert result["classification"] == "endpoint_alias_mismatch"
    assert endpoint_alias_match("北极星", "北极星检索平台")


def test_direction_reversal_is_not_silently_accepted():
    result = classify_extraction_gap(TARGET, [_edge(subject="Atlas事件服务", object="北极星检索平台")], stage_observed=True)
    assert result["classification"] == "direction_reversed"


def test_safety_downgrade_is_canonicalization_gap():
    result = classify_extraction_gap(
        {**TARGET, "predicate": "NOT_DEPENDS_ON", "object": "天枢知识平台"},
        [_edge(predicate="RELATED_TO", raw_predicate="不依赖", object="天枢知识平台")], stage_observed=True,
    )
    assert result["classification"] == "canonicalization_gap"


def test_cross_chunk_is_insufficient_without_chunk_metadata():
    result = classify_extraction_gap(TARGET, [], stage_observed=True)
    assert result["cross_chunk_separation"] == "insufficient_evidence"
    assert result["chunk_index_observed"] is False


def test_observed_empty_relations_are_no_relation_not_entity_absence():
    result = classify_extraction_gap(TARGET, [], stage_observed=True)
    assert result["classification"] == "no_relation_extracted"
    assert result["entity_not_extracted"] == "insufficient_evidence"


def test_retrieval_alias_edge_is_not_traversal_failure():
    stages = {
        "persisted": [_edge(subject="北极星", object="Atlas 事件服务")],
        "retrieved_raw": [_edge(subject="北极星", object="Atlas 事件服务")],
        "entered_prompt": [_edge(subject="北极星", object="Atlas 事件服务")],
    }
    result = classify_retrieval_gap(TARGET, stages)
    assert result["classification"] == "endpoint_alias_mismatch"
    assert result["retrieved_raw_alias_count"] == 1


def test_persisted_edge_without_query_trace_is_insufficient_not_traversal_blame():
    stages = {"persisted": [_edge()], "retrieved_raw": []}
    result = classify_retrieval_gap(TARGET, stages)
    assert result["classification"] == "insufficient_evidence"
    assert result["query_hops_observed"] is False


def test_persisted_edge_with_traced_wrong_query_entity_is_query_entity_missing():
    stages = {
        "persisted": [_edge()], "retrieved_raw": [],
        "query_trace": {"entities": ["无关实体"], "max_hops": 2, "traversal_observed": True},
    }
    result = classify_retrieval_gap(TARGET, stages)
    assert result["classification"] == "persisted_but_query_entity_missing"


def test_persisted_edge_with_traced_traversal_miss_is_not_blended_with_ranking():
    stages = {
        "persisted": [_edge()], "retrieved_raw": [],
        "query_trace": {"entities": ["北极星检索平台"], "max_hops": 1, "traversal_observed": True},
    }
    result = classify_retrieval_gap(TARGET, stages)
    assert result["classification"] == "persisted_but_traversal_missing"


def test_retrieved_exact_edge_missing_from_ranked_is_ranking_missing():
    stages = {"persisted": [_edge()], "retrieved_raw": [_edge()], "ranked": []}
    result = classify_retrieval_gap(TARGET, stages)
    assert result["classification"] == "persisted_but_ranking_missing"


def test_missing_trace_fields_fail_closed():
    result = classify_extraction_gap(TARGET, [], stage_observed=False)
    assert result["classification"] == "insufficient_evidence"


def test_only_bounded_edge_fields_are_exposed():
    result = classify_extraction_gap(TARGET, [_edge(raw_predicate="safe")], stage_observed=True)
    rendered = json.dumps(result, ensure_ascii=False)
    assert "C:\\" not in rendered
    assert "mongodb://" not in rendered
    assert "sk-" not in rendered


def test_immutable_input_hashes_cover_all_five_files(tmp_path):
    names = ("graph-ingestion-trace.jsonl", "graph-trace.json", "results.json", "safe-run-metadata.json", "ingestion-state.json")
    for name in names:
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    before = immutable_input_hashes(tmp_path)
    after = immutable_input_hashes(tmp_path)
    assert before == after
    assert before["results.json"] == hashlib.sha256((tmp_path / "results.json").read_bytes()).hexdigest()


def test_hashing_refuses_missing_immutable_input(tmp_path):
    with pytest.raises(GraphTraceGapAnalysisError, match="missing_input"):
        immutable_input_hashes(tmp_path)


def test_offline_gap_analyser_does_not_initialise_provider_or_database_clients():
    # Importing this module is the only operation exercised in this file.  Its
    # dependency boundary must remain JSON/fixture/semantics only.
    assert "pymongo" not in sys.modules
    assert "neo4j" not in sys.modules
