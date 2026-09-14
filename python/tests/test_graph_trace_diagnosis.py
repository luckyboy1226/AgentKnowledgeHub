"""Deterministic first-loss classifier tests; no graph or model is used."""

from __future__ import annotations

import pytest

from services.graph_evidence_trace import edge_fingerprint
from services.graph_trace_diagnosis import diagnose_first_loss


EXPECTED = [("北极星", "PROVIDES_INDEX", "天枢", "forward")]


def _edge(predicate="PROVIDES_INDEX", direction="forward"):
    return {"subject": "北极星", "predicate": predicate, "object": "天枢", "direction": direction}


def _trace(*stages):
    return {"stages": {stage: [_edge()] for stage in stages}, "rejections": []}


@pytest.mark.parametrize(("stages", "expected"), [
    ((), "insufficient_evidence"),
    (("extracted",), "insufficient_evidence"),
    (("extracted", "normalized"), "insufficient_evidence"),
    (("extracted", "normalized", "persisted"), "insufficient_evidence"),
    (("extracted", "normalized", "persisted", "retrieved_raw"), "insufficient_evidence"),
    (("extracted", "normalized", "persisted", "retrieved_raw", "scope_accepted"), "insufficient_evidence"),
    (("extracted", "normalized", "persisted", "retrieved_raw", "scope_accepted", "relevance_scored"), "insufficient_evidence"),
    (("extracted", "normalized", "persisted", "retrieved_raw", "scope_accepted", "relevance_scored", "entered_final_top_k"), "insufficient_evidence"),
    (("extracted", "normalized", "persisted", "retrieved_raw", "scope_accepted", "relevance_scored", "ranked", "entered_final_top_k", "entered_prompt"), "prompt_present"),
])
def test_missing_unobserved_stage_never_causes_false_attribution(stages, expected):
    diagnosis = diagnose_first_loss(EXPECTED, _trace(*stages))
    assert diagnosis["edge_diagnoses"][0]["first_loss"] == expected


def test_empty_extraction_snapshot_is_an_extraction_loss():
    diagnosis = diagnose_first_loss(EXPECTED, {"stages": {"extracted": []}})
    assert diagnosis["edge_diagnoses"][0]["first_loss"] == "extraction_missing"


_ORDERED_STAGES = (
    "extracted", "normalized", "persisted", "retrieved_raw", "scope_accepted",
    "relevance_scored", "ranked", "entered_final_top_k", "entered_prompt",
)


def _trace_with_explicit_empty(first_empty):
    snapshots = {}
    for stage in _ORDERED_STAGES:
        snapshots[stage] = [] if stage == first_empty else [_edge()]
    payload = {"stages": snapshots}
    if first_empty == "scope_accepted":
        payload["rejections"] = [{"fingerprint": edge_fingerprint(_edge())}]
    return payload


@pytest.mark.parametrize(("stage", "expected"), [
    ("normalized", "normalization_changed"),
    ("persisted", "persistence_missing"),
    ("retrieved_raw", "retrieval_missing"),
    ("scope_accepted", "scope_filtered"),
    ("relevance_scored", "relevance_filtered"),
    ("entered_final_top_k", "top_k_truncated"),
])
def test_first_explicitly_empty_transition_is_classified(stage, expected):
    diagnosis = diagnose_first_loss(EXPECTED, _trace_with_explicit_empty(stage))
    assert diagnosis["edge_diagnoses"][0]["first_loss"] == expected


@pytest.mark.parametrize("predicate,direction", [
    ("DEPENDS_ON", "forward"), ("PROVIDES_INDEX", "reverse"),
    ("PROVIDES_INDEX", "forward"), ("提供检索索引", "forward"),
])
def test_predicate_and_direction_are_scored_separately(predicate, direction):
    trace = {"stages": {stage: [_edge(predicate, direction)] for stage in (
        "extracted", "normalized", "persisted", "retrieved_raw", "scope_accepted",
        "relevance_scored", "ranked", "entered_final_top_k", "entered_prompt",
    )}}
    result = diagnose_first_loss(EXPECTED, trace)
    expected = "prompt_present" if (predicate in {"PROVIDES_INDEX", "提供检索索引"} and direction == "forward") else "extraction_missing"
    assert result["edge_diagnoses"][0]["first_loss"] == expected


def test_multi_hop_is_diagnosed_edge_by_edge_and_requires_complete_prompt_path():
    expected = [
        ("A", "DEPENDS_ON", "B", "forward"),
        ("B", "RESPONSIBLE_FOR", "C", "forward"),
    ]
    complete_edge = {"subject": "A", "predicate": "DEPENDS_ON", "object": "B", "direction": "forward"}
    trace = {"stages": {stage: [complete_edge] for stage in (
        "extracted", "normalized", "persisted", "retrieved_raw", "scope_accepted",
        "relevance_scored", "ranked", "entered_final_top_k", "entered_prompt",
    )}}
    result = diagnose_first_loss(expected, trace)
    assert result["edge_diagnoses"][0]["first_loss"] == "prompt_present"
    assert result["edge_diagnoses"][1]["first_loss"] == "extraction_missing"
    assert result["complete_path_entered_prompt"] is False


def test_diagnosis_exposes_separate_predicate_direction_and_endpoint_fields():
    trace = {"stages": {"extracted": [_edge("DEPENDS_ON", "reverse")]}}
    edge = diagnose_first_loss(EXPECTED, trace)["edge_diagnoses"][0]
    assert edge["endpoint_match"] is True
    assert edge["canonical_predicate_match"] is False
    assert edge["direction_match"] is False
    assert edge["failure_category"] == "predicate_mismatch"
    assert edge["evidence_sufficient"] is False
