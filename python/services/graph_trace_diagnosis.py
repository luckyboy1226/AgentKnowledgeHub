"""Deterministic first-loss diagnosis over safe GraphEvidenceTrace snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from services.graph_evidence_trace import GraphTraceStage, edge_fingerprint
from services.relation_semantics import canonicalize_relation


class FirstLoss(str, Enum):
    EXTRACTION_MISSING = "extraction_missing"
    NORMALIZATION_CHANGED = "normalization_changed"
    PERSISTENCE_MISSING = "persistence_missing"
    RETRIEVAL_MISSING = "retrieval_missing"
    SCOPE_FILTERED = "scope_filtered"
    RELEVANCE_FILTERED = "relevance_filtered"
    TOP_K_TRUNCATED = "top_k_truncated"
    PROMPT_PRESENT = "prompt_present"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True)
class ExpectedTraceEdge:
    subject: str
    predicate: str
    object: str
    direction: str

    @classmethod
    def from_value(cls, value: Iterable[object]) -> "ExpectedTraceEdge":
        parts = tuple(value)
        if len(parts) not in {3, 4}:
            raise ValueError("expected relation edge must contain 3 or 4 values")
        semantic = canonicalize_relation(str(parts[1]))
        return cls(str(parts[0]).strip(), semantic.predicate, str(parts[2]).strip(), str(parts[3]).strip() if len(parts) == 4 else "forward")


def _matches(expected: ExpectedTraceEdge, edge: dict[str, Any]) -> bool:
    predicate = canonicalize_relation(str(edge.get("predicate", "")), raw_predicate=edge.get("raw_predicate")).predicate
    return (
        str(edge.get("subject", "")).strip() == expected.subject
        and predicate == expected.predicate
        and str(edge.get("object", "")).strip() == expected.object
        and str(edge.get("direction", "")).strip() == expected.direction
    )


def _endpoint_match(expected: ExpectedTraceEdge, edge: dict[str, Any]) -> bool:
    return str(edge.get("subject", "")).strip() == expected.subject and str(edge.get("object", "")).strip() == expected.object


def _predicate_match(expected: ExpectedTraceEdge, edge: dict[str, Any]) -> bool:
    return canonicalize_relation(
        str(edge.get("predicate", "")), raw_predicate=edge.get("raw_predicate")
    ).predicate == expected.predicate


def diagnose_first_loss(expected_path: Iterable[Iterable[object]], trace: dict[str, Any]) -> dict[str, Any]:
    """Classify each expected edge without inferring an unavailable snapshot.

    A missing stage cannot prove an earlier loss. Such records are deliberately
    labelled ``insufficient_evidence`` rather than blamed on extraction or a
    storage layer.
    """
    stages = trace.get("stages") if isinstance(trace, dict) else None
    if not isinstance(stages, dict):
        stages = {}
    rejections = trace.get("rejections") if isinstance(trace, dict) else []
    normalized_expected = [ExpectedTraceEdge.from_value(edge) for edge in expected_path]
    ordered = (
        (GraphTraceStage.EXTRACTED.value, FirstLoss.EXTRACTION_MISSING),
        (GraphTraceStage.NORMALIZED.value, FirstLoss.NORMALIZATION_CHANGED),
        (GraphTraceStage.PERSISTED.value, FirstLoss.PERSISTENCE_MISSING),
        (GraphTraceStage.RETRIEVED_RAW.value, FirstLoss.RETRIEVAL_MISSING),
        (GraphTraceStage.SCOPE_ACCEPTED.value, FirstLoss.SCOPE_FILTERED),
        (GraphTraceStage.RELEVANCE_SCORED.value, FirstLoss.RELEVANCE_FILTERED),
        (GraphTraceStage.ENTERED_FINAL_TOP_K.value, FirstLoss.TOP_K_TRUNCATED),
        (GraphTraceStage.ENTERED_PROMPT.value, FirstLoss.PROMPT_PRESENT),
    )
    results: list[dict[str, Any]] = []
    for expected in normalized_expected:
        first_loss = FirstLoss.INSUFFICIENT_EVIDENCE
        matched_stages: list[str] = []
        previous_present = False
        for stage, missing_loss in ordered:
            snapshot = stages.get(stage)
            if not isinstance(snapshot, list):
                # No snapshot means no causal inference can be made.
                break
            present = any(isinstance(edge, dict) and _matches(expected, edge) for edge in snapshot)
            if present:
                matched_stages.append(stage)
                previous_present = True
                continue
            if previous_present:
                first_loss = missing_loss
            elif stage == GraphTraceStage.EXTRACTED.value:
                first_loss = FirstLoss.EXTRACTION_MISSING
            break
        else:
            first_loss = FirstLoss.PROMPT_PRESENT if GraphTraceStage.ENTERED_PROMPT.value in matched_stages else FirstLoss.INSUFFICIENT_EVIDENCE
        # A scope loss requires a concrete rejection of the same raw edge;
        # merely lacking an accepted snapshot is not enough to blame scope.
        if first_loss is FirstLoss.SCOPE_FILTERED:
            raw_edges = stages.get(GraphTraceStage.RETRIEVED_RAW.value, [])
            matching_fingerprints = {
                edge_fingerprint(edge) for edge in raw_edges
                if isinstance(edge, dict) and _matches(expected, edge)
            }
            rejected_fingerprints = {
                str(item.get("fingerprint")) for item in rejections
                if isinstance(item, dict) and item.get("fingerprint")
            } if isinstance(rejections, list) else set()
            if not matching_fingerprints.intersection(rejected_fingerprints):
                first_loss = FirstLoss.INSUFFICIENT_EVIDENCE
        all_edges = [edge for snapshot in stages.values() if isinstance(snapshot, list) for edge in snapshot if isinstance(edge, dict)]
        endpoint_match = any(_endpoint_match(expected, edge) for edge in all_edges)
        canonical_predicate_match = any(_predicate_match(expected, edge) for edge in all_edges)
        direction_match = any(
            _endpoint_match(expected, edge) and _predicate_match(expected, edge)
            and str(edge.get("direction", "")).strip() == expected.direction
            for edge in all_edges
        )
        prompt_edges = stages.get(GraphTraceStage.ENTERED_PROMPT.value, [])
        exact_prompt_match = isinstance(prompt_edges, list) and any(
            isinstance(edge, dict) and _matches(expected, edge) for edge in prompt_edges
        )
        if not endpoint_match:
            failure_category = "endpoint_mismatch"
        elif not canonical_predicate_match:
            failure_category = "predicate_mismatch"
        elif not direction_match:
            failure_category = "direction_mismatch"
        else:
            failure_category = first_loss.value
        expected_mapping = {"subject": expected.subject, "predicate": expected.predicate, "object": expected.object, "direction": expected.direction}
        results.append({
            "expected": expected_mapping,
            "expected_edge_fingerprint": edge_fingerprint(expected_mapping),
            "matched_stages": matched_stages,
            "first_loss": first_loss.value,
            "exact_prompt_match": bool(exact_prompt_match),
            "canonical_predicate_match": canonical_predicate_match,
            "direction_match": direction_match,
            "endpoint_match": endpoint_match,
            "evidence_sufficient": all(stage in stages for stage, _loss in ordered),
            "failure_category": failure_category,
        })
    complete = bool(results) and all(item["first_loss"] == FirstLoss.PROMPT_PRESENT.value for item in results)
    return {"edge_diagnoses": results, "complete_path_entered_prompt": complete}
