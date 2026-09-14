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


# This is intentionally separate from ``FirstLoss``.  The latter is the
# original public result vocabulary used by the S4.6a fake-only contracts.
# The expanded vocabulary is used by the immutable-run analyzer and makes the
# evidence boundary explicit without changing historic reports.
_DETAILED_STAGE = {
    FirstLoss.EXTRACTION_MISSING: "lost_at_extraction",
    FirstLoss.NORMALIZATION_CHANGED: "lost_at_normalization",
    FirstLoss.PERSISTENCE_MISSING: "lost_at_persistence",
    FirstLoss.RETRIEVAL_MISSING: "lost_at_retrieval",
    FirstLoss.SCOPE_FILTERED: "rejected_by_scope",
    FirstLoss.RELEVANCE_FILTERED: "lost_at_relevance_filter",
    FirstLoss.TOP_K_TRUNCATED: "lost_at_top_k",
    FirstLoss.PROMPT_PRESENT: "entered_prompt",
    FirstLoss.INSUFFICIENT_EVIDENCE: "insufficient_evidence",
}

_ORDERED_STAGES = (
    GraphTraceStage.EXTRACTED.value,
    GraphTraceStage.NORMALIZED.value,
    GraphTraceStage.PERSISTED.value,
    GraphTraceStage.RETRIEVED_RAW.value,
    GraphTraceStage.SCOPE_ACCEPTED.value,
    GraphTraceStage.RELEVANCE_SCORED.value,
    GraphTraceStage.RANKED.value,
    GraphTraceStage.ENTERED_FINAL_TOP_K.value,
    GraphTraceStage.ENTERED_PROMPT.value,
)


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


def _provenance_complete(edge: dict[str, Any]) -> bool:
    """A graph edge is safe to attribute only with its complete provenance."""
    return bool(
        str(edge.get("document_id", "")).strip()
        and isinstance(edge.get("document_version"), int)
        and not isinstance(edge.get("document_version"), bool)
        and str(edge.get("source", "")).strip()
        and str(edge.get("evidence_key", "")).strip()
    )


def _stage_counts(expected: ExpectedTraceEdge, snapshot: object) -> dict[str, int | bool]:
    """Return independent endpoint/predicate/direction observations.

    A false exact match must not be collapsed into an endpoint mismatch: that
    distinction is the core safeguard against blaming a predicate conversion
    on retrieval.  An absent snapshot is represented explicitly rather than
    as an empty list.
    """
    if not isinstance(snapshot, list):
        return {
            "snapshot_present": False, "edge_count": 0, "exact_matches": 0,
            "endpoint_matches": 0, "predicate_matches": 0,
            "direction_matches": 0, "provenance_complete_matches": 0,
        }
    edges = [edge for edge in snapshot if isinstance(edge, dict)]
    endpoint = [edge for edge in edges if _endpoint_match(expected, edge)]
    predicate = [edge for edge in edges if _predicate_match(expected, edge)]
    direction = [
        edge for edge in edges
        if _endpoint_match(expected, edge) and _predicate_match(expected, edge)
        and str(edge.get("direction", "")).strip() == expected.direction
    ]
    exact = [edge for edge in direction if _matches(expected, edge)]
    return {
        "snapshot_present": True,
        "edge_count": len(edges),
        "exact_matches": len(exact),
        "endpoint_matches": len(endpoint),
        "predicate_matches": len(predicate),
        "direction_matches": len(direction),
        "provenance_complete_matches": sum(_provenance_complete(edge) for edge in exact),
    }


def _detailed_first_loss(
    stage_counts: dict[str, dict[str, int | bool]], rejections: object, raw_snapshot: object,
) -> tuple[FirstLoss, str, str]:
    """Classify only an observed adjacent transition.

    There is deliberately no fallback from ``retrieved_raw == 0`` to an
    ingestion conclusion: the real S4.6b trace begins at retrieval.  The
    analyzer therefore returns ``insufficient_evidence`` unless extraction,
    normalization and persistence snapshots are all actually present.
    """
    first = _ORDERED_STAGES[0]
    if not bool(stage_counts[first]["snapshot_present"]):
        return FirstLoss.INSUFFICIENT_EVIDENCE, "missing_extraction_snapshot", "pre-retrieval snapshots were not persisted"
    if int(stage_counts[first]["exact_matches"]) == 0:
        return FirstLoss.EXTRACTION_MISSING, "extracted_snapshot_has_no_exact_edge", "extraction snapshot is present and has no exact edge"

    transitions = (
        (GraphTraceStage.NORMALIZED.value, FirstLoss.NORMALIZATION_CHANGED, "normalized snapshot lacks exact edge"),
        (GraphTraceStage.PERSISTED.value, FirstLoss.PERSISTENCE_MISSING, "persisted snapshot lacks exact edge"),
        (GraphTraceStage.RETRIEVED_RAW.value, FirstLoss.RETRIEVAL_MISSING, "retrieved raw snapshot lacks exact edge"),
        (GraphTraceStage.SCOPE_ACCEPTED.value, FirstLoss.SCOPE_FILTERED, "scope accepted snapshot lacks exact edge"),
        (GraphTraceStage.RELEVANCE_SCORED.value, FirstLoss.RELEVANCE_FILTERED, "relevance snapshot lacks exact edge"),
        (GraphTraceStage.RANKED.value, FirstLoss.RELEVANCE_FILTERED, "ranked snapshot lacks exact edge"),
        (GraphTraceStage.ENTERED_FINAL_TOP_K.value, FirstLoss.TOP_K_TRUNCATED, "final Top-K snapshot lacks exact edge"),
        (GraphTraceStage.ENTERED_PROMPT.value, FirstLoss.TOP_K_TRUNCATED, "prompt snapshot lacks exact edge"),
    )
    previous = first
    for stage, loss, reason in transitions:
        if not bool(stage_counts[stage]["snapshot_present"]):
            return FirstLoss.INSUFFICIENT_EVIDENCE, f"missing_{stage}_snapshot", "an adjacent stage snapshot was not persisted"
        if int(stage_counts[previous]["exact_matches"]) > 0 and int(stage_counts[stage]["exact_matches"]) == 0:
            if stage == GraphTraceStage.SCOPE_ACCEPTED.value:
                # Scope is special: a missing accepted edge proves nothing
                # without a rejection record for the retrieved identity.
                rejected = {
                    str(item.get("fingerprint", "")) for item in rejections
                    if isinstance(item, dict)
                } if isinstance(rejections, list) else set()
                raw_matches = {
                    edge_fingerprint(edge) for edge in raw_snapshot
                    if isinstance(raw_snapshot, list) and isinstance(edge, dict)
                }
                if not rejected.intersection(raw_matches):
                    return FirstLoss.INSUFFICIENT_EVIDENCE, "scope_snapshot_without_matching_rejection", "scope transition has no matching rejection evidence"
            if stage == GraphTraceStage.RANKED.value:
                return loss, "ranked_snapshot_lacks_exact_edge", reason
            if stage == GraphTraceStage.ENTERED_PROMPT.value:
                return loss, "prompt_snapshot_lacks_exact_edge", "edge entered final Top-K but not prompt"
            return loss, f"{stage}_snapshot_lacks_exact_edge", reason
        previous = stage
    return FirstLoss.PROMPT_PRESENT, "exact_edge_entered_prompt", "exact edge appears in every observed stage through prompt"


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
    results: list[dict[str, Any]] = []
    for expected in normalized_expected:
        stage_counts = {stage: _stage_counts(expected, stages.get(stage)) for stage in _ORDERED_STAGES}
        matched_stages = [
            stage for stage in _ORDERED_STAGES
            if int(stage_counts[stage]["exact_matches"]) > 0
        ]
        first_loss, loss_reason, evidence_boundary = _detailed_first_loss(
            stage_counts,
            rejections,
            [
                edge for edge in stages.get(GraphTraceStage.RETRIEVED_RAW.value, [])
                if isinstance(edge, dict) and _matches(expected, edge)
            ],
        )
        detailed_stage = _DETAILED_STAGE[first_loss]
        if loss_reason == "ranked_snapshot_lacks_exact_edge":
            detailed_stage = "lost_at_ranking"
        elif loss_reason == "prompt_snapshot_lacks_exact_edge":
            detailed_stage = "lost_before_prompt"
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
            "evidence_sufficient": all(bool(stage_counts[stage]["snapshot_present"]) for stage in _ORDERED_STAGES),
            "failure_category": failure_category,
            "first_loss_stage": detailed_stage,
            "loss_reason": loss_reason,
            "confidence": "high" if first_loss is not FirstLoss.INSUFFICIENT_EVIDENCE else "low",
            "evidence_boundary": evidence_boundary,
            "stage_counts": stage_counts,
        })
    complete = bool(results) and all(item["first_loss"] == FirstLoss.PROMPT_PRESENT.value for item in results)
    return {"edge_diagnoses": results, "complete_path_entered_prompt": complete}
