"""Offline-only fine-grained diagnosis for an immutable graph trace journal.

This module deliberately depends only on JSON files, the reviewed fixture and
relation canonicalisation.  It neither creates providers nor imports database
clients, so it is safe to use on a completed production-like evaluation run.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from services.rag_evaluation import load_evaluation_fixture, select_evaluation_subset
from services.relation_semantics import canonicalize_relation


ANALYSIS_SCHEMA_VERSION = "s4.8a-gap-analysis-v1"
_INPUTS = (
    "graph-ingestion-trace.jsonl", "graph-trace.json", "results.json",
    "safe-run-metadata.json", "ingestion-state.json",
)
_SAFE_NAME = re.compile(r"^[^\\/:*?\"<>|\x00-\x1f]+$")
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,120}$")


class GraphTraceGapAnalysisError(ValueError):
    """Raised when evidence cannot be safely analysed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def immutable_input_hashes(trace_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in _INPUTS:
        path = trace_dir / name
        if not path.is_file():
            raise GraphTraceGapAnalysisError(f"missing_input_{name}")
        hashes[name] = _sha256(path)
    return hashes


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GraphTraceGapAnalysisError(f"invalid_json_{path.name}") from exc
    if not isinstance(payload, dict):
        raise GraphTraceGapAnalysisError(f"invalid_json_root_{path.name}")
    return payload


def _read_journal(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError("journal_item")
            events.append(item)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise GraphTraceGapAnalysisError("invalid_ingestion_journal") from exc
    return events


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _safe_source(value: object) -> str:
    source = str(value or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not source or not _SAFE_NAME.fullmatch(source):
        raise GraphTraceGapAnalysisError("unsafe_source")
    return source


def _compact(value: object) -> str:
    return re.sub(r"[\s\-_]", "", str(value or "")).strip()


def endpoint_alias_match(expected: object, observed: object) -> bool:
    """Conservative, explainable short-name matching; never a retrieval fallback."""
    left, right = _compact(expected), _compact(observed)
    return bool(left and right and (left == right or (min(len(left), len(right)) >= 2 and (left in right or right in left))))


def _predicate(edge: dict[str, Any]) -> str:
    return canonicalize_relation(
        str(edge.get("canonical_predicate") or edge.get("predicate") or ""),
        raw_predicate=edge.get("raw_predicate"),
    ).predicate


def _edge(event: dict[str, Any]) -> dict[str, Any]:
    """Return bounded identity fields only; never copy text, prompts or answers."""
    return {
        "subject": str(event.get("subject") or "").strip(),
        "predicate": _predicate(event),
        "raw_predicate": " ".join(str(event.get("raw_predicate") or "").split())[:160],
        "object": str(event.get("object") or "").strip(),
        "direction": str(event.get("direction") or "").strip(),
        "document_id": str(event.get("document_id") or "").strip(),
        "document_version": event.get("document_version"),
        "source": _safe_source(event.get("source")),
        "evidence_key": str(event.get("evidence_key") or "").strip(),
        "stage": str(event.get("stage") or "").strip(),
        "reason": str(event.get("reason") or "").strip()[:80] or None,
    }


def _exact(target: dict[str, str], edge: dict[str, Any]) -> bool:
    return (
        target["subject"] == edge["subject"] and target["predicate"] == edge["predicate"]
        and target["object"] == edge["object"] and target["direction"] == edge["direction"]
    )


def _aliased_endpoints(target: dict[str, str], edge: dict[str, Any]) -> bool:
    return endpoint_alias_match(target["subject"], edge["subject"]) and endpoint_alias_match(target["object"], edge["object"])


def _reverse_endpoints(target: dict[str, str], edge: dict[str, Any]) -> bool:
    return endpoint_alias_match(target["subject"], edge["object"]) and endpoint_alias_match(target["object"], edge["subject"])


def _predicate_cues(predicate: str) -> tuple[str, ...]:
    return {
        "HAS_ROLE": ("工程师", "架构师", "负责人", "角色", "职位"),
        "MONITORS": ("监控",),
        "NOT_DEPENDS_ON": ("不依赖", "not_depends"),
        "DEPENDS_ON": ("依赖",),
        "PROVIDES_INDEX": ("索引",),
    }.get(predicate, ())


def _raw_has_target_semantics(target: dict[str, str], edge: dict[str, Any]) -> bool:
    raw = _compact(edge["raw_predicate"]).lower()
    return any(_compact(cue).lower() in raw for cue in _predicate_cues(target["predicate"]))


def classify_extraction_gap(target: dict[str, str], extracted: Iterable[dict[str, Any]], *, stage_observed: bool, chunk_index_observed: bool = False) -> dict[str, Any]:
    """Classify a target absent as an exact relation without inventing entities.

    The journal records relations, not a complete entity inventory or chunk
    positions.  Consequently the function can confirm relation-shape evidence
    but must leave entity absence and cross-chunk claims unproven.
    """
    edges = list(extracted)
    exact = [edge for edge in edges if _exact(target, edge)]
    endpoint_alias = [edge for edge in edges if _aliased_endpoints(target, edge)]
    reversed_edges = [edge for edge in edges if _reverse_endpoints(target, edge)]
    subject_mentions = [edge for edge in edges if endpoint_alias_match(target["subject"], edge["subject"]) or endpoint_alias_match(target["subject"], edge["object"])]
    object_mentions = [edge for edge in edges if endpoint_alias_match(target["object"], edge["subject"]) or endpoint_alias_match(target["object"], edge["object"])]
    semantic_candidates = [edge for edge in edges if _raw_has_target_semantics(target, edge)]
    same_endpoint_semantic = [edge for edge in endpoint_alias if _raw_has_target_semantics(target, edge)]
    canonical_changed = [edge for edge in same_endpoint_semantic if edge["predicate"] != target["predicate"]]
    reversed_target = [edge for edge in reversed_edges if edge["predicate"] == target["predicate"]]

    classification, strength, reason = "insufficient_evidence", "insufficient_evidence", "no_extraction_stage_marker"
    if not stage_observed:
        pass
    elif exact:
        classification, strength, reason = "insufficient_evidence", "confirmed", "exact_edge_was_extracted"
    elif reversed_target:
        classification, strength, reason = "direction_reversed", "confirmed", "expected_predicate_with_reversed_endpoints"
    elif canonical_changed:
        classification, strength, reason = "canonicalization_gap", "confirmed", "target_semantics_raw_predicate_canonicalized_differently"
    elif any(edge["predicate"] == target["predicate"] for edge in endpoint_alias):
        classification, strength, reason = "endpoint_alias_mismatch", "confirmed", "target_predicate_with_endpoint_alias_variation"
    elif semantic_candidates:
        classification, strength, reason = "raw_predicate_semantic_gap", "probable", "target_semantic_cue_observed_but_no_target_edge_shape"
    elif subject_mentions or object_mentions:
        classification, strength, reason = "no_relation_extracted", "confirmed", "endpoint_relation_candidates_present_without_target_relation"
    else:
        classification, strength, reason = "no_relation_extracted", "confirmed", "observed_extraction_stage_has_no_related_relation_edge"

    return {
        "classification": classification,
        "evidence_strength": strength,
        "reason": reason,
        "exact_edge_count": len(exact),
        "endpoint_alias_candidate_count": len(endpoint_alias),
        "subject_relation_candidate_count": len(subject_mentions),
        "object_relation_candidate_count": len(object_mentions),
        "semantic_candidate_count": len(semantic_candidates),
        "direction_reversed_candidate_count": len(reversed_target),
        "entity_inventory_observed": False,
        "chunk_index_observed": chunk_index_observed,
        "entity_not_extracted": "insufficient_evidence",
        "cross_chunk_separation": "insufficient_evidence" if not chunk_index_observed else "not_observed",
        "candidate_edges": [edge for edge in (canonical_changed or endpoint_alias or reversed_target or semantic_candidates)[:4]],
    }


def classify_retrieval_gap(target: dict[str, str], stages: dict[str, Any]) -> dict[str, Any]:
    """Differentiate an exact retrieval loss from a retrieved alias relation."""
    persisted = [edge for edge in stages.get("persisted", []) if isinstance(edge, dict)]
    raw = [edge for edge in stages.get("retrieved_raw", []) if isinstance(edge, dict)]
    ranked = [edge for edge in stages.get("ranked", []) if isinstance(edge, dict)]
    prompt = [edge for edge in stages.get("entered_prompt", []) if isinstance(edge, dict)]
    query_trace = stages.get("query_trace") if isinstance(stages.get("query_trace"), dict) else None
    exact_persisted = [edge for edge in persisted if _exact(target, edge)]
    alias_persisted = [edge for edge in persisted if _aliased_endpoints(target, edge) and edge["predicate"] == target["predicate"]]
    exact_raw = [edge for edge in raw if _exact(target, edge)]
    alias_raw = [edge for edge in raw if _aliased_endpoints(target, edge) and edge["predicate"] == target["predicate"]]
    exact_ranked = [edge for edge in ranked if _exact(target, edge)]
    alias_ranked = [edge for edge in ranked if _aliased_endpoints(target, edge) and edge["predicate"] == target["predicate"]]
    exact_prompt = [edge for edge in prompt if _exact(target, edge)]
    alias_prompt = [edge for edge in prompt if _aliased_endpoints(target, edge) and edge["predicate"] == target["predicate"]]
    query_entities = query_trace.get("entities") if query_trace else None
    safe_query_entities = [str(item).strip() for item in query_entities] if isinstance(query_entities, list) else []
    entity_trace_observed = isinstance(query_entities, list)
    seeds_match_target = any(
        endpoint_alias_match(value, target["subject"]) or endpoint_alias_match(value, target["object"])
        for value in safe_query_entities
    )
    traversal_observed = bool(query_trace and query_trace.get("traversal_observed") is True)
    predicate_filter_observed = bool(query_trace and "predicate_filter" in query_trace)
    if exact_raw and not exact_ranked and "ranked" in stages:
        classification, strength, reason = "persisted_but_ranking_missing", "confirmed", "exact_edge_retrieved_but_not_ranked"
    elif exact_raw:
        classification, strength, reason = "insufficient_evidence", "confirmed", "exact_edge_retrieved"
    elif alias_raw:
        if not alias_ranked and "ranked" in stages:
            classification, strength, reason = "endpoint_alias_mismatch", "confirmed", "alias_equivalent_edge_retrieved_but_exact_identity_missing_before_ranking"
        else:
            classification, strength, reason = "endpoint_alias_mismatch", "confirmed", "alias_equivalent_edge_retrieved_but_exact_identity_missing"
    elif not exact_persisted and not alias_persisted:
        classification, strength, reason = "insufficient_evidence", "insufficient_evidence", "persisted_target_not_observed"
    elif entity_trace_observed and not seeds_match_target:
        classification, strength, reason = "persisted_but_query_entity_missing", "confirmed", "traced_query_entities_do_not_match_target_endpoints"
    elif traversal_observed:
        classification, strength, reason = "persisted_but_traversal_missing", "confirmed", "traced_traversal_completed_without_target_edge"
    elif "retrieved_raw" not in stages:
        classification, strength, reason = "insufficient_evidence", "insufficient_evidence", "retrieval_snapshot_missing"
    else:
        classification, strength, reason = "insufficient_evidence", "insufficient_evidence", "query_entities_hops_and_predicate_filters_not_traced"
    return {
        "classification": classification,
        "evidence_strength": strength,
        "reason": reason,
        "persisted_exact_count": len(exact_persisted),
        "persisted_alias_count": len(alias_persisted),
        "retrieved_raw_exact_count": len(exact_raw),
        "retrieved_raw_alias_count": len(alias_raw),
        "entered_prompt_exact_count": len(exact_prompt),
        "entered_prompt_alias_count": len(alias_prompt),
        "query_entities_observed": entity_trace_observed,
        "query_hops_observed": bool(query_trace and "max_hops" in query_trace),
        "predicate_filter_observed": predicate_filter_observed,
        "candidate_edges": [edge for edge in (alias_raw or alias_persisted or exact_persisted)[:4]],
    }


def _targets(selected: Any) -> tuple[list[tuple[Any, dict[str, str]]], tuple[Any, dict[str, str]]]:
    wanted = {
        ("Q01", "HAS_ROLE"), ("Q24", "MONITORS"), ("Q24", "DEPENDS_ON"),
        ("Q41", "NOT_DEPENDS_ON"), ("Q41", "DEPENDS_ON"),
    }
    extraction: list[tuple[Any, dict[str, str]]] = []
    retrieval: tuple[Any, dict[str, str]] | None = None
    for case in selected.cases:
        for path in case.expected_relation_path:
            if len(path) < 3:
                continue
            target = {
                "subject": str(path[0]).strip(),
                "predicate": canonicalize_relation(str(path[1])).predicate,
                "object": str(path[2]).strip(),
                "direction": str(path[3]).strip() if len(path) > 3 else "forward",
            }
            if (case.question_id, target["predicate"]) in wanted:
                extraction.append((case, target))
            if case.question_id == "Q41" and target["predicate"] == "PROVIDES_INDEX":
                retrieval = (case, target)
    if len(extraction) != 5 or retrieval is None:
        raise GraphTraceGapAnalysisError("target_edges_not_found_in_fixture")
    return extraction, retrieval


def _events_for_sources(events: list[dict[str, Any]], sources: Iterable[str], stage: str) -> tuple[list[dict[str, Any]], bool]:
    safe_sources = {_safe_source(item) for item in sources}
    subset = [item for item in events if _safe_source(item.get("source")) in safe_sources]
    marker = any(item.get("stage") == stage and item.get("event_kind") == "stage_observed" for item in subset)
    edges = [_edge(item) for item in subset if item.get("stage") == stage and item.get("event_kind") == "edge"]
    return edges, marker


def _safe_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ("question_id", "subject", "expected_predicate", "object", "direction", "classification", "evidence_strength", "reason", "exact_edge_count", "endpoint_alias_candidate_count", "semantic_candidate_count", "direction_reversed_candidate_count", "entity_inventory_observed", "chunk_index_observed")
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    os.replace(temporary, path)


def _candidates(matrix: list[dict[str, Any]], retrieval: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = [{
        "problem_layer": "extract", "evidence_strength": "confirmed",
        "minimal_candidate_change": "Record a bounded entity identity and chunk index/ordinal in the evaluation-only journal; do not record chunk text.",
        "expected_affected_relations": [row["expected_predicate"] for row in matrix],
        "required_fake_tests": ["empty_entity_manifest_is_insufficient", "cross_chunk_marker_is_safe"],
        "requires_new_real_validation": True,
    }]
    if any(row["classification"] == "canonicalization_gap" for row in matrix):
        candidates.append({
            "problem_layer": "semantics", "evidence_strength": "confirmed",
            "minimal_candidate_change": "Add reviewed predicate mappings only for observed raw forms, with a direction-preserving fake contract.",
            "expected_affected_relations": [row["expected_predicate"] for row in matrix if row["classification"] == "canonicalization_gap"],
            "required_fake_tests": ["raw_monitor_to_monitors", "raw_not_depends_to_negative_predicate", "no_dependency_inference"],
            "requires_new_real_validation": True,
        })
    if retrieval["classification"] == "endpoint_alias_mismatch":
        candidates.append({
            "problem_layer": "retrieval", "evidence_strength": "confirmed",
            "minimal_candidate_change": "Use a validated alias-normalisation key for graph query seeds and preserve the original evidence identity in output.",
            "expected_affected_relations": ["PROVIDES_INDEX"],
            "required_fake_tests": ["northstar_short_name_retrieves_full_name", "scope_remains_fail_closed", "no_cross_document_alias_leak"],
            "requires_new_real_validation": True,
        })
    else:
        candidates.append({
            "problem_layer": "retrieval", "evidence_strength": "insufficient_evidence",
            "minimal_candidate_change": "Trace query entities, hop limit and predicate filters before changing traversal or ranking.",
            "expected_affected_relations": ["PROVIDES_INDEX"],
            "required_fake_tests": ["query_plan_trace_has_no_prompt_or_text", "missing_query_trace_is_insufficient"],
            "requires_new_real_validation": True,
        })
    return candidates


def analyze_trace_gaps(trace_dir: Path, output_dir: Path, benchmark_dir: Path) -> dict[str, Any]:
    """Create only derived reports after validating five immutable input hashes."""
    trace_dir, output_dir, benchmark_dir = Path(trace_dir), Path(output_dir), Path(benchmark_dir)
    before = immutable_input_hashes(trace_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise GraphTraceGapAnalysisError("output_directory_already_exists")
    state, trace_payload = _read_json(trace_dir / "ingestion-state.json"), _read_json(trace_dir / "graph-trace.json")
    run_id = str(state.get("run_id") or "")
    if not _SAFE_RUN_ID.fullmatch(run_id) or run_id != trace_dir.name:
        raise GraphTraceGapAnalysisError("invalid_run_id")
    fixture_state = state.get("fixture")
    if not isinstance(fixture_state, dict):
        raise GraphTraceGapAnalysisError("missing_fixture_state")
    fixture = load_evaluation_fixture(benchmark_dir)
    selected = select_evaluation_subset(fixture, document_ids=tuple(map(str, fixture_state.get("document_ids", []))), question_ids=tuple(map(str, fixture_state.get("question_ids", []))))
    events = _read_journal(trace_dir / "graph-ingestion-trace.jsonl")
    traces = {str(item.get("question_id")): item for item in trace_payload.get("traces", []) if isinstance(item, dict)}
    if set(traces) != {case.question_id for case in selected.cases}:
        raise GraphTraceGapAnalysisError("trace_question_coverage_mismatch")
    extraction_targets, retrieval_target = _targets(selected)
    matrix: list[dict[str, Any]] = []
    for case, target in extraction_targets:
        extracted, marker = _events_for_sources(events, case.expected_sources, "extracted")
        diagnosis = classify_extraction_gap(target, extracted, stage_observed=marker)
        matrix.append({
            "question_id": case.question_id, "category": case.category,
            "subject": target["subject"], "expected_predicate": target["predicate"], "object": target["object"], "direction": target["direction"],
            "expected_sources": [_safe_source(item) for item in case.expected_sources], **diagnosis,
        })
    retrieval_case, target = retrieval_target
    trace = traces.get(retrieval_case.question_id)
    stages = {name: [_edge(edge) for edge in values if isinstance(edge, dict)] for name, values in trace.get("stages", {}).items() if isinstance(values, list)}
    # Current real traces do not have this optional safe query record.  Keeping
    # it optional makes missing fields fail closed while allowing fake fixtures
    # to exercise the bounded future classifications below.
    if isinstance(trace.get("query_trace"), dict):
        stages["query_trace"] = trace["query_trace"]
    retrieval = {
        "question_id": retrieval_case.question_id, "subject": target["subject"], "expected_predicate": target["predicate"], "object": target["object"], "direction": target["direction"],
        "expected_sources": [_safe_source(item) for item in retrieval_case.expected_sources], **classify_retrieval_gap(target, stages),
    }
    after = immutable_input_hashes(trace_dir)
    if before != after:
        raise GraphTraceGapAnalysisError("immutable_input_changed_during_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {"run_id": run_id, "sha256_before": before, "sha256_after": after, "unchanged": True}
    remediation = _candidates(matrix, retrieval)
    _atomic_json(output_dir / "immutable-input-hashes.json", hashes)
    _atomic_json(output_dir / "extraction-gap-matrix.json", {"run_id": run_id, "schema_version": ANALYSIS_SCHEMA_VERSION, "rows": matrix})
    _safe_csv(output_dir / "extraction-gap-matrix.csv", matrix)
    _atomic_json(output_dir / "retrieval-gap-analysis.json", {"run_id": run_id, "schema_version": ANALYSIS_SCHEMA_VERSION, "analysis": retrieval})
    _atomic_json(output_dir / "remediation-candidates.json", {"run_id": run_id, "candidates": remediation})
    counts = Counter(row["classification"] for row in matrix)
    lines = ["# S4.8a Relation-Gap Analysis", "", f"Run: `{run_id}`", "", "## Extraction classifications", "", "| Edge | Classification | Evidence |", "|---|---|---|"]
    for row in matrix:
        lines.append(f"| {row['question_id']} {row['subject']} --{row['expected_predicate']}--> {row['object']} | {row['classification']} | {row['evidence_strength']} |")
    lines.extend(["", "## Retrieval classification", "", f"Q41 `PROVIDES_INDEX`: **{retrieval['classification']}** ({retrieval['evidence_strength']}).", "", "The journal contains relation edges but not a complete entity inventory, chunk ordinal, query rewrite entities, actual hop limit, or predicate filter. Those unrecorded fields remain `insufficient_evidence`; this report does not infer them from an exact-string miss."])
    _atomic_text(output_dir / "summary.md", "\n".join(lines) + "\n")
    return {"run_id": run_id, "matrix": matrix, "retrieval": retrieval, "classification_counts": dict(sorted(counts.items())), "output_dir": str(output_dir)}
