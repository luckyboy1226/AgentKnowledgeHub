"""Fail-closed, offline-only first-loss analysis for persisted GraphRAG traces.

This module reads JSON artifacts only.  It deliberately imports no provider,
database client, API application, or evaluation runner, so it can never turn a
post-run diagnosis into another model or storage operation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from services.graph_trace_diagnosis import diagnose_first_loss
from services.rag_evaluation import load_evaluation_fixture, select_evaluation_subset


ANALYSIS_SCHEMA_VERSION = "s4.7a-first-loss-v1"
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,120}$")
_REQUIRED_INPUTS = (
    "results.json", "graph-trace.json", "safe-run-metadata.json", "ingestion-state.json",
)
_SAFE_SOURCE = re.compile(r"^[^\\/:*?\"<>|\x00-\x1f]+$")


class GraphTraceAnalysisError(ValueError):
    """Raised before writing output when immutable evidence is incomplete."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GraphTraceAnalysisError(f"unable_to_read_{path.name}") from exc
    if not isinstance(value, dict):
        raise GraphTraceAnalysisError(f"invalid_json_root_{path.name}")
    return value


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def immutable_input_hashes(trace_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for filename in _REQUIRED_INPUTS:
        path = trace_dir / filename
        if not path.is_file():
            raise GraphTraceAnalysisError(f"missing_input_{filename}")
        hashes[filename] = _file_hash(path)
    return hashes


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
    if not source or not _SAFE_SOURCE.fullmatch(source):
        raise GraphTraceAnalysisError("unsafe_source_identifier")
    return source


def _validate_inputs(trace_dir: Path, benchmark_dir: Path) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], Any]:
    if not _SAFE_RUN_ID.fullmatch(trace_dir.name):
        raise GraphTraceAnalysisError("unsafe_trace_run_id")
    results = _read_json(trace_dir / "results.json")
    traces = _read_json(trace_dir / "graph-trace.json")
    metadata = _read_json(trace_dir / "safe-run-metadata.json")
    state = _read_json(trace_dir / "ingestion-state.json")
    run_id = str(state.get("run_id") or "")
    if not _SAFE_RUN_ID.fullmatch(run_id) or any(str(payload.get("run_id") or "") != run_id for payload in (results, metadata)):
        raise GraphTraceAnalysisError("run_id_mismatch")
    if run_id != trace_dir.name:
        raise GraphTraceAnalysisError("trace_directory_run_id_mismatch")
    if metadata.get("selection_fingerprint") != state.get("selection_fingerprint"):
        raise GraphTraceAnalysisError("selection_fingerprint_mismatch")
    fixture_info = state.get("fixture")
    if not isinstance(fixture_info, dict):
        raise GraphTraceAnalysisError("missing_fixture_state")
    document_ids, question_ids = fixture_info.get("document_ids"), fixture_info.get("question_ids")
    if not all(isinstance(value, list) and value for value in (document_ids, question_ids)):
        raise GraphTraceAnalysisError("invalid_fixture_selection")
    fixture = load_evaluation_fixture(benchmark_dir)
    selected = select_evaluation_subset(fixture, document_ids=tuple(map(str, document_ids)), question_ids=tuple(map(str, question_ids)))
    if tuple(item.document_id for item in selected.documents) != tuple(map(str, document_ids)):
        raise GraphTraceAnalysisError("fixture_document_selection_mismatch")
    if tuple(item.question_id for item in selected.cases) != tuple(map(str, question_ids)):
        raise GraphTraceAnalysisError("fixture_question_selection_mismatch")
    if not isinstance(traces.get("traces"), list):
        raise GraphTraceAnalysisError("invalid_trace_schema")
    seen_questions: set[str] = set()
    for trace in traces["traces"]:
        if not isinstance(trace, dict) or trace.get("mode") != "graph_rag" or trace.get("run_id") != run_id:
            raise GraphTraceAnalysisError("trace_pair_schema_mismatch")
        question_id = str(trace.get("question_id") or "")
        if question_id not in {case.question_id for case in selected.cases} or question_id in seen_questions:
            raise GraphTraceAnalysisError("trace_question_pair_duplicate_or_unknown")
        if not isinstance(trace.get("stages"), dict):
            raise GraphTraceAnalysisError("trace_stages_missing")
        seen_questions.add(question_id)
    if seen_questions != {case.question_id for case in selected.cases}:
        raise GraphTraceAnalysisError("trace_question_coverage_mismatch")
    # Selected results must have exactly one vector and one graph result per Q.
    rows = results.get("results")
    if not isinstance(rows, list):
        raise GraphTraceAnalysisError("invalid_results_schema")
    pairs = {(str(row.get("question_id") or ""), str(row.get("mode") or "")) for row in rows if isinstance(row, dict)}
    expected_pairs = {(case.question_id, mode) for case in selected.cases for mode in ("vector_only", "graph_rag")}
    if pairs != expected_pairs:
        raise GraphTraceAnalysisError("result_pair_coverage_mismatch")
    return run_id, results, traces, metadata, state, selected


def _report_row(case: Any, edge_index: int, diagnosis: dict[str, Any]) -> dict[str, Any]:
    expected = diagnosis["expected"]
    stage_counts = diagnosis["stage_counts"]
    return {
        "question_id": case.question_id,
        "edge_index": edge_index,
        "category": case.category,
        "subject": expected["subject"],
        "expected_predicate": expected["predicate"],
        "object": expected["object"],
        "direction": expected["direction"],
        "expected_sources": list(case.expected_sources),
        "expected_edge_fingerprint": diagnosis["expected_edge_fingerprint"],
        "stage_counts": stage_counts,
        "stage_presence": {stage: bool(values["snapshot_present"]) for stage, values in stage_counts.items()},
        "matched_stages": diagnosis["matched_stages"],
        "endpoint_match": diagnosis["endpoint_match"],
        "predicate_match": diagnosis["canonical_predicate_match"],
        "direction_match": diagnosis["direction_match"],
        "provenance_completeness": any(values["provenance_complete_matches"] for values in stage_counts.values()),
        "first_loss_stage": diagnosis["first_loss_stage"],
        "loss_reason": diagnosis["loss_reason"],
        "confidence": diagnosis["confidence"],
        "evidence_boundary": diagnosis["evidence_boundary"],
        "failure_category": diagnosis["failure_category"],
        "entered_prompt": diagnosis["first_loss_stage"] == "entered_prompt",
    }


def analyze_trace_run(trace_dir: Path, output_dir: Path, benchmark_dir: Path, *, strict: bool = False) -> dict[str, Any]:
    """Produce a derived, immutable-safe edge report for one real trace run."""
    trace_dir, output_dir, benchmark_dir = Path(trace_dir), Path(output_dir), Path(benchmark_dir)
    before_hashes = immutable_input_hashes(trace_dir)
    run_id, _results, trace_payload, metadata, state, fixture = _validate_inputs(trace_dir, benchmark_dir)
    by_question = {str(item["question_id"]): item for item in trace_payload["traces"]}
    edges: list[dict[str, Any]] = []
    for case in fixture.cases:
        diagnosis = diagnose_first_loss(case.expected_relation_path, by_question[case.question_id])
        for index, edge_diagnosis in enumerate(diagnosis["edge_diagnoses"], start=1):
            edges.append(_report_row(case, index, edge_diagnosis))
    if strict and not edges:
        raise GraphTraceAnalysisError("no_expected_relation_edges")
    after_hashes = immutable_input_hashes(trace_dir)
    if before_hashes != after_hashes:
        raise GraphTraceAnalysisError("immutable_input_changed_during_analysis")
    counts = Counter(row["first_loss_stage"] for row in edges)
    stage_counts = {
        "run_id": run_id,
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "expected_edge_count": len(edges),
        "first_loss_stage_counts": dict(sorted(counts.items())),
        "trace_stage_snapshot_coverage": {
            question_id: sorted(str(stage) for stage in trace.get("stages", {}))
            for question_id, trace in sorted(by_question.items())
        },
    }
    critical = [
        row for row in edges
        if (row["question_id"], row["expected_predicate"]) in {
            ("Q01", "HAS_ROLE"), ("Q24", "MONITORS"), ("Q41", "NOT_DEPENDS_ON"),
        }
    ]
    report = {
        "run_id": run_id,
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "selection_fingerprint": state["selection_fingerprint"],
        "fixture_document_ids": [item.document_id for item in fixture.documents],
        "fixture_question_ids": [item.question_id for item in fixture.cases],
        "trace_count": len(by_question),
        "expected_edges": edges,
        "first_loss_stage_counts": dict(sorted(counts.items())),
        "evidence_boundary": "pre-retrieval ingestion snapshots were not present in this persisted run; missing snapshots are insufficient_evidence, not an extraction or persistence attribution",
        "immutable_input_hashes": before_hashes,
    }
    if output_dir.exists() and any(output_dir.iterdir()):
        raise GraphTraceAnalysisError("output_directory_already_exists")
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "immutable-input-hashes.json", {"run_id": run_id, "sha256": before_hashes})
    _atomic_json(output_dir / "first-loss.json", report)
    _atomic_json(output_dir / "stage-counts.json", stage_counts)
    _atomic_json(output_dir / "critical-edges.json", {"run_id": run_id, "edges": critical})
    headers = (
        "question_id", "edge_index", "category", "subject", "expected_predicate", "object", "direction",
        "first_loss_stage", "loss_reason", "confidence", "failure_category", "endpoint_match",
        "predicate_match", "direction_match", "provenance_completeness", "entered_prompt",
    )
    csv_path = output_dir / "first-loss.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows({key: row[key] for key in headers} for row in edges)
    os.replace(temporary, csv_path)
    markdown = [
        "# S4.7a First-Loss Analysis", "", f"Run: `{run_id}`", "",
        "This is an offline derived report.  The persisted real trace begins at `retrieved_raw`; it contains no extraction, normalization, or persistence snapshots. Therefore no pre-retrieval absence is attributed to extraction or persistence.",
        "", "## Counts", "", "| First-loss stage | Edges |", "|---|---:|",
    ]
    markdown.extend(f"| {stage} | {count} |" for stage, count in sorted(counts.items()))
    markdown.extend(["", "## Critical edges", "", "| Question | Edge | Conclusion | Boundary |", "|---|---|---|---|"])
    markdown.extend(
        f"| {row['question_id']} | {row['subject']} --{row['expected_predicate']}--> {row['object']} | {row['first_loss_stage']} | {row['evidence_boundary']} |"
        for row in critical
    )
    _atomic_text(output_dir / "summary.md", "\n".join(markdown) + "\n")
    return report
