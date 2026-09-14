"""Offline-only, immutable analysis for persisted RAG evaluation runs.

This module deliberately imports neither providers nor storage clients.  It
classifies only evidence that was persisted in a result file, so it never
claims to identify an extraction, database-write, or pre-prompt retrieval
failure when those stage snapshots are unavailable.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from services.relation_semantics import RELATION_SEMANTICS_VERSION, canonical_predicate


DIAGNOSIS_VERSION = "deterministic-diagnosis-v1"
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,120}$")
_SAFE_SOURCE = re.compile(r"^[^\\/:*?\"<>|\x00-\x1f]+$")
_MODES = frozenset({"vector_only", "graph_rag"})


class DiagnosisError(ValueError):
    """Raised before output when an immutable input is incomplete or unsafe."""


@dataclass(frozen=True)
class DiagnosisCase:
    question_id: str
    category: str
    expected_sources: tuple[str, ...]
    expected_edges: tuple[tuple[str, str, str, str], ...]
    requires_abstention: bool


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosisError(f"unable to read {path.name} ({type(exc).__name__})") from exc


def _safe_source(value: object) -> str:
    text = str(value or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not text or not _SAFE_SOURCE.fullmatch(text):
        raise DiagnosisError("result contains an unsafe source identifier")
    return text


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def immutable_hashes(results_path: Path, benchmark_root: Path) -> dict[str, str]:
    names = (
        ("results.json", results_path),
        ("results.csv", results_path.with_name("results.csv")),
        ("summary.md", results_path.with_name("summary.md")),
        ("safe-run-metadata.json", results_path.with_name("safe-run-metadata.json")),
        ("benchmark_manifest.json", benchmark_root / "benchmark_manifest.json"),
        ("benchmark_documents.json", benchmark_root / "benchmark_documents.json"),
        ("benchmark_questions.json", benchmark_root / "benchmark_questions.json"),
    )
    hashes: dict[str, str] = {}
    for label, path in names:
        if not path.is_file():
            raise DiagnosisError(f"immutable input missing: {label}")
        hashes[label] = _hash_file(path)
    return hashes


def _edge(raw: object) -> tuple[str, str, str, str]:
    if not isinstance(raw, (list, tuple)) or len(raw) not in (3, 4):
        raise DiagnosisError("fixture relation path has an invalid edge")
    subject, predicate, object_ = (" ".join(str(value).split()) for value in raw[:3])
    direction = " ".join(str(raw[3] if len(raw) == 4 else "forward").split())
    if not all((subject, predicate, object_)) or direction not in {"forward", "reverse"}:
        raise DiagnosisError("fixture relation edge is incomplete")
    return subject, canonical_predicate(predicate), object_, direction


def load_cases(benchmark_root: Path) -> tuple[dict[str, DiagnosisCase], dict[str, Any]]:
    manifest = _read_json(benchmark_root / "benchmark_manifest.json")
    documents = _read_json(benchmark_root / "benchmark_documents.json")
    questions = _read_json(benchmark_root / "benchmark_questions.json")
    if not all(isinstance(value, dict) for value in (manifest, documents, questions)):
        raise DiagnosisError("fixture roots must be JSON objects")
    document_rows = documents.get("documents")
    question_rows = questions.get("questions")
    if not isinstance(document_rows, list) or not isinstance(question_rows, list):
        raise DiagnosisError("fixture documents/questions must be lists")
    if manifest.get("documents") != len(document_rows) or manifest.get("questions") != len(question_rows):
        raise DiagnosisError("fixture manifest count mismatch")
    filenames = {_safe_source(row.get("filename")) for row in document_rows if isinstance(row, dict)}
    if len(filenames) != len(document_rows):
        raise DiagnosisError("fixture document filenames are invalid or duplicate")
    cases: dict[str, DiagnosisCase] = {}
    categories: Counter[str] = Counter()
    for row in question_rows:
        if not isinstance(row, dict):
            raise DiagnosisError("fixture question entry must be an object")
        question_id = str(row.get("question_id", "")).strip()
        category = str(row.get("category", "")).strip()
        paths = row.get("expected_relation_path")
        sources = row.get("expected_sources")
        if not question_id or question_id in cases or not category or not isinstance(paths, list) or not isinstance(sources, list):
            raise DiagnosisError("fixture question identity or relation fields are invalid")
        normalized_sources = tuple(_safe_source(value) for value in sources)
        if any(source not in filenames for source in normalized_sources):
            raise DiagnosisError("fixture question references an unknown source")
        cases[question_id] = DiagnosisCase(
            question_id=question_id,
            category=category,
            expected_sources=normalized_sources,
            expected_edges=tuple(_edge(value) for value in paths),
            requires_abstention=bool(row.get("requires_abstention", False)),
        )
        categories[category] += 1
    declared = manifest.get("question_categories")
    if not isinstance(declared, dict) or {str(key): int(value) for key, value in declared.items()} != dict(categories):
        raise DiagnosisError("fixture category distribution mismatch")
    return cases, manifest


def _observed_edge(raw: object) -> tuple[str, str, str, str] | None:
    if not isinstance(raw, dict):
        return None
    subject = " ".join(str(raw.get("subject", "")).split())
    predicate = " ".join(str(raw.get("predicate", "")).split())
    object_ = " ".join(str(raw.get("object", "")).split())
    direction = " ".join(str(raw.get("direction", "")).split())
    if not all((subject, predicate, object_)) or direction not in {"forward", "reverse"}:
        return None
    return subject, canonical_predicate(predicate), object_, direction


def _answer_hash(value: object) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _score_value(row: dict[str, Any], name: str, default: Any = None) -> Any:
    value = row.get("deterministic_score_v2")
    return value.get(name, default) if isinstance(value, dict) else default


def _category_bucket(category: str) -> str:
    value = category.casefold()
    if value in {"unanswerable", "abstention"}:
        return "abstention"
    if value.startswith("distractor"):
        return "distractor"
    if value.startswith("constraint"):
        return "constraint"
    return value


def _unique_edges(edges: Iterable[tuple[str, str, str, str]]) -> set[tuple[str, str, str, str]]:
    return set(edges)


def diagnose_row(row: dict[str, Any], case: DiagnosisCase, diagnosis_version: str) -> dict[str, Any]:
    mode = row.get("mode")
    if mode not in _MODES:
        raise DiagnosisError("result row has an invalid mode")
    sources = row.get("sources", [])
    if not isinstance(sources, list):
        raise DiagnosisError("result sources must be a list")
    source_names = sorted({_safe_source(source.get("source")) for source in sources if isinstance(source, dict)})
    raw_edges = row.get("graph_evidence_edges", [])
    if not isinstance(raw_edges, list):
        raise DiagnosisError("graph evidence must be a list")
    observed = _unique_edges(edge for edge in (_observed_edge(value) for value in raw_edges) if edge is not None)
    expected = _unique_edges(case.expected_edges)
    exact = expected & observed
    predicate_matches = {
        edge for edge in expected
        if any(candidate[1] == edge[1] for candidate in observed)
    }
    observed_predicate_matches = {
        candidate for candidate in observed
        if any(candidate[1] == edge[1] for edge in expected)
    }
    endpoint_matches = {
        edge for edge in expected
        if any(candidate[0] == edge[0] and candidate[2] == edge[2] for candidate in observed)
    }
    predicate_endpoint_matches = {
        edge for edge in expected
        if any(candidate[:3] == edge[:3] for candidate in observed)
    }
    direction_matches = {
        edge for edge in expected
        if any(candidate == edge for candidate in observed)
    }
    relevant = {
        candidate for candidate in observed
        if any(candidate[0] == edge[0] or candidate[2] == edge[2] for edge in expected)
    }
    graph_calls = row.get("model_call_counts", {}).get("graph_calls") if isinstance(row.get("model_call_counts"), dict) else None
    graph_called = bool(isinstance(graph_calls, int) and graph_calls > 0)
    graph_count = int(row.get("graph_context_count") or 0)
    score_v1 = row.get("deterministic_score") if isinstance(row.get("deterministic_score"), dict) else {}
    answer_semantic = _score_value(row, "answer_semantic_score")
    source_coverage = _score_value(row, "source_coverage_score", 0.0)
    abstention_score = _score_value(row, "abstention_score")
    forbidden_v2 = _score_value(row, "forbidden_keyword_hits_v2", [])
    forbidden = bool(row.get("forbidden_fact_violation")) or bool(forbidden_v2)
    failure: set[str] = set()
    if mode == "vector_only":
        failure.add("scorer_not_applicable")
    elif not expected:
        failure.update({"scorer_not_applicable", "no_expected_relation"})
    else:
        if not graph_called:
            failure.add("graph_not_called")
        if graph_called and not observed:
            failure.add("no_graph_evidence")
        if expected - observed:
            failure.add("expected_edge_missing_from_prompt")
            # The result contains no extraction/write/full retrieval snapshots.
            failure.add("insufficient_evidence")
        if endpoint_matches - predicate_endpoint_matches:
            failure.add("predicate_mismatch")
        if predicate_endpoint_matches - direction_matches:
            failure.add("direction_reversed")
        if predicate_matches - endpoint_matches:
            failure.add("subject_object_mismatch")
        if len(expected) > 1 and exact != expected:
            failure.add("incomplete_multi_hop_path")
        if observed - relevant:
            failure.add("irrelevant_graph_evidence")
        if observed and not relevant:
            failure.add("graph_top_k_dilution")
        if exact == expected and answer_semantic != 1.0:
            failure.add("correct_evidence_but_wrong_answer")
    if forbidden:
        failure.add("forbidden_fact_generated")
    if case.requires_abstention and abstention_score != 1.0:
        failure.add("abstention_error")
    if source_coverage != 1.0 and not case.requires_abstention:
        failure.add("source_coverage_error")
    if answer_semantic == 1.0 and score_v1.get("question_correct") is False:
        failure.add("scorer_false_negative")
    applicable = mode == "graph_rag" and bool(expected)
    return {
        "run_id": str(row.get("run_id", "")),
        "question_id": case.question_id,
        "category": _category_bucket(case.category),
        "mode": mode,
        "success": bool(row.get("success")),
        "scorer_version": _score_value(row, "scorer_version"),
        "diagnosis_version": diagnosis_version,
        "answer_semantic_correct": answer_semantic == 1.0,
        "source_coverage_score": source_coverage,
        "abstention_correct": abstention_score == 1.0 if abstention_score is not None else None,
        "forbidden_fact_violation": forbidden,
        "forbidden_pattern_ids": [f"forbidden_pattern_{index + 1}" for index, _ in enumerate(forbidden_v2 or [])],
        "graph_called": graph_called,
        "graph_context_count": graph_count,
        "graph_evidence_entered_top_k": bool(observed),
        "expected_edge_count": len(expected) if applicable else None,
        "observed_edge_count": len(observed) if mode == "graph_rag" else None,
        "exact_edge_match_count": len(exact) if applicable else None,
        "canonical_predicate_match_count": len(predicate_matches) if applicable else None,
        "observed_predicate_match_count": len(observed_predicate_matches) if applicable else None,
        "predicate_endpoint_match_count": len(predicate_endpoint_matches) if applicable else None,
        "direction_match_count": len(direction_matches) if applicable else None,
        "endpoint_match_count": len(endpoint_matches) if applicable else None,
        "relevant_graph_edge_count": len(relevant) if mode == "graph_rag" else None,
        "irrelevant_graph_edge_count": len(observed - relevant) if mode == "graph_rag" else None,
        "relation_edge_recall": round(len(exact) / len(expected), 3) if applicable else None,
        "relation_predicate_precision": round(len(observed_predicate_matches) / len(observed), 3) if applicable and observed else None,
        "relation_direction_accuracy": round(len(direction_matches) / len(predicate_endpoint_matches), 3) if applicable and predicate_endpoint_matches else None,
        "complete_path": exact == expected if applicable and len(expected) > 1 else None,
        "failure_categories": sorted(failure),
        "evidence_sufficiency": "prompt_only" if expected - observed else "sufficient_for_prompt_generation_check",
        "safe_source_names": source_names,
        "answer_hash": _answer_hash(row.get("answer")),
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["run_id", "question_id", "mode"]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (list, dict)) else value for key, value in row.items()})
    os.replace(temporary, path)


def _mean(values: list[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return round(sum(clean) / len(clean), 3) if clean else None


def _summary(rows: list[dict[str, Any]], original_summary: object) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    matrix: list[dict[str, Any]] = []
    for mode in sorted(_MODES):
        for category in ("single_hop", "multi_hop", "constraint", "distractor", "abstention"):
            bucket = [row for row in rows if row["mode"] == mode and row["category"] == category]
            failures = Counter(item for row in bucket for item in row["failure_categories"])
            matrix.append({"mode": mode, "category": category, "count": len(bucket), "failures": dict(sorted(failures.items()))})
    graph_rows = [row for row in rows if row["mode"] == "graph_rag" and row["expected_edge_count"] is not None]
    denominator = sum(int(row["expected_edge_count"] or 0) for row in graph_rows)
    exact = sum(int(row["exact_edge_match_count"] or 0) for row in graph_rows)
    observed_predicate = sum(int(row["observed_predicate_match_count"] or 0) for row in graph_rows)
    observed_total = sum(int(row["observed_edge_count"] or 0) for row in graph_rows)
    direction_candidates = sum(int(row["predicate_endpoint_match_count"] or 0) for row in graph_rows)
    direction_matches = sum(int(row["direction_match_count"] or 0) for row in graph_rows)
    multi_paths = [row for row in graph_rows if row["expected_edge_count"] and row["expected_edge_count"] > 1]
    confusion = Counter()
    for row in graph_rows:
        if "predicate_mismatch" in row["failure_categories"]:
            confusion["expected_predicate_vs_observed_same_endpoint"] += 1
        if "direction_reversed" in row["failure_categories"]:
            confusion["same_predicate_endpoint_direction_reversed"] += 1
        if "subject_object_mismatch" in row["failure_categories"]:
            confusion["same_predicate_subject_object_mismatch"] += 1
    priority_specs = (
        ("P0", "scorer denominator / N-A", "scorer_not_applicable"),
        ("P1", "graph evidence relevance filtering", "irrelevant_graph_evidence"),
        ("P2", "predicate and relation-intent matching", "predicate_mismatch"),
        ("P3", "multi-hop path completeness", "incomplete_multi_hop_path"),
        ("P4", "answer evidence-faithfulness", "correct_evidence_but_wrong_answer"),
        ("P5", "conflict-aware abstention", "forbidden_fact_generated"),
        ("P6", "extraction/write observability", "insufficient_evidence"),
    )
    priorities = []
    for priority, layer, category in priority_specs:
        affected_rows = [row for row in rows if category in row["failure_categories"]]
        # Vector-only has no graph relation metric by design; it is not a
        # scoring defect.  P0 concerns only GraphRAG's no-path N/A rows.
        if priority == "P0":
            affected_rows = [row for row in affected_rows if row["mode"] == "graph_rag"]
        affected = [row["question_id"] for row in affected_rows]
        if affected:
            priorities.append({"priority": priority, "candidate": layer, "count": len(affected), "question_ids": sorted(set(affected)), "evidence_strength": "direct prompt evidence" if category != "insufficient_evidence" else "bounded: pre-prompt stage unavailable", "requires_real_model_recheck": category in {"correct_evidence_but_wrong_answer", "forbidden_fact_generated"}, "may_affect_public_qa": category not in {"scorer_not_applicable", "insufficient_evidence"}})
    summary = {
        "original_metrics": original_summary if isinstance(original_summary, dict) else {},
        "relationship_recalculation": {
            "original_relation_fidelity": (original_summary or {}).get("graph_rag", {}).get("relation_fidelity") if isinstance(original_summary, dict) else None,
            "applicable_graph_question_count": len(graph_rows),
            "edge_denominator": denominator,
            "exact_edge_numerator": exact,
            "applicable_relation_fidelity": round(exact / denominator, 3) if denominator else None,
            "edge_level_recall": round(exact / denominator, 3) if denominator else None,
            "predicate_precision": round(observed_predicate / observed_total, 3) if observed_total else None,
            "direction_accuracy": round(direction_matches / direction_candidates, 3) if direction_candidates else None,
            "complete_path_accuracy": round(sum(row["complete_path"] is True for row in multi_paths) / len(multi_paths), 3) if multi_paths else None,
            "complete_path_denominator": len(multi_paths),
            "note": "Vector-only relation metrics are N/A. Questions without expected_relation_path are excluded rather than scored as zero.",
        },
        "failure_category_counts": dict(sorted(Counter(item for row in rows for item in row["failure_categories"]).items())),
        "forbidden_fact_rows": [{"question_id": row["question_id"], "mode": row["mode"], "category": row["category"], "pattern_ids": row["forbidden_pattern_ids"], "graph_evidence_entered_top_k": row["graph_evidence_entered_top_k"], "conflict": "predicate_mismatch" in row["failure_categories"]} for row in rows if row["forbidden_fact_violation"]],
        "priorities": priorities,
        "limitations": "The persisted run contains final-prompt evidence only. Missing prompt evidence cannot be attributed to extraction, Neo4j persistence, or pre-prompt retrieval without stage snapshots.",
    }
    confusion_rows = [{"confusion": key, "count": value} for key, value in sorted(confusion.items())]
    return summary, matrix, confusion_rows


def _markdown(summary: dict[str, Any], matrix: list[dict[str, Any]]) -> str:
    relation = summary["relationship_recalculation"]
    lines = [
        "# Deterministic GraphRAG failure diagnosis",
        "",
        "This is a derived, offline-only analysis. It does not change original answers, scores, or inputs.",
        "",
        "## Relation metrics",
        "",
        f"- Original GraphRAG relation fidelity: {relation['original_relation_fidelity']}",
        f"- Applicable graph rows: {relation['applicable_graph_question_count']}; edge denominator: {relation['edge_denominator']}; exact matches: {relation['exact_edge_numerator']}",
        f"- Edge-level recall: {relation['edge_level_recall']}; predicate precision: {relation['predicate_precision']}; direction accuracy: {relation['direction_accuracy']}; complete-path accuracy: {relation['complete_path_accuracy']}",
        "",
        "## Failure matrix",
        "",
        "| Mode | Category | Rows | Failure categories |",
        "|---|---|---:|---|",
    ]
    for row in matrix:
        failures = ", ".join(f"{key}:{value}" for key, value in row["failures"].items()) or "none"
        lines.append(f"| {row['mode']} | {row['category']} | {row['count']} | {failures} |")
    lines.extend(["", "## Evidence boundary", "", summary["limitations"], "", "## Evidence-supported priorities", ""])
    for item in summary["priorities"]:
        lines.append(f"- **{item['priority']} {item['candidate']}** — {item['count']} rows; evidence: {item['evidence_strength']}; real-model recheck: {item['requires_real_model_recheck']}.")
    return "\n".join(lines) + "\n"


def run_diagnosis(results_path: str | Path, benchmark_root: str | Path, output_dir: str | Path, *, diagnosis_version: str = DIAGNOSIS_VERSION, strict: bool = False) -> dict[str, Any]:
    results_path = Path(results_path)
    benchmark_root = Path(benchmark_root)
    output_dir = Path(output_dir)
    if diagnosis_version != DIAGNOSIS_VERSION:
        raise DiagnosisError("unsupported diagnosis version")
    before = immutable_hashes(results_path, benchmark_root)
    cases, manifest = load_cases(benchmark_root)
    payload = _read_json(results_path)
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise DiagnosisError("results payload is invalid")
    run_id = str(payload.get("run_id", "")).strip()
    if not _SAFE_RUN_ID.fullmatch(run_id):
        raise DiagnosisError("results run_id is unsafe")
    if output_dir.exists() and any(output_dir.iterdir()):
        existing = _read_json(output_dir / "safe-diagnosis-metadata.json")
        if not isinstance(existing, dict) or existing.get("run_id") != run_id or existing.get("diagnosis_version") != diagnosis_version:
            raise DiagnosisError("output directory belongs to a different diagnosis")
    rows = payload["results"]
    expected_pairs = {(case_id, mode) for case_id in cases for mode in _MODES}
    seen: set[tuple[str, str]] = set()
    diagnosed: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, dict):
            raise DiagnosisError("result row is invalid")
        pair = (str(raw.get("question_id", "")), str(raw.get("mode", "")))
        if pair in seen:
            raise DiagnosisError("duplicate question/mode result")
        if pair not in expected_pairs:
            raise DiagnosisError("result question/mode does not match fixture")
        if str(raw.get("run_id", "")) != run_id or str(raw.get("category", "")) != cases[pair[0]].category:
            raise DiagnosisError("result run or category does not match fixture")
        seen.add(pair)
        diagnosed.append(diagnose_row(raw, cases[pair[0]], diagnosis_version))
    if strict and seen != expected_pairs:
        raise DiagnosisError("strict diagnosis requires exactly one result per fixture question and mode")
    summary, matrix, confusion = _summary(diagnosed, payload.get("summary"))
    after = immutable_hashes(results_path, benchmark_root)
    if before != after:
        raise DiagnosisError("immutable input changed during diagnosis")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"run_id": run_id, "diagnosis_version": diagnosis_version, "fixture_name": str(manifest.get("benchmark_name", "")), "rows": len(diagnosed), "network_accessed": False, "storage_initialized": False, "answer_text_persisted": False}
    output = {"run_id": run_id, "diagnosis_version": diagnosis_version, "results": diagnosed, "summary": summary, "notice": "Derived deterministic diagnosis; original model answers are represented only by SHA-256 hashes."}
    _atomic_json(output_dir / "immutable-input-hashes.json", before)
    _atomic_json(output_dir / "diagnosis.json", output)
    _atomic_json(output_dir / "failure-matrix.json", matrix)
    _atomic_csv(output_dir / "diagnosis.csv", diagnosed)
    _atomic_csv(output_dir / "predicate-confusion.csv", confusion)
    _atomic_json(output_dir / "safe-diagnosis-metadata.json", metadata)
    temporary = output_dir / "summary.md.tmp"
    temporary.write_text(_markdown(summary, matrix), encoding="utf-8")
    os.replace(temporary, output_dir / "summary.md")
    return output
