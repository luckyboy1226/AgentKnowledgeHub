"""Fake-only contracts for immutable evaluation diagnosis."""
from __future__ import annotations

import csv
import json
import socket
from pathlib import Path

import pytest

from services.rag_evaluation_diagnosis import DiagnosisCase, DiagnosisError, diagnose_row, load_cases, run_diagnosis


def _fixture(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "benchmark_manifest.json").write_text(json.dumps({
        "benchmark_name": "unit", "documents": 2, "questions": 2,
        "question_categories": {"single_hop": 1, "abstention": 1},
    }), encoding="utf-8")
    (root / "benchmark_documents.json").write_text(json.dumps({"documents": [
        {"id": "D1", "filename": "a.txt", "content": "synthetic"},
        {"id": "D2", "filename": "b.txt", "content": "synthetic"},
    ]}), encoding="utf-8")
    (root / "benchmark_questions.json").write_text(json.dumps({"questions": [
        {"question_id": "Q1", "category": "single_hop", "expected_sources": ["a.txt"], "expected_relation_path": [["A", "PROVIDES_INDEX_TO", "B", "forward"]], "requires_abstention": False},
        {"question_id": "Q2", "category": "abstention", "expected_sources": [], "expected_relation_path": [], "requires_abstention": True},
    ]}), encoding="utf-8")
    return root


def _row(question_id: str, mode: str, *, edge: dict | None = None, answer_semantic: float = 1.0, v1: bool = True, source: str = "a.txt", abstention: float | None = None) -> dict:
    return {
        "run_id": "unit-run", "question_id": question_id,
        "mode": mode, "category": "single_hop" if question_id == "Q1" else "abstention",
        "success": True, "answer": "never persist this answer", "sources": [] if question_id == "Q2" else [{"source": source}],
        "graph_context_count": int(edge is not None),
        "graph_evidence_edges": [] if edge is None else [edge],
        "model_call_counts": {"graph_calls": 1 if mode == "graph_rag" else 0},
        "deterministic_score": {"question_correct": v1, "forbidden_keyword_hits": []},
        "deterministic_score_v2": {"scorer_version": "deterministic-v2", "answer_semantic_score": answer_semantic, "source_coverage_score": 1.0, "abstention_score": abstention, "forbidden_keyword_hits_v2": []},
        "forbidden_fact_violation": False,
    }


def _edge(predicate="PROVIDES_INDEX", direction="forward", subject="A", object_="B") -> dict:
    return {"subject": subject, "predicate": predicate, "object": object_, "direction": direction}


def _result_files(root: Path, rows: list[dict]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    result = root / "results.json"
    result.write_text(json.dumps({"run_id": "unit-run", "results": rows, "summary": {"graph_rag": {"relation_fidelity": 0.0}}}), encoding="utf-8")
    (root / "results.csv").write_text("placeholder\n", encoding="utf-8")
    (root / "summary.md").write_text("original\n", encoding="utf-8")
    (root / "safe-run-metadata.json").write_text("{}", encoding="utf-8")
    return result


def test_expected_edge_complete_match_and_alias_are_deterministic(tmp_path):
    cases, _ = load_cases(_fixture(tmp_path / "fixture"))
    diagnosed = diagnose_row(_row("Q1", "graph_rag", edge=_edge("PROVIDES_INDEX")), cases["Q1"], "deterministic-diagnosis-v1")
    assert diagnosed["exact_edge_match_count"] == 1
    assert diagnosed["relation_edge_recall"] == 1.0
    assert "predicate_mismatch" not in diagnosed["failure_categories"]


@pytest.mark.parametrize(("edge", "category"), [
    (_edge("DEPENDS_ON"), "predicate_mismatch"),
    (_edge("PROVIDES_INDEX", "reverse"), "direction_reversed"),
    (_edge("PROVIDES_INDEX", "forward", "B", "A"), "subject_object_mismatch"),
])
def test_relation_mismatch_categories(tmp_path, edge, category):
    cases, _ = load_cases(_fixture(tmp_path / "fixture"))
    diagnosed = diagnose_row(_row("Q1", "graph_rag", edge=edge), cases["Q1"], "deterministic-diagnosis-v1")
    assert category in diagnosed["failure_categories"]


def test_no_expected_relation_and_vector_only_are_not_scored_as_zero(tmp_path):
    cases, _ = load_cases(_fixture(tmp_path / "fixture"))
    no_relation = diagnose_row(_row("Q2", "graph_rag", abstention=1.0), cases["Q2"], "deterministic-diagnosis-v1")
    vector = diagnose_row(_row("Q1", "vector_only"), cases["Q1"], "deterministic-diagnosis-v1")
    assert no_relation["relation_edge_recall"] is None
    assert vector["relation_edge_recall"] is None
    assert "scorer_not_applicable" in no_relation["failure_categories"]


def test_prompt_evidence_boundary_and_correct_evidence_wrong_answer(tmp_path):
    cases, _ = load_cases(_fixture(tmp_path / "fixture"))
    missing = diagnose_row(_row("Q1", "graph_rag"), cases["Q1"], "deterministic-diagnosis-v1")
    wrong_answer = diagnose_row(_row("Q1", "graph_rag", edge=_edge(), answer_semantic=0.0), cases["Q1"], "deterministic-diagnosis-v1")
    assert {"expected_edge_missing_from_prompt", "insufficient_evidence"} <= set(missing["failure_categories"])
    assert "correct_evidence_but_wrong_answer" in wrong_answer["failure_categories"]


def test_forbidden_abstention_and_scorer_false_negative_are_separate(tmp_path):
    cases, _ = load_cases(_fixture(tmp_path / "fixture"))
    row = _row("Q2", "graph_rag", abstention=0.0, answer_semantic=1.0, v1=False)
    row["forbidden_fact_violation"] = True
    row["deterministic_score_v2"]["forbidden_keyword_hits_v2"] = ["positive_claim"]
    diagnosed = diagnose_row(row, cases["Q2"], "deterministic-diagnosis-v1")
    assert {"forbidden_fact_generated", "abstention_error", "scorer_false_negative"} <= set(diagnosed["failure_categories"])


def test_strict_diagnosis_preserves_inputs_and_writes_safe_derived_reports(tmp_path):
    fixture = _fixture(tmp_path / "fixture")
    rows = [
        _row("Q1", "vector_only"), _row("Q1", "graph_rag", edge=_edge()),
        _row("Q2", "vector_only", abstention=1.0), _row("Q2", "graph_rag", abstention=1.0),
    ]
    results = _result_files(tmp_path / "run", rows)
    output = tmp_path / "diagnosis"
    payload = run_diagnosis(results, fixture, output, strict=True)
    assert len(payload["results"]) == 4
    assert json.loads((output / "diagnosis.json").read_text(encoding="utf-8"))["run_id"] == "unit-run"
    assert len(list(csv.DictReader((output / "diagnosis.csv").open(encoding="utf-8")))) == 4
    rendered = (output / "summary.md").read_text(encoding="utf-8")
    serialized = (output / "diagnosis.json").read_text(encoding="utf-8")
    assert "never persist this answer" not in rendered + serialized
    assert "D:" not in serialized
    assert (output / "immutable-input-hashes.json").exists()


def test_duplicate_or_fixture_mismatch_fails_closed(tmp_path):
    fixture = _fixture(tmp_path / "fixture")
    results = _result_files(tmp_path / "run", [_row("Q1", "vector_only"), _row("Q1", "vector_only")])
    with pytest.raises(DiagnosisError, match="duplicate"):
        run_diagnosis(results, fixture, tmp_path / "out", strict=True)


def test_diagnosis_is_network_and_database_free(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path / "fixture")
    rows = [
        _row("Q1", "vector_only"), _row("Q1", "graph_rag", edge=_edge()),
        _row("Q2", "vector_only", abstention=1.0), _row("Q2", "graph_rag", abstention=1.0),
    ]
    results = _result_files(tmp_path / "run", rows)
    monkeypatch.setattr(socket, "create_connection", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("network")))
    payload = run_diagnosis(results, fixture, tmp_path / "out", strict=True)
    assert payload["summary"]["relationship_recalculation"]["applicable_graph_question_count"] == 1


@pytest.mark.parametrize(("name", "row", "case", "expected"), [
    ("complete_multi_hop", _row("Q1", "graph_rag", edge=_edge()), DiagnosisCase("QX", "multi_hop", (), (("A", "PROVIDES_INDEX", "B", "forward"),), False), set()),
    ("predicate_is_not_dependency", _row("Q1", "graph_rag", edge=_edge("DEPENDS_ON")), DiagnosisCase("QX", "single_hop", (), (("A", "PROVIDES_INDEX", "B", "forward"),), False), {"predicate_mismatch"}),
    ("reversed_edge", _row("Q1", "graph_rag", edge=_edge(direction="reverse")), DiagnosisCase("QX", "single_hop", (), (("A", "PROVIDES_INDEX", "B", "forward"),), False), {"direction_reversed"}),
    ("mismatched_endpoint", _row("Q1", "graph_rag", edge=_edge(subject="B", object_="A")), DiagnosisCase("QX", "single_hop", (), (("A", "PROVIDES_INDEX", "B", "forward"),), False), {"subject_object_mismatch"}),
    ("no_graph_call", {**_row("Q1", "graph_rag"), "model_call_counts": {"graph_calls": 0}}, DiagnosisCase("QX", "single_hop", (), (("A", "PROVIDES_INDEX", "B", "forward"),), False), {"graph_not_called"}),
    ("no_graph_evidence", {**_row("Q1", "graph_rag"), "model_call_counts": {"graph_calls": 1}}, DiagnosisCase("QX", "single_hop", (), (("A", "PROVIDES_INDEX", "B", "forward"),), False), {"no_graph_evidence"}),
    ("irrelevant_evidence", _row("Q1", "graph_rag", edge=_edge(subject="X", object_="Y")), DiagnosisCase("QX", "single_hop", (), (("A", "PROVIDES_INDEX", "B", "forward"),), False), {"irrelevant_graph_evidence", "graph_top_k_dilution"}),
    ("correct_evidence_wrong_answer", _row("Q1", "graph_rag", edge=_edge(), answer_semantic=0.0), DiagnosisCase("QX", "single_hop", (), (("A", "PROVIDES_INDEX", "B", "forward"),), False), {"correct_evidence_but_wrong_answer"}),
    ("forbidden", {**_row("Q1", "graph_rag", edge=_edge()), "forbidden_fact_violation": True}, DiagnosisCase("QX", "single_hop", (), (("A", "PROVIDES_INDEX", "B", "forward"),), False), {"forbidden_fact_generated"}),
    ("abstention", _row("Q2", "graph_rag", abstention=0.0), DiagnosisCase("Q2", "abstention", (), (), True), {"abstention_error"}),
    ("source_coverage", {**_row("Q1", "graph_rag", edge=_edge()), "deterministic_score_v2": {"scorer_version": "deterministic-v2", "answer_semantic_score": 1.0, "source_coverage_score": 0.0, "abstention_score": None, "forbidden_keyword_hits_v2": []}}, DiagnosisCase("QX", "single_hop", (), (("A", "PROVIDES_INDEX", "B", "forward"),), False), {"source_coverage_error"}),
    ("v1_false_negative", _row("Q1", "graph_rag", edge=_edge(), v1=False), DiagnosisCase("QX", "single_hop", (), (("A", "PROVIDES_INDEX", "B", "forward"),), False), {"scorer_false_negative"}),
    ("vector_na", _row("Q1", "vector_only"), DiagnosisCase("QX", "single_hop", (), (("A", "PROVIDES_INDEX", "B", "forward"),), False), {"scorer_not_applicable"}),
    ("no_expected_na", _row("Q2", "graph_rag", abstention=1.0), DiagnosisCase("Q2", "abstention", (), (), True), {"no_expected_relation", "scorer_not_applicable"}),
])
def test_diagnosis_failure_enum_contract(name, row, case, expected):
    diagnosed = diagnose_row(row, case, "deterministic-diagnosis-v1")
    assert expected <= set(diagnosed["failure_categories"]), name
    if name in {"vector_na", "no_expected_na"}:
        assert diagnosed["relation_edge_recall"] is None
