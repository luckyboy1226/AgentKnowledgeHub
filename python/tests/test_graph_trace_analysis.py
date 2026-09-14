"""Offline contracts for immutable first-loss trace analysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from services.graph_evidence_trace import edge_fingerprint
from services.graph_trace_analysis import GraphTraceAnalysisError, analyze_trace_run


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmarks" / "enterprise_20docs_60q_expanded"
RUN_ID = "trace-analysis-fake"


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _edge(predicate: str = "MEMBER_OF", direction: str = "forward") -> dict[str, object]:
    return {
        "subject": "陈航", "predicate": predicate, "raw_predicate": predicate,
        "object": "数据平台部", "direction": direction, "document_id": "a" * 36,
        "document_version": 1, "source": "D02_data_platform_team.txt",
        "evidence_key": "evidence-1", "status": "ready", "is_current": True,
    }


def _trace_dir(tmp_path: Path, *, stages: dict[str, list[dict[str, object]]] | None = None, rejections=None) -> Path:
    trace_dir = tmp_path / RUN_ID
    trace_dir.mkdir()
    selected_fingerprint = "selection-fingerprint"
    _write(trace_dir / "results.json", {
        "run_id": RUN_ID, "results": [
            {"question_id": "Q01", "mode": "vector_only"},
            {"question_id": "Q01", "mode": "graph_rag"},
        ],
    })
    _write(trace_dir / "safe-run-metadata.json", {"run_id": RUN_ID, "selection_fingerprint": selected_fingerprint})
    _write(trace_dir / "ingestion-state.json", {
        "run_id": RUN_ID, "selection_fingerprint": selected_fingerprint,
        "fixture": {"document_ids": ["D02"], "question_ids": ["Q01"]},
    })
    _write(trace_dir / "graph-trace.json", {"traces": [{
        "run_id": RUN_ID, "question_id": "Q01", "mode": "graph_rag",
        "stages": stages if stages is not None else {}, "rejections": rejections or [],
    }]})
    return trace_dir


def _all_stages(edge: dict[str, object], *, empty: str | None = None) -> dict[str, list[dict[str, object]]]:
    names = ("extracted", "normalized", "persisted", "retrieved_raw", "scope_accepted", "relevance_scored", "ranked", "entered_final_top_k", "entered_prompt")
    return {name: ([] if name == empty else [edge]) for name in names}


def test_analysis_reports_insufficient_evidence_when_ingestion_snapshots_are_absent(tmp_path):
    stages = {name: [_edge()] for name in ("retrieved_raw", "scope_accepted", "relevance_scored", "ranked", "entered_final_top_k", "entered_prompt")}
    trace_dir = _trace_dir(tmp_path, stages=stages)
    result = analyze_trace_run(trace_dir, tmp_path / "derived", BENCHMARK, strict=True)
    first = result["expected_edges"][0]
    assert first["first_loss_stage"] == "insufficient_evidence"
    assert first["evidence_boundary"] == "pre-retrieval snapshots were not persisted"
    assert first["stage_presence"]["extracted"] is False


@pytest.mark.parametrize(("empty", "expected"), [
    ("normalized", "lost_at_normalization"),
    ("persisted", "lost_at_persistence"),
    ("retrieved_raw", "lost_at_retrieval"),
    ("relevance_scored", "lost_at_relevance_filter"),
    ("ranked", "lost_at_ranking"),
    ("entered_final_top_k", "lost_at_top_k"),
    ("entered_prompt", "lost_before_prompt"),
])
def test_analysis_classifies_only_observed_adjacent_stage_transitions(tmp_path, empty, expected):
    trace_dir = _trace_dir(tmp_path, stages=_all_stages(_edge(), empty=empty))
    result = analyze_trace_run(trace_dir, tmp_path / "derived", BENCHMARK)
    assert result["expected_edges"][0]["first_loss_stage"] == expected


def test_scope_requires_a_matching_rejection_fingerprint(tmp_path):
    edge = _edge()
    stages = _all_stages(edge, empty="scope_accepted")
    trace_dir = _trace_dir(tmp_path, stages=stages, rejections=[{"fingerprint": "foreign-edge"}])
    result = analyze_trace_run(trace_dir, tmp_path / "derived", BENCHMARK)
    assert result["expected_edges"][0]["first_loss_stage"] == "insufficient_evidence"
    assert result["expected_edges"][0]["loss_reason"] == "scope_snapshot_without_matching_rejection"


def test_scope_with_matching_rejection_is_classified(tmp_path):
    edge = _edge()
    stages = _all_stages(edge, empty="scope_accepted")
    trace_dir = _trace_dir(tmp_path, stages=stages, rejections=[{"fingerprint": edge_fingerprint(edge)}])
    result = analyze_trace_run(trace_dir, tmp_path / "derived", BENCHMARK)
    assert result["expected_edges"][0]["first_loss_stage"] == "rejected_by_scope"


def test_analysis_outputs_safe_derived_reports_and_preserves_immutable_inputs(tmp_path):
    trace_dir = _trace_dir(tmp_path, stages=_all_stages(_edge()))
    before = {name: hashlib.sha256((trace_dir / name).read_bytes()).hexdigest() for name in (
        "results.json", "graph-trace.json", "safe-run-metadata.json", "ingestion-state.json",
    )}
    output = tmp_path / "derived"
    analyze_trace_run(trace_dir, output, BENCHMARK)
    after = {name: hashlib.sha256((trace_dir / name).read_bytes()).hexdigest() for name in before}
    assert before == after
    assert {item.name for item in output.iterdir()} == {
        "first-loss.json", "first-loss.csv", "stage-counts.json", "critical-edges.json",
        "summary.md", "immutable-input-hashes.json",
    }
    assert str(tmp_path) not in (output / "first-loss.json").read_text(encoding="utf-8")


def test_analysis_fails_closed_for_duplicate_trace_pair_and_existing_output(tmp_path):
    trace_dir = _trace_dir(tmp_path, stages=_all_stages(_edge()))
    payload = json.loads((trace_dir / "graph-trace.json").read_text(encoding="utf-8"))
    payload["traces"].append(payload["traces"][0])
    _write(trace_dir / "graph-trace.json", payload)
    with pytest.raises(GraphTraceAnalysisError, match="duplicate"):
        analyze_trace_run(trace_dir, tmp_path / "derived", BENCHMARK)
