"""Fake-only contracts for the immutable 20-document enterprise fixture."""

from __future__ import annotations

import asyncio
import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from agents.qa_agent import RetrievalMode
from services.rag_evaluation import (
    DeterministicScorerV2,
    EvaluationCase,
    _nearest_rank,
    build_offline_runner,
    load_evaluation_fixture,
    score_relation_fidelity,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = (
    PROJECT_ROOT / "benchmarks" / "enterprise_20docs_60q"
)


@pytest.fixture(scope="module")
def enterprise_fixture():
    return load_evaluation_fixture(FIXTURE_ROOT)


def _load_runner_module():
    path = PROJECT_ROOT / "scripts" / "run-rag-eval.py"
    spec = importlib.util.spec_from_file_location("enterprise_run_rag_eval", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_enterprise_fixture_has_exactly_twenty_documents_and_sixty_questions(enterprise_fixture):
    assert len(enterprise_fixture.documents) == 20
    assert len(enterprise_fixture.cases) == 60
    assert Counter(case.category for case in enterprise_fixture.cases) == {
        "single_hop": 12, "multi_hop": 12, "constraint": 12,
        "distractor": 12, "abstention": 12,
    }
    assert all((FIXTURE_ROOT / "documents" / document.filename).read_text(encoding="utf-8").strip() == document.content for document in enterprise_fixture.documents)


def test_relation_fidelity_requires_exact_directed_predicate():
    case = EvaluationCase(
        "relation", "关系是什么？", "multi_hop", (), (),
        expected_relation_path=(("北极星", "PROVIDES_INDEX_TO", "天枢", "forward"),),
    )
    matching = SimpleNamespace(
        retrieval_type="graph",
        metadata={"graph_evidence": [{
            "subject": "北极星", "predicate": "PROVIDES_INDEX_TO", "object": "天枢", "direction": "forward",
        }]},
    )
    confused = SimpleNamespace(
        retrieval_type="graph",
        metadata={"graph_evidence": [{
            "subject": "北极星", "predicate": "DEPENDS_ON", "object": "天枢", "direction": "forward",
        }]},
    )
    assert score_relation_fidelity(case, [matching])["relation_fidelity_score"] == 1.0
    assert score_relation_fidelity(case, [confused])["relation_fidelity_score"] == 0.0


def test_fixture_owned_abstention_expression_is_scored_without_a_model(enterprise_fixture):
    case = next(case for case in enterprise_fixture.cases if case.question_id == "Q49")
    score = DeterministicScorerV2.score(
        case, answer=case.expected_answer, sources=[], mode=RetrievalMode.VECTOR_ONLY, graph_context_count=0,
    )
    assert score["abstention_score"] == 1.0


def test_nearest_rank_p95_is_deterministic():
    assert _nearest_rank([1.0, 2.0, 3.0, 4.0, 5.0], 95) == 5.0


def test_exact_cleanup_accepts_arbitrary_fixture_sized_document_set():
    module = _load_runner_module()
    created = [{"document_id": str(uuid4())} for _ in range(20)]
    assert module.exact_cleanup_document_ids(created) == tuple(item["document_id"] for item in created)


@pytest.mark.asyncio
async def test_fixture_driven_offline_runner_writes_all_one_hundred_twenty_results(tmp_path, enterprise_fixture):
    runner = build_offline_runner("enterprise-offline", fixture=enterprise_fixture)
    payload = await runner.run(
        run_id="enterprise-offline",
        modes=(RetrievalMode.VECTOR_ONLY, RetrievalMode.GRAPH_RAG),
        output_root=tmp_path,
    )

    assert len(payload["results"]) == 120
    assert len([row for row in payload["results"] if row["mode"] == "vector_only"]) == 60
    assert len([row for row in payload["results"] if row["mode"] == "graph_rag"]) == 60
    assert all(row["model_call_counts"]["graph_calls"] == 0 for row in payload["results"] if row["mode"] == "vector_only")
    assert all(row["model_call_counts"]["shared_preprocess_chat"] == 2 for row in payload["results"])
    assert all(row["scope_verified"] is True for row in payload["results"])
    assert all(
        source["document_id"] in runner.scope.allowed_document_ids
        for row in payload["results"] for source in row["sources"]
    )
    for mode in ("vector_only", "graph_rag"):
        categories = payload["summary"][mode]["categories"]
        assert {category: metrics["total"] for category, metrics in categories.items()} == {
            "single_hop": 12, "multi_hop": 12, "constraint": 12,
            "distractor": 12, "abstention": 12,
        }
        assert "relation_fidelity" in payload["summary"][mode]
        assert "forbidden_fact_violation_rate" in payload["summary"][mode]
        assert "latency_p95_ms" in payload["summary"][mode]

    output = tmp_path / "enterprise-offline"
    assert len(json.loads((output / "results.json").read_text(encoding="utf-8"))["results"]) == 120
    assert len(list(csv.DictReader((output / "results.csv").open(encoding="utf-8")))) == 120
    assert "Offline RAG evaluation" in (output / "summary.md").read_text(encoding="utf-8")


def test_fixture_loader_refuses_missing_or_changed_fixed_text(tmp_path):
    with pytest.raises(ValueError, match="Invalid evaluation fixture"):
        load_evaluation_fixture(tmp_path)
