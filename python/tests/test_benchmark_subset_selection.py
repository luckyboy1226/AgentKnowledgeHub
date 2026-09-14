"""Fake-only contracts for fail-closed benchmark subset selection."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from agents.qa_agent import RetrievalMode
from services.rag_evaluation import load_evaluation_fixture, select_evaluation_subset


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "benchmarks" / "enterprise_20docs_60q_expanded"
DOCS = ("D02", "D07", "D08", "D09", "D10", "D11", "D14", "D15", "D16")
QUESTIONS = ("Q01", "Q03", "Q05", "Q06", "Q08", "Q10", "Q24", "Q41")


def _runner():
    spec = importlib.util.spec_from_file_location("subset_runner", ROOT / "scripts" / "run-rag-eval.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_subset_preserves_fixture_order_and_required_sources():
    fixture = load_evaluation_fixture(FIXTURE)
    subset = select_evaluation_subset(fixture, document_ids=tuple(reversed(DOCS)), question_ids=tuple(reversed(QUESTIONS)))
    assert tuple(item.document_id for item in subset.documents) == DOCS
    assert tuple(item.question_id for item in subset.cases) == QUESTIONS


@pytest.mark.parametrize("documents,questions,error", [
    ((), QUESTIONS, "invalid id"), (("D99",), QUESTIONS, "unknown id"),
    (("D02", "D02"), QUESTIONS, "duplicate"), (("../D02",), QUESTIONS, "invalid id"),
    (DOCS, (), "invalid id"), (DOCS, ("Q99",), "unknown id"),
    (DOCS, ("Q01", "Q01"), "duplicate"), (DOCS, ("../Q01",), "invalid id"),
])
def test_invalid_subset_is_fail_closed(documents, questions, error):
    with pytest.raises(ValueError, match=error):
        select_evaluation_subset(load_evaluation_fixture(FIXTURE), document_ids=documents, question_ids=questions)


def test_question_source_outside_document_subset_is_rejected():
    with pytest.raises(ValueError, match="source outside"):
        select_evaluation_subset(load_evaluation_fixture(FIXTURE), document_ids=("D02",), question_ids=("Q24",))


def test_selection_fingerprint_is_order_independent_and_scope_sensitive():
    runner = _runner()
    fixture = load_evaluation_fixture(FIXTURE)
    one = select_evaluation_subset(fixture, document_ids=DOCS, question_ids=QUESTIONS)
    two = select_evaluation_subset(fixture, document_ids=tuple(reversed(DOCS)), question_ids=tuple(reversed(QUESTIONS)))
    modes = (RetrievalMode.VECTOR_ONLY, RetrievalMode.GRAPH_RAG)
    full_fingerprint = runner._fixture_fingerprint(fixture.documents, fixture.cases)
    fp_one = runner._selection_fingerprint(fixture_fingerprint=full_fingerprint, documents=one.documents, cases=one.cases, modes=modes)
    fp_two = runner._selection_fingerprint(fixture_fingerprint=full_fingerprint, documents=two.documents, cases=two.cases, modes=modes)
    assert fp_one == fp_two
    changed = select_evaluation_subset(fixture, document_ids=("D02",), question_ids=("Q01",))
    assert fp_one != runner._selection_fingerprint(fixture_fingerprint=full_fingerprint, documents=changed.documents, cases=changed.cases, modes=modes)


def test_dual_hashes_bind_raw_fixture_bytes_to_actual_upload_payload():
    runner = _runner()
    fixture = load_evaluation_fixture(FIXTURE)
    subset = select_evaluation_subset(fixture, document_ids=DOCS, question_ids=QUESTIONS)
    rows = runner._document_dual_hashes(subset, subset.documents)
    assert [row["fixture_document_id"] for row in rows] == list(DOCS)
    assert all(set(row) == {"fixture_document_id", "fixture_file_sha256", "upload_payload_sha256"} for row in rows)
    assert all(len(row["fixture_file_sha256"]) == 64 and len(row["upload_payload_sha256"]) == 64 for row in rows)
    assert all(str(FIXTURE) not in str(row) for row in rows)


@pytest.mark.asyncio
async def test_subset_offline_run_is_scoped_and_vector_never_calls_graph(tmp_path):
    from services.rag_evaluation import build_offline_runner
    fixture = select_evaluation_subset(load_evaluation_fixture(FIXTURE), document_ids=DOCS, question_ids=QUESTIONS)
    payload = await build_offline_runner("subset", fixture=fixture).run(
        run_id="subset", modes=(RetrievalMode.VECTOR_ONLY, RetrievalMode.GRAPH_RAG), output_root=tmp_path,
    )
    assert len(payload["results"]) == 16
    assert {row["question_id"] for row in payload["results"]} == set(QUESTIONS)
    assert all(row["model_call_counts"]["graph_calls"] == 0 for row in payload["results"] if row["mode"] == "vector_only")
