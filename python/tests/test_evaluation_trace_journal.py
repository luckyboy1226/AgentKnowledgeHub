"""Fake-only safety contracts for the cross-request ingestion trace journal."""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import api.main as api
from agents.doc_parser_agent import DocType, DocumentChunk
from agents.knowledge_extract_agent import Entity, ExtractionResult, Relation
from services.document_processor import DocumentProcessorAdapter
from services.graph_evidence_trace import EvaluationTraceJournal, GraphEvidenceTrace
from services.graph_trace_diagnosis import diagnose_first_loss
from services.knowledge_graph import KnowledgeGraphService


def _ids():
    return "journal-run-1", str(uuid4()), str(uuid4())


def _edge(document_id: str, *, predicate: str = "DEPENDS_ON", source: str = "fixture.txt"):
    return {
        "subject": "A", "predicate": predicate, "raw_predicate": predicate,
        "object": "B", "direction": "forward", "document_id": document_id,
        "document_version": 1, "source": source, "evidence_key": "evidence-1",
        "status": "processing", "is_current": False,
        "relation_semantics_version": "relation-semantics-v1",
    }


def test_journal_records_extracted_normalized_and_persisted_without_document_text(tmp_path):
    run_id, operation_id, document_id = _ids()
    journal = EvaluationTraceJournal(root=tmp_path, run_id=run_id, operation_id=operation_id, fixture_id="D01")
    for stage in ("extracted", "normalized", "persisted"):
        journal.record_edges(stage, [_edge(document_id, source=r"C:\private\fixture.txt")])
    rows = EvaluationTraceJournal.read_events(root=tmp_path, run_id=run_id)
    assert [row["stage"] for row in rows] == ["extracted", "normalized", "persisted"]
    assert all(row["source"] == "fixture.txt" and "private" not in json.dumps(row) for row in rows)


def test_journal_marks_an_observed_empty_stage_without_inventing_an_edge(tmp_path):
    run_id, operation_id, document_id = _ids()
    journal = EvaluationTraceJournal(root=tmp_path, run_id=run_id, operation_id=operation_id)
    journal.record_stage(
        "normalized", document_id=document_id, document_version=1,
        source="fixture.txt", status="processing", is_current=False,
    )
    rows = EvaluationTraceJournal.read_events(root=tmp_path, run_id=run_id)
    assert len(rows) == 1 and rows[0]["event_kind"] == "stage_observed"
    trace = GraphEvidenceTrace(run_id=run_id, question_id="Q01", scope_verified=True, allowed_document_ids_count=1)
    assert EvaluationTraceJournal.merge_into_question_trace(
        trace, root=tmp_path, run_id=run_id, allowed_document_ids=frozenset({document_id})
    ) == 0
    assert trace.to_dict()["stages"]["normalized"] == []


def test_journal_deduplicates_retry_and_concurrent_appends(tmp_path):
    run_id, operation_id, document_id = _ids()
    journal = EvaluationTraceJournal(root=tmp_path, run_id=run_id, operation_id=operation_id)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: journal.record_edges("extracted", [_edge(document_id)]), range(8)))
    rows = EvaluationTraceJournal.read_events(root=tmp_path, run_id=run_id)
    assert len(rows) == 1 and rows[0]["stage"] == "extracted"


@pytest.mark.parametrize("run_id", ["", "../escape", "bad/run", "x" * 122])
def test_journal_rejects_unsafe_run_id_and_path_traversal(tmp_path, run_id):
    with pytest.raises(ValueError):
        EvaluationTraceJournal(root=tmp_path, run_id=run_id, operation_id=str(uuid4()))


def test_journal_rejects_invalid_operation_or_document_identity(tmp_path):
    with pytest.raises(ValueError):
        EvaluationTraceJournal(root=tmp_path, run_id="journal-run-1", operation_id="not-a-uuid")
    run_id, operation_id, _ = _ids()
    journal = EvaluationTraceJournal(root=tmp_path, run_id=run_id, operation_id=operation_id)
    with pytest.raises(ValueError):
        journal.record_edges("extracted", [_edge("not-a-uuid")])


@pytest.mark.parametrize("stage,version", [("retrieved_raw", 1), ("extracted", 0), ("persisted", True)])
def test_journal_stage_marker_rejects_non_ingestion_stage_or_invalid_version(tmp_path, stage, version):
    run_id, operation_id, document_id = _ids()
    journal = EvaluationTraceJournal(root=tmp_path, run_id=run_id, operation_id=operation_id)
    with pytest.raises(ValueError):
        journal.record_stage(stage, document_id=document_id, document_version=version, source="fixture.txt")


def test_journal_stage_marker_sanitizes_source_to_basename(tmp_path):
    run_id, operation_id, document_id = _ids()
    journal = EvaluationTraceJournal(root=tmp_path, run_id=run_id, operation_id=operation_id)
    journal.record_stage("extracted", document_id=document_id, document_version=1, source=r"C:\private\fixture.txt")
    assert EvaluationTraceJournal.read_events(root=tmp_path, run_id=run_id)[0]["source"] == "fixture.txt"


def test_journal_events_are_isolated_by_run_id(tmp_path):
    _, operation_id, document_id = _ids()
    first = EvaluationTraceJournal(root=tmp_path, run_id="journal-run-one", operation_id=operation_id)
    second = EvaluationTraceJournal(root=tmp_path, run_id="journal-run-two", operation_id=str(uuid4()))
    first.record_edges("extracted", [_edge(document_id)])
    second.record_edges("extracted", [_edge(document_id)])
    assert len(EvaluationTraceJournal.read_events(root=tmp_path, run_id="journal-run-one")) == 1
    assert len(EvaluationTraceJournal.read_events(root=tmp_path, run_id="journal-run-two")) == 1


def test_journal_read_fails_closed_for_corrupt_or_duplicate_rows(tmp_path):
    run_id, operation_id, document_id = _ids()
    journal = EvaluationTraceJournal(root=tmp_path, run_id=run_id, operation_id=operation_id)
    journal.record_edges("extracted", [_edge(document_id)])
    row = EvaluationTraceJournal.read_events(root=tmp_path, run_id=run_id)[0]
    journal.path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        EvaluationTraceJournal.read_events(root=tmp_path, run_id=run_id)


def test_journal_rejection_has_only_bounded_identity_and_reason(tmp_path):
    run_id, operation_id, document_id = _ids()
    journal = EvaluationTraceJournal(root=tmp_path, run_id=run_id, operation_id=operation_id)
    journal.reject("dangling_object", stage="normalized", edge=_edge(document_id), detail="api-key=never-persist")
    row = EvaluationTraceJournal.read_events(root=tmp_path, run_id=run_id)[0]
    assert row["event_kind"] == "rejection" and row["reason"] == "dangling_object"
    assert "api-key" not in json.dumps(row)


def test_journal_merges_allowlisted_ingestion_stages_into_qa_trace(tmp_path):
    run_id, operation_id, allowed = _ids()
    other = str(uuid4())
    journal = EvaluationTraceJournal(root=tmp_path, run_id=run_id, operation_id=operation_id)
    journal.record_edges("extracted", [_edge(allowed)])
    journal.record_edges("normalized", [_edge(allowed)])
    journal.record_edges("persisted", [_edge(allowed)])
    journal.record_edges("persisted", [_edge(other)])
    trace = GraphEvidenceTrace(run_id=run_id, question_id="Q01", scope_verified=True, allowed_document_ids_count=1)
    assert EvaluationTraceJournal.merge_into_question_trace(trace, root=tmp_path, run_id=run_id, allowed_document_ids=frozenset({allowed})) == 3
    payload = trace.to_dict()
    assert set(payload["stages"]) == {"extracted", "normalized", "persisted"}
    assert all(edge["document_id"] == allowed for edges in payload["stages"].values() for edge in edges)


def test_journal_merge_requires_a_nonempty_validated_allowlist(tmp_path):
    run_id, operation_id, _ = _ids()
    trace = GraphEvidenceTrace(run_id=run_id, question_id="Q01", scope_verified=True, allowed_document_ids_count=1)
    with pytest.raises(ValueError):
        EvaluationTraceJournal.merge_into_question_trace(trace, root=tmp_path, run_id=run_id, allowed_document_ids=frozenset())


def test_journal_merge_does_not_turn_rejected_ingestion_edge_into_evidence(tmp_path):
    run_id, operation_id, document_id = _ids()
    journal = EvaluationTraceJournal(root=tmp_path, run_id=run_id, operation_id=operation_id)
    journal.reject("dangling_subject", stage="normalized", edge=_edge(document_id))
    trace = GraphEvidenceTrace(run_id=run_id, question_id="Q01", scope_verified=True, allowed_document_ids_count=1)
    assert EvaluationTraceJournal.merge_into_question_trace(
        trace, root=tmp_path, run_id=run_id, allowed_document_ids=frozenset({document_id})
    ) == 0
    assert "normalized" not in trace.to_dict()["stages"]


def test_complete_fake_chain_and_each_first_loss_transition_are_distinguishable(tmp_path):
    run_id, operation_id, document_id = _ids()
    journal = EvaluationTraceJournal(root=tmp_path, run_id=run_id, operation_id=operation_id)
    edge = _edge(document_id)
    for stage in ("extracted", "normalized", "persisted"):
        journal.record_edges(stage, [edge])
    trace = GraphEvidenceTrace(run_id=run_id, question_id="Q01", scope_verified=True, allowed_document_ids_count=1)
    EvaluationTraceJournal.merge_into_question_trace(trace, root=tmp_path, run_id=run_id, allowed_document_ids=frozenset({document_id}))
    for stage in ("retrieved_raw", "scope_accepted", "relevance_scored", "ranked", "entered_final_top_k", "entered_prompt"):
        trace.record_edges(stage, [edge])
    diagnosis = diagnose_first_loss([("A", "DEPENDS_ON", "B", "forward")], trace.to_dict())["edge_diagnoses"][0]
    assert diagnosis["first_loss_stage"] == "entered_prompt"
    snapshots = trace.to_dict()["stages"]
    for stage, expected in (("extracted", "lost_at_extraction"), ("normalized", "lost_at_normalization"), ("persisted", "lost_at_persistence"), ("retrieved_raw", "lost_at_retrieval")):
        broken = {name: list(edges) for name, edges in snapshots.items()}
        broken[stage] = []
        assert diagnose_first_loss([("A", "DEPENDS_ON", "B", "forward")], {"stages": broken})["edge_diagnoses"][0]["first_loss_stage"] == expected


def test_journal_has_no_provider_or_database_client_side_effect(tmp_path, monkeypatch):
    run_id, operation_id, document_id = _ids()
    journal = EvaluationTraceJournal(root=tmp_path, run_id=run_id, operation_id=operation_id)
    journal.record_edges("extracted", [_edge(document_id)])
    assert (tmp_path / run_id / "graph-ingestion-trace.jsonl").is_file()
    assert not hasattr(journal, "client") and not hasattr(journal, "provider")


class _TraceParser:
    async def parse(self, _path):
        return [DocumentChunk("A depends on B", "parser", 0, DocType.TEXT, {})]


class _TraceExtractor:
    async def extract(self, _chunks):
        return [ExtractionResult(
            [Entity("A", "Concept"), Entity("B", "Concept")],
            [Relation("A", "depends_on", "B", 0.9)], [],
        )]


@pytest.mark.asyncio
async def test_processor_records_observed_extraction_and_normalization_stages(tmp_path):
    run_id, operation_id, document_id = _ids()
    journal = EvaluationTraceJournal(root=tmp_path, run_id=run_id, operation_id=operation_id)
    processor = DocumentProcessorAdapter(_TraceParser(), _TraceExtractor(), temp_root=tmp_path / "temp")
    await processor.prepare(
        content=b"A depends on B", filename="fixture.txt", document_id=document_id,
        version=1, content_hash="a" * 64, operation_id=operation_id, graph_trace=journal,
    )
    rows = EvaluationTraceJournal.read_events(root=tmp_path, run_id=run_id)
    assert {row["stage"] for row in rows} >= {"extracted", "normalized"}
    assert {row["stage"] for row in rows if row["event_kind"] == "stage_observed"} >= {"extracted", "normalized"}


class _TraceProbe:
    def __init__(self):
        self.edges = []
        self.stages = []

    def record_edges(self, stage, edges):
        self.edges.append((stage, list(edges)))

    def record_stage(self, stage, **kwargs):
        self.stages.append((stage, kwargs))


@pytest.mark.asyncio
async def test_graph_persistence_marker_is_recorded_only_after_successful_write(monkeypatch):
    graph, trace = KnowledgeGraphService(), _TraceProbe()
    document_id = str(uuid4())

    async def successful_write(*_args):
        return {"mentions": 0, "evidence": 1}

    monkeypatch.setattr(graph, "_execute_write", successful_write)
    await graph.stage_document_version(document_id, 1, "hash", "fixture.txt", [], [Relation("A", "depends_on", "B")], trace)
    assert trace.stages == [("persisted", {
        "document_id": document_id, "document_version": 1, "source": "fixture.txt",
        "status": "processing", "is_current": False,
    })]


@pytest.mark.asyncio
async def test_graph_persistence_marker_is_absent_after_failed_write(monkeypatch):
    graph, trace = KnowledgeGraphService(), _TraceProbe()

    async def failed_write(*_args):
        raise RuntimeError("transaction failed")

    monkeypatch.setattr(graph, "_execute_write", failed_write)
    with pytest.raises(RuntimeError):
        await graph.stage_document_version(str(uuid4()), 1, "hash", "fixture.txt", [], [], trace)
    assert trace.edges == [] and trace.stages == []


class _FakeCoordinator:
    def __init__(self):
        self.calls = []

    async def create_document_version(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "operation_id": kwargs["operation_id"], "document_id": str(uuid4()), "version": 1,
            "content_hash": "a" * 64, "status": "succeeded", "changed": True,
            "completed_steps": [], "processing_metadata": {},
        }


@pytest.fixture
def api_client(monkeypatch, tmp_path):
    coordinator = _FakeCoordinator()
    monkeypatch.setattr(api, "document_coordinator", coordinator)
    api.app.state.document_coordinator = coordinator
    monkeypatch.setattr(api, "PROJECT_ROOT", tmp_path)
    return TestClient(api.app, raise_server_exceptions=False), coordinator


def test_normal_upload_does_not_create_or_pass_trace(api_client):
    client, coordinator = api_client
    response = client.post("/api/documents", files={"file": ("safe.txt", b"safe", "text/plain")})
    assert response.status_code == 200 and "graph_trace" not in coordinator.calls[0]
    assert "trace" not in response.json()


def test_internal_trace_header_is_not_exposed_in_openapi(api_client):
    client, _ = api_client
    parameters = client.get("/openapi.json").json()["paths"]["/api/documents"]["post"].get("parameters", [])
    assert all(item.get("name") != "X-Evaluation-Trace-Run-Id" for item in parameters)


def test_trace_header_is_rejected_when_disabled_or_non_loopback(api_client, monkeypatch):
    client, coordinator = api_client
    header = {"X-Evaluation-Trace-Run-Id": "journal-run-1"}
    assert client.post("/api/documents", headers=header, files={"file": ("safe.txt", b"safe", "text/plain")}).status_code == 403
    monkeypatch.setattr(api.settings, "evaluation_trace_enabled", True)
    assert client.post("/api/documents", headers=header, files={"file": ("safe.txt", b"safe", "text/plain")}).status_code == 403
    assert not coordinator.calls


def test_enabled_loopback_trace_is_internal_and_validated(api_client, monkeypatch):
    client, coordinator = api_client
    monkeypatch.setattr(api.settings, "evaluation_trace_enabled", True)
    monkeypatch.setattr(api, "_is_loopback_request", lambda _request: True)
    operation_id = str(uuid4())
    headers = {"X-Evaluation-Trace-Run-Id": "journal-run-1", "X-Operation-Id": operation_id}
    response = client.post("/api/documents", headers=headers, files={"file": ("safe.txt", b"safe", "text/plain")})
    assert response.status_code == 200
    trace = coordinator.calls[0]["graph_trace"]
    assert isinstance(trace, EvaluationTraceJournal) and trace.operation_id == operation_id
    invalid = client.post("/api/documents", headers={**headers, "X-Evaluation-Trace-Run-Id": "../bad"}, files={"file": ("safe.txt", b"safe", "text/plain")})
    assert invalid.status_code == 422
