"""Fake-only contracts for S4.6a evaluation graph evidence tracing."""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from agents.qa_agent import RetrievalMode
from agents.knowledge_extract_agent import Entity, ExtractionResult, Relation
from services.document_processor import DocumentProcessorAdapter
from services.graph_evidence_trace import GraphEvidenceTrace
from services.knowledge_graph import KnowledgeGraphService
from services.rag_evaluation import RAGEvaluationRunner, build_offline_runner


def _edge(**overrides):
    value = {
        "subject": "北极星", "predicate": "PROVIDES_INDEX", "raw_predicate": "提供检索索引",
        "object": "天枢", "direction": "forward", "document_id": str(uuid4()),
        "document_version": 1, "source": "fixture.txt", "evidence_key": "evidence-1",
        "relation_semantics_version": "v1",
    }
    value.update(overrides)
    return value


def test_trace_keeps_only_safe_edge_metadata_not_document_text():
    trace = GraphEvidenceTrace(run_id="fake", question_id="Q03", scope_verified=True, allowed_document_ids_count=1)
    trace.record_edges("retrieved_raw", [_edge(source="C:/private/document.txt")])
    payload = trace.to_dict()
    assert payload["stages"]["retrieved_raw"][0]["source"] == "document.txt"
    assert "private" not in json.dumps(payload, ensure_ascii=False)


@pytest.mark.parametrize("verified,count", [(False, 1), (True, 0)])
def test_trace_refuses_unverified_or_empty_scope(verified, count):
    with pytest.raises(ValueError, match="verified nonempty"):
        GraphEvidenceTrace(run_id="fake", question_id="Q01", scope_verified=verified, allowed_document_ids_count=count)


@pytest.mark.parametrize("stage", ["extracted", "normalized", "persisted"])
def test_pre_retrieval_stages_are_available_to_a_future_ingestion_trace(stage):
    trace = GraphEvidenceTrace(run_id="fake", question_id="Q01", scope_verified=True, allowed_document_ids_count=1)
    trace.record_edges(stage, [_edge()])
    captured = trace.to_dict()["stages"][stage][0]
    assert captured["predicate"] == "PROVIDES_INDEX"
    assert captured["raw_predicate"] == "提供检索索引"


@pytest.mark.asyncio
async def test_graph_evaluation_trace_covers_retrieval_scope_rank_and_prompt(tmp_path):
    runner = build_offline_runner()
    case = next(item for item in runner.cases if item.question_id == "Q03")
    await runner.run(run_id="trace-q03", modes=(RetrievalMode.GRAPH_RAG,), output_root=tmp_path)
    trace_path = tmp_path / "trace-q03" / "graph-trace.json"
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    trace = next(item for item in payload["traces"] if item["question_id"] == "Q03")
    assert trace["scope_verified"] is True
    assert trace["stages"]["retrieved_raw"]
    assert trace["stages"]["scope_accepted"]
    assert trace["stages"]["relevance_scored"]
    assert trace["stages"]["ranked"]
    assert trace["stages"]["entered_prompt"]


@pytest.mark.asyncio
async def test_vector_only_creates_no_graph_trace_or_graph_call(tmp_path):
    runner = build_offline_runner()
    await runner.run(run_id="trace-vector", modes=(RetrievalMode.VECTOR_ONLY,), output_root=tmp_path)
    assert runner.agent.knowledge_graph.calls == 0
    assert not (tmp_path / "trace-vector" / "graph-trace.json").exists()


@pytest.mark.asyncio
async def test_recovery_keeps_completed_question_trace_without_rewriting_it(tmp_path):
    initial = build_offline_runner()
    cases = initial.cases[:2]
    initial_runner = RAGEvaluationRunner(initial.agent, cases, scope=initial.scope)
    first = await initial_runner.run(run_id="trace-recovery", modes=(RetrievalMode.GRAPH_RAG,), output_root=tmp_path)
    trace_path = tmp_path / "trace-recovery" / "graph-trace.json"
    before = json.loads(trace_path.read_text(encoding="utf-8"))
    resumed = build_offline_runner()
    resumed_runner = RAGEvaluationRunner(resumed.agent, cases, scope=resumed.scope)
    await resumed_runner.run(
        run_id="trace-recovery", modes=(RetrievalMode.GRAPH_RAG,), output_root=tmp_path,
        existing_results=[row for row in first["results"] if row["question_id"] == cases[0].question_id],
    )
    after = json.loads(trace_path.read_text(encoding="utf-8"))
    before_q1 = next(trace for trace in before["traces"] if trace["question_id"] == cases[0].question_id)
    after_q1 = next(trace for trace in after["traces"] if trace["question_id"] == cases[0].question_id)
    assert after_q1 == before_q1


@pytest.mark.parametrize("reason", [
    "invalid_relation", "dangling_subject", "dangling_object", "unsafe_predicate",
    "missing_provenance", "outside_allowlist", "inactive", "not_ready", "legacy_disallowed",
    "malformed_record", "entity_query_miss", "predicate_query_miss", "low_relevance",
    "top_k_truncated", "duplicate_edge", "direction_mismatch", "endpoint_mismatch",
])
def test_trace_rejection_reasons_are_finite(reason):
    trace = GraphEvidenceTrace(run_id="fake", question_id="Q01", scope_verified=True, allowed_document_ids_count=1)
    trace.reject(reason)
    assert trace.to_dict()["rejections"][0]["reason"] == reason


class _ReadOnlyGraph(KnowledgeGraphService):
    def __init__(self):
        super().__init__()
        self.calls = []

    async def execute_cypher(self, cypher, params=None):
        self.calls.append((cypher, params))
        return [{"subject": "A", "predicate": "OWNS", "object": "B", "document_id": params["allowed_document_ids"][0]}]


@pytest.mark.asyncio
async def test_scoped_snapshot_is_uuid_validated_parameterized_and_read_only():
    service = _ReadOnlyGraph()
    identifier = str(uuid4())
    rows = await service.list_scoped_evaluation_evidence(frozenset({identifier}))
    assert rows and service.calls[0][1] == {"allowed_document_ids": [identifier]}
    assert "$allowed_document_ids" in service.calls[0][0]
    assert "MATCH (dv:DocumentVersion)-[:MENTIONS]->(a:Entity)-[rel]->(b:Entity)" in service.calls[0][0]


@pytest.mark.parametrize("scope", [frozenset(), frozenset({"not-a-uuid"})])
@pytest.mark.asyncio
async def test_scoped_snapshot_fails_closed_before_query(scope):
    service = _ReadOnlyGraph()
    with pytest.raises(ValueError):
        await service.list_scoped_evaluation_evidence(scope)
    assert service.calls == []


def test_trace_merges_duplicate_fingerprint_and_keeps_safe_source_set():
    trace = GraphEvidenceTrace(run_id="fake", question_id="Q01", scope_verified=True, allowed_document_ids_count=1)
    edge = _edge(source="one.txt")
    trace.record_edges("retrieved_raw", [edge, {**edge, "source": "two.txt"}])
    captured = trace.to_dict()["stages"]["retrieved_raw"]
    assert len(captured) == 1 and captured[0]["sources"] == ["one.txt", "two.txt"]


@pytest.mark.asyncio
async def test_normal_evaluation_without_trace_has_no_trace_side_effect():
    runner = build_offline_runner()
    case = next(item for item in runner.cases if item.question_id == "Q03")
    plan = await runner.agent.build_evaluation_query_plan(case.question, run_id="normal", question_id="Q03")
    await runner.agent.answer_with_evaluation_plan(plan, RetrievalMode.GRAPH_RAG, scope=runner.scope)
    assert runner.agent.knowledge_graph.calls > 0


def test_processor_normalization_and_dangling_rejection_flow_into_optional_trace(tmp_path):
    trace = GraphEvidenceTrace(run_id="fake", question_id="Q01", scope_verified=True, allowed_document_ids_count=1)
    processor = DocumentProcessorAdapter(None, None, temp_root=tmp_path)
    identifier = str(uuid4())
    _entities, relations, dropped = processor._normalize_extraction(
        [ExtractionResult(
            entities=[Entity("A", "Person"), Entity("B", "Project")],
            relations=[Relation("A", "owns", "B", 1.0), Relation("A", "owns", "missing", 1.0)], events=[],
        )],
        graph_trace=trace, document_id=identifier, version=1, source="fixture.txt",
    )
    payload = trace.to_dict()
    assert dropped == 1 and relations[0].relation == "OWNS"
    assert payload["stages"]["normalized"][0]["predicate"] == "OWNS"
    assert payload["stages"]["normalized"][0]["evidence_key"]
    assert payload["rejections"][0]["reason"] == "dangling_object"


@pytest.mark.asyncio
async def test_graph_stage_records_persisted_provenance_when_trace_is_explicitly_supplied():
    service = KnowledgeGraphService()

    async def fake_write(*_args):
        return {"mentions": 2, "evidence": 1}

    service._execute_write = fake_write
    trace = GraphEvidenceTrace(run_id="fake", question_id="Q01", scope_verified=True, allowed_document_ids_count=1)
    identifier = str(uuid4())
    await service.stage_document_version(
        identifier, 1, "a" * 64, "fixture.txt", [Entity("A", "Person")],
        [Relation("A", "owns", "B", 1.0)], graph_trace=trace,
    )
    edge = trace.to_dict()["stages"]["persisted"][0]
    assert edge["document_id"] == identifier and edge["status"] == "processing"
    assert edge["is_current"] is False and edge["evidence_key"]
