"""Fake-only S4 document allowlist contracts; no external service is contacted."""

from __future__ import annotations

from uuid import uuid4

import pytest

from agents.doc_parser_agent import DocType, DocumentChunk
from agents.qa_agent import RetrievalMode
from services.evaluation_scope import EvaluationScope, EvaluationScopeError
from services.knowledge_graph import KnowledgeGraphService
from services.rag_evaluation import RAGEvaluationRunner, build_offline_runner
from services.vector_store import VectorStoreService


class FakeEmbeddings:
    dimensions = 3

    async def aembed_query(self, _query):
        return [1.0, 0.0, 0.0]


class FakeCollection:
    def __init__(self, rows):
        self.rows = rows
        self.last_query_n_results = None

    def query(self, query_embeddings, n_results, include):
        del query_embeddings, include
        self.last_query_n_results = n_results
        selected = list(self.rows)[:n_results]
        return {
            "documents": [[self.rows[item]["document"] for item in selected]],
            "metadatas": [[self.rows[item]["metadata"] for item in selected]],
            "distances": [[0.1 for _ in selected]],
        }


def _versioned_row(document_id, *, version=1, source="allowed.txt"):
    return {
        "document": f"evidence for {document_id}",
        "metadata": {
            "document_id": str(document_id), "document_version": version,
            "is_current": True, "status": "ready", "source": source,
        },
    }


@pytest.fixture
def document_ids():
    return str(uuid4()), str(uuid4())


@pytest.fixture
def scoped_vector(document_ids):
    allowed, foreign = document_ids
    service = VectorStoreService(FakeEmbeddings())
    service._backend = "chroma"
    service._store = FakeCollection({
        "allowed": _versioned_row(allowed),
        "foreign": _versioned_row(foreign, source="foreign.txt"),
        "legacy": {"document": "legacy", "metadata": {"doc_id": "old", "source": "legacy.txt"}},
    })
    return service


def test_scope_accepts_only_uploaded_uuid_ids(document_ids):
    scope = EvaluationScope.from_uploaded_document_ids("scope-unit", {document_ids[0]})
    assert scope.is_verified() and scope.allowed_document_ids == frozenset({document_ids[0]})


def test_empty_scope_fails_closed():
    with pytest.raises(EvaluationScopeError, match="cannot be empty"):
        EvaluationScope.from_uploaded_document_ids("scope-unit", [])


@pytest.mark.parametrize("bad_id", ["doc-a", "../document", "", "source.txt"])
def test_invalid_scope_document_id_fails_closed(bad_id):
    with pytest.raises(EvaluationScopeError):
        EvaluationScope.from_uploaded_document_ids("scope-unit", [bad_id])


@pytest.mark.asyncio
async def test_chroma_scope_returns_only_allowed_document(scoped_vector, document_ids):
    diagnostics = {}
    found = await scoped_vector.search(
        "question", allowed_document_ids=frozenset({document_ids[0]}), scope_diagnostics=diagnostics
    )
    assert [row[0]["metadata"]["document_id"] for row in found] == [document_ids[0]]
    assert diagnostics["vector_scope_rejected_count"] == 2


@pytest.mark.asyncio
async def test_chroma_scope_never_falls_back_when_only_foreign_rows_exist(document_ids):
    service = VectorStoreService(FakeEmbeddings())
    service._backend = "chroma"
    service._store = FakeCollection({"foreign": _versioned_row(document_ids[1])})
    diagnostics = {}
    assert await service.search(
        "question", allowed_document_ids=frozenset({document_ids[0]}), scope_diagnostics=diagnostics
    ) == []
    assert diagnostics["vector_scope_rejected_count"] == 1


class FakeResult:
    def __init__(self, records):
        self.records = records

    async def data(self):
        return self.records


class FakeSession:
    def __init__(self, driver):
        self.driver = driver

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def run(self, query, params=None):
        self.driver.queries.append((query, params or {}))
        return FakeResult(self.driver.records)


class FakeDriver:
    def __init__(self, records):
        self.records = records
        self.queries = []

    def session(self):
        return FakeSession(self)


@pytest.mark.asyncio
async def test_neo4j_scope_filters_foreign_and_legacy_records(document_ids):
    allowed, foreign = document_ids
    graph = KnowledgeGraphService()
    graph._driver = FakeDriver([
        {"document_id": allowed, "document_version": 1, "provenance_source": "allowed.txt"},
        {"document_id": foreign, "document_version": 1, "provenance_source": "foreign.txt"},
        {"provenance_source": "legacy.txt"},
    ])
    diagnostics = {}

    records = await graph.get_neighbors(
        "Alice", allowed_document_ids=frozenset({allowed}), scope_diagnostics=diagnostics
    )

    assert records == [{"document_id": allowed, "document_version": 1, "provenance_source": "allowed.txt"}]
    assert diagnostics["graph_scope_rejected_count"] == 2
    query, parameters = graph._driver.queries[-1]
    assert "rel.document_id IN $allowed_document_ids" in query
    assert parameters["allowed_document_ids"] == [allowed]
    assert "rel.document_id IS NULL" not in query


@pytest.mark.asyncio
async def test_unscoped_graph_keeps_legacy_compatibility(document_ids):
    graph = KnowledgeGraphService()
    graph._driver = FakeDriver([{"provenance_source": "legacy.txt"}])
    assert await graph.get_neighbors("Alice") == [{"provenance_source": "legacy.txt"}]
    assert "rel.document_id IS NULL OR" in graph._driver.queries[-1][0]


@pytest.mark.asyncio
async def test_runner_scope_filters_context_sources_and_final_prompt(tmp_path):
    runner = build_offline_runner()
    payload = await runner.run(run_id="scope-output", modes=("vector_only", "graph_rag"), output_root=tmp_path)
    allowed = runner.scope.allowed_document_ids

    for row in payload["results"]:
        assert row["scope_verified"] is True
        assert row["allowed_document_ids_count"] == len(allowed)
        assert all(source["document_id"] in allowed for source in row["sources"])
        if row["category"] != "unanswerable":
            assert row["vector_scope_rejected_count"] >= 2
    graph_rows = [row for row in payload["results"] if row["mode"] == "graph_rag"]
    assert any(row["graph_scope_rejected_count"] > 0 for row in graph_rows)
    assert all("foreign evidence" not in prompt and "legacy evidence" not in prompt for prompt in runner.agent.llm.final_prompts)


@pytest.mark.asyncio
async def test_vector_only_still_makes_zero_graph_calls_with_scope():
    runner = build_offline_runner()
    case = runner.cases[0]
    plan = await runner.agent.build_evaluation_query_plan(case.question, run_id="scope", question_id=case.question_id)
    result = await runner.agent.answer_with_evaluation_plan(plan, RetrievalMode.VECTOR_ONLY, scope=runner.scope)
    assert runner.agent.knowledge_graph.calls == 0
    assert all(context.retrieval_type != "graph" for context in result.contexts)


@pytest.mark.asyncio
async def test_graph_rag_contexts_are_limited_to_allowlist():
    runner = build_offline_runner()
    case = next(item for item in runner.cases if item.question_id == "Q03")
    plan = await runner.agent.build_evaluation_query_plan(case.question, run_id="scope", question_id=case.question_id)
    result = await runner.agent.answer_with_evaluation_plan(plan, RetrievalMode.GRAPH_RAG, scope=runner.scope)
    graph_contexts = [context for context in result.contexts if context.retrieval_type == "graph"]
    assert graph_contexts
    assert all(context.metadata["document_id"] in runner.scope.allowed_document_ids for context in graph_contexts)
    assert result.scope_diagnostics["graph_scope_rejected_count"] == 2 * len(plan.entities)


def test_runner_without_verified_scope_is_rejected():
    runner = build_offline_runner()
    with pytest.raises(EvaluationScopeError):
        RAGEvaluationRunner(runner.agent, scope=None)


def test_manually_unverified_scope_is_rejected(document_ids):
    unsafe = EvaluationScope("scope-unit", frozenset({document_ids[0]}), scope_verified=False)
    with pytest.raises(EvaluationScopeError):
        RAGEvaluationRunner(build_offline_runner().agent, scope=unsafe)


@pytest.mark.asyncio
async def test_ordinary_qa_without_scope_remains_compatible():
    runner = build_offline_runner()
    result = await runner.agent.answer(_case_question := runner.cases[0].question)
    assert result.answer and runner.agent.knowledge_graph.calls > 0
