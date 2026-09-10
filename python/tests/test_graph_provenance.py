"""Offline Neo4j provenance contracts using a recording async driver."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from agents.knowledge_extract_agent import Entity, Relation
from agents.qa_agent import QAAgent
from services.graph_rag import GraphRAGPipeline
from services.knowledge_graph import KnowledgeGraphService


class FakeResult:
    def __init__(self, records=None):
        self.records = records or []

    async def data(self):
        return self.records


class FakeTransaction:
    def __init__(self, driver):
        self.driver = driver
        self.calls = []

    async def run(self, query, *args, **kwargs):
        self.calls.append((query, args, kwargs))
        self.driver.calls.append((query, args, kwargs))
        if self.driver.fail_at and len(self.calls) == self.driver.fail_at:
            raise RuntimeError("simulated graph write failure")
        if "collect(DISTINCT e.name)" in query:
            return FakeResult([{"entity_names": self.driver.entity_names}])
        return FakeResult()


class FakeSession:
    def __init__(self, driver):
        self.driver = driver

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def run(self, query, params=None, **kwargs):
        call = (query, (params,) if params is not None else (), kwargs)
        self.driver.calls.append(call)
        if "collect(DISTINCT e.name)" in query:
            return FakeResult([{"entity_names": self.driver.entity_names}])
        if "count(r) AS cnt" in query:
            return FakeResult([{"cnt": 2}])
        return FakeResult(self.driver.read_records)

    async def execute_write(self, operation, *args):
        transaction = FakeTransaction(self.driver)
        try:
            result = await operation(transaction, *args)
            self.driver.committed += 1
            return result
        except Exception:
            self.driver.rolled_back += 1
            raise


class FakeDriver:
    def __init__(self):
        self.calls = []
        self.fail_at = None
        self.committed = 0
        self.rolled_back = 0
        self.entity_names = ["Alice", "Bob"]
        self.read_records = []

    def session(self):
        return FakeSession(self)


@pytest.fixture
def graph():
    service = KnowledgeGraphService()
    service._driver = FakeDriver()
    return service


def entities():
    return [Entity(name="Alice", type="Person", description="Owner")]


def relations():
    return [Relation(head="Alice", relation="owns", tail="Project", confidence=0.9)]


def queries(graph):
    return "\n".join(query for query, _, _ in graph._driver.calls)


def params(graph):
    return [args[0] for _, args, _ in graph._driver.calls if args and isinstance(args[0], dict)] + [
        kwargs for _, _, kwargs in graph._driver.calls if kwargs
    ]


def test_document_version_key_is_stable():
    assert KnowledgeGraphService.document_version_key("doc-a", 2) == "doc-a:v2"


def test_evidence_key_is_stable_and_version_specific():
    first = KnowledgeGraphService.evidence_key("doc-a", 1, "Alice", "owns", "Project")
    assert first == KnowledgeGraphService.evidence_key("doc-a", 1, "Alice", "owns", "Project")
    assert first != KnowledgeGraphService.evidence_key("doc-a", 2, "Alice", "owns", "Project")


@pytest.mark.asyncio
async def test_stage_creates_processing_inactive_document_version(graph):
    result = await graph.stage_document_version("doc-a", 1, "hash", r"C:\uploads\a.txt", entities(), relations())
    assert result == {"mentions": 2, "evidence": 1}
    assert "status = 'processing'" in queries(graph) and "is_current = false" in queries(graph)


@pytest.mark.asyncio
async def test_repeated_stage_uses_merge_for_idempotence(graph):
    await graph.stage_document_version("doc-a", 1, "hash", "a.txt", entities(), relations())
    await graph.stage_document_version("doc-a", 1, "hash", "a.txt", entities(), relations())
    assert queries(graph).count("MERGE (dv:DocumentVersion {key: $key})") == 2
    assert queries(graph).count("MERGE (dv)-[:MENTIONS]->(e)") >= 2


@pytest.mark.asyncio
async def test_mentions_are_merged_not_duplicated(graph):
    await graph.stage_document_version("doc-a", 1, "hash", "a.txt", entities(), relations())
    assert "MERGE (dv)-[:MENTIONS]->(e)" in queries(graph)


@pytest.mark.asyncio
async def test_evidence_relationship_is_merged_by_stable_key(graph):
    await graph.stage_document_version("doc-a", 1, "hash", "a.txt", entities(), relations())
    assert "MERGE (h)-[r:OWNS {evidence_key: $evidence_key}]->(t)" in queries(graph)


@pytest.mark.asyncio
async def test_evidence_properties_include_document_and_version(graph):
    await graph.stage_document_version("doc-a", 2, "hash", "a.txt", entities(), relations())
    stage_params = params(graph)
    assert any(item.get("document_id") == "doc-a" and item.get("version") == 2 for item in stage_params)


def test_dynamic_relationship_type_is_validated():
    assert KnowledgeGraphService.safe_relationship_type("owns project") == "OWNS_PROJECT"
    assert KnowledgeGraphService.safe_relationship_type("x`]->[:PWNED") == "RELATED_TO"
    assert KnowledgeGraphService.safe_relationship_type("中文关系") == "RELATED_TO"


@pytest.mark.asyncio
async def test_relation_values_are_parameterized_not_interpolated(graph):
    dangerous = Relation(head="Alice'; DELETE n", relation="owns", tail="Project", confidence=0.9)
    await graph.stage_document_version("doc-a", 1, "hash", "a.txt", [], [dangerous])
    assert "Alice'; DELETE n" not in queries(graph)
    assert any(item.get("head") == "Alice'; DELETE n" for item in params(graph))


@pytest.mark.asyncio
async def test_activate_promotes_new_version(graph):
    await graph.activate_document_version("doc-a", 2)
    assert "target.is_current = true, target.status = 'ready'" in queries(graph)
    assert "current.is_current = true, current.status = 'ready'" in queries(graph)


@pytest.mark.asyncio
async def test_activate_deactivates_old_document_and_evidence(graph):
    await graph.activate_document_version("doc-a", 2)
    text = queries(graph)
    assert "SET old.is_current = false" in text
    assert "old.document_version <> $version" in text


@pytest.mark.asyncio
async def test_activate_failure_rolls_back_single_transaction(graph):
    graph._driver.fail_at = 2
    with pytest.raises(RuntimeError, match="graph write failure"):
        await graph.activate_document_version("doc-a", 2)
    assert graph._driver.rolled_back == 1 and graph._driver.committed == 0


@pytest.mark.asyncio
async def test_deactivate_targets_exact_document_version(graph):
    await graph.deactivate_document_version("doc-a", 3)
    assert all(item.get("document_id") == "doc-a" and item.get("version") == 3 for item in params(graph))


@pytest.mark.asyncio
async def test_rollback_deletes_only_target_version(graph):
    await graph.delete_document_version("doc-a", 2)
    text = queries(graph)
    assert "r.document_id = $document_id AND r.document_version = $version" in text
    assert "DETACH DELETE" not in text


@pytest.mark.asyncio
async def test_delete_document_targets_only_one_document(graph):
    await graph.delete_document("doc-a")
    assert all(item.get("document_id") == "doc-a" for item in params(graph) if "document_id" in item)


@pytest.mark.asyncio
async def test_other_document_provenance_is_not_targeted(graph):
    await graph.delete_document_version("doc-a", 1)
    assert "document_version = $version" in queries(graph)
    assert not any(item.get("document_id") == "doc-b" for item in params(graph))


@pytest.mark.asyncio
async def test_shared_entities_are_protected_by_remaining_mentions(graph):
    await graph.delete_document_version("doc-a", 1)
    assert "NOT EXISTS { MATCH (:DocumentVersion)-[:MENTIONS]->(e) }" in queries(graph)


@pytest.mark.asyncio
async def test_entities_with_business_relationships_are_protected(graph):
    await graph.delete_document_version("doc-a", 1)
    assert "AND NOT (e)--()" in queries(graph)


@pytest.mark.asyncio
async def test_orphan_cleanup_is_limited_to_target_mentions(graph):
    await graph.delete_document_version("doc-a", 1)
    assert "e.name IN $entity_names" in queries(graph)


@pytest.mark.asyncio
async def test_count_document_evidence_uses_exact_identity(graph):
    assert await graph.count_document_evidence("doc-a", 2) == 2
    assert any(item.get("document_id") == "doc-a" and item.get("version") == 2 for item in params(graph))


@pytest.mark.asyncio
async def test_list_evidence_returns_safe_provenance_shape(graph):
    graph._driver.read_records = [{"evidence_key": "k", "source": "a.txt", "document_id": "doc-a"}]
    records = await graph.list_document_evidence("doc-a", 1)
    assert records[0]["evidence_key"] == "k"
    assert "r.evidence_key AS evidence_key" in queries(graph)


@pytest.mark.asyncio
async def test_current_neighbor_query_filters_processing_failed_deleted(graph):
    await graph.get_neighbors("Alice")
    text = queries(graph)
    assert "rel.is_current = true AND rel.status = 'ready'" in text
    assert "type(rel) <> 'MENTIONS'" in text


@pytest.mark.asyncio
async def test_neighbor_query_keeps_legacy_relationships(graph):
    await graph.get_neighbors("Alice")
    assert "rel.document_id IS NULL OR" in queries(graph)


@pytest.mark.asyncio
async def test_inactive_new_relationship_cannot_use_legacy_compatibility(graph):
    filter_text = graph._current_or_legacy_relationship_filter("rel")
    assert "rel.document_id IS NULL" in filter_text
    assert "rel.is_current = true AND rel.status = 'ready'" in filter_text


@pytest.mark.asyncio
async def test_current_paths_apply_same_version_filter(graph):
    await graph.get_current_paths("Alice", "Project")
    text = queries(graph)
    assert "ALL(rel IN rels WHERE" in text and "rel.document_id IS NULL OR" in text


@pytest.mark.asyncio
async def test_graph_rag_subgraph_context_has_safe_source_and_version(graph):
    graph._driver.read_records = [{
        "source": "Alice", "relations": ["OWNS"], "target": "Project", "target_type": "Concept",
        "target_desc": "Work", "document_id": "doc-a", "document_version": 1,
        "provenance_source": r"C:\uploads\a.txt",
    }]
    pipeline = GraphRAGPipeline(vector_store=None, knowledge_graph=graph, chat_provider=None)
    contexts = await pipeline._subgraph_search(["Alice"])
    assert contexts[0].metadata["document_id"] == "doc-a"
    assert contexts[0].metadata["document_version"] == 1
    assert contexts[0].metadata["source"] == "a.txt"


@pytest.mark.asyncio
async def test_graph_rag_merges_duplicate_fact_contexts(graph):
    graph._driver.read_records = [
        {"source": "Alice", "relations": ["OWNS"], "target": "Project", "target_type": "Concept", "target_desc": "Work", "document_id": "doc-a", "document_version": 1, "provenance_source": "a.txt"},
        {"source": "Alice", "relations": ["OWNS"], "target": "Project", "target_type": "Concept", "target_desc": "Work", "document_id": "doc-b", "document_version": 1, "provenance_source": "b.txt"},
    ]
    pipeline = GraphRAGPipeline(vector_store=None, knowledge_graph=graph, chat_provider=None)
    contexts = await pipeline._subgraph_search(["Alice"])
    assert len(contexts) == 1 and len(contexts[0].metadata["sources"]) == 2


@pytest.mark.asyncio
async def test_qa_graph_retrieval_uses_provenance_aware_neighbors(graph):
    graph._driver.read_records = [{
        "source": "Alice", "relations": ["OWNS"], "target": "Project", "target_type": "Concept",
        "target_desc": "Work", "document_id": "doc-a", "document_version": 1,
        "provenance_source": r"C:\uploads\a.txt",
    }]
    agent = QAAgent(chat_provider=None, knowledge_graph=graph)
    contexts = await agent._graph_retrieve("unused", {"entities": ["Alice"]})
    assert contexts[0].metadata["document_id"] == "doc-a"
    assert contexts[0].metadata["document_version"] == 1
    assert contexts[0].source == "a.txt"
    assert "rel.is_current = true AND rel.status = 'ready'" in queries(graph)


@pytest.mark.asyncio
async def test_qa_graph_retrieval_does_not_run_model_generated_cypher(graph):
    agent = QAAgent(chat_provider=None, knowledge_graph=graph)
    await agent._graph_retrieve("unused", {"entities": ["Alice"]})
    assert all("shortestPath" not in query for query, _, _ in graph._driver.calls)


@pytest.mark.asyncio
async def test_stage_failure_uses_transaction_rollback(graph):
    graph._driver.fail_at = 3
    with pytest.raises(RuntimeError, match="graph write failure"):
        await graph.stage_document_version("doc-a", 1, "hash", "a.txt", entities(), relations())
    assert graph._driver.rolled_back == 1 and graph._driver.committed == 0


@pytest.mark.asyncio
async def test_constraints_are_additive_and_keep_entity_schema(graph):
    await graph._ensure_indexes()
    text = queries(graph)
    assert "CREATE CONSTRAINT document_version_key IF NOT EXISTS" in text
    assert "DROP" not in text and "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.name)" in text


@pytest.mark.asyncio
async def test_legacy_add_relation_uses_safe_type_and_safe_source(graph):
    await graph.add_relation(Relation("Alice", "bad` type", "Project", 1.0), r"C:\uploads\a.txt")
    assert "RELATED_TO" in queries(graph)
    assert any(item.get("source") == "a.txt" for item in params(graph))
