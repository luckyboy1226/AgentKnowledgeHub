"""Offline tests for Chroma document-version lifecycle behavior."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from agents.doc_parser_agent import DocType, DocumentChunk
from services.vector_store import VectorStoreService


class FakeEmbeddings:
    dimensions = 3

    async def aembed_documents(self, texts):
        return [[float(index), 0.0, 1.0] for index, _ in enumerate(texts)]

    async def aembed_query(self, query):
        return [1.0, 0.0, 0.0]


class FakeCollection:
    def __init__(self):
        self.rows = {}
        self.upsert_calls = 0
        self.update_calls = 0
        self.fail_upsert_after = None
        self.fail_update_once_at = None
        self.query_ids = None
        self.last_query_n_results = None

    def upsert(self, ids, embeddings, documents, metadatas):
        self.upsert_calls += 1
        if self.fail_upsert_after is not None and self.upsert_calls > self.fail_upsert_after:
            raise RuntimeError("simulated partial write")
        for vector_id, embedding, document, metadata in zip(ids, embeddings, documents, metadatas):
            self.rows[vector_id] = {
                "embedding": embedding,
                "document": document,
                "metadata": dict(metadata),
            }

    def update(self, ids, metadatas):
        self.update_calls += 1
        if self.fail_update_once_at == self.update_calls:
            raise RuntimeError("simulated activation failure")
        for vector_id, metadata in zip(ids, metadatas):
            self.rows[vector_id]["metadata"].update(metadata)

    def get(self, ids=None, where=None, include=None):
        selected = [vector_id for vector_id in (ids or self.rows) if vector_id in self.rows]
        if where:
            selected = [
                vector_id
                for vector_id in selected
                if all(self.rows[vector_id]["metadata"].get(key) == value for key, value in where.items())
            ]
        return {
            "ids": selected,
            "metadatas": [self.rows[vector_id]["metadata"] for vector_id in selected],
        }

    def delete(self, ids=None, where=None, **_):
        selected = list(ids or [])
        if where:
            selected = self.get(where=where)["ids"]
        for vector_id in selected:
            self.rows.pop(vector_id, None)

    def query(self, query_embeddings, n_results, include):
        del query_embeddings, include
        self.last_query_n_results = n_results
        selected = self.query_ids or list(self.rows)
        selected = selected[:n_results]
        return {
            "documents": [[self.rows[vector_id]["document"] for vector_id in selected]],
            "metadatas": [[self.rows[vector_id]["metadata"] for vector_id in selected]],
            "distances": [[index / 100 for index, _ in enumerate(selected)]],
        }

    def count(self):
        return len(self.rows)


@pytest.fixture
def vector_store():
    service = VectorStoreService(FakeEmbeddings())
    service._backend = "chroma"
    service._store = FakeCollection()
    return service


def chunks(source=Path(r"C:\private\uploads\policy.txt")):
    return [
        DocumentChunk("first", "legacy-doc", 0, DocType.TEXT, {"source": source}),
        DocumentChunk("second", "legacy-doc", 1, DocType.TEXT, {"source": source}),
    ]


async def stage_and_activate(vector_store, document_id="document-a", version=1):
    await vector_store.stage_document_version(document_id, version, chunks(), "hash-1")
    await vector_store.activate_document_version(document_id, version)


@pytest.mark.asyncio
async def test_versioned_vector_id_is_stable(vector_store):
    assert vector_store.vector_id("document-a", 2, 3) == "document-a:v2:c3"
    await vector_store.stage_document_version("document-a", 2, chunks(), "hash-2")
    assert vector_store.list_document_vector_ids("document-a", 2) == [
        "document-a:v2:c0",
        "document-a:v2:c1",
    ]


@pytest.mark.asyncio
async def test_repeated_stage_is_idempotent(vector_store):
    await vector_store.stage_document_version("document-a", 1, chunks(), "hash-1")
    await vector_store.stage_document_version("document-a", 1, chunks(), "hash-1")
    assert vector_store._store.count() == 2


def test_metadata_scalar_conversion_handles_path_uuid_and_datetime(vector_store):
    assert vector_store.scalar_metadata(Path("relative.txt")) == "relative.txt"
    assert vector_store.scalar_metadata(uuid4()).count("-") == 4
    assert vector_store.scalar_metadata(datetime(2026, 1, 1, tzinfo=timezone.utc)).startswith("2026")


def test_local_chroma_host_uses_ipv4_without_changing_remote_hosts(vector_store):
    assert vector_store.chroma_http_host("localhost") == "127.0.0.1"
    assert vector_store.chroma_http_host("chroma.internal") == "chroma.internal"


@pytest.mark.asyncio
async def test_stage_metadata_is_scalar_and_source_is_safe(vector_store):
    await vector_store.stage_document_version("document-a", 1, chunks(), "hash-1")
    metadata = vector_store._store.rows["document-a:v1:c0"]["metadata"]
    assert set((str, int, float, bool)).issuperset({type(value) for value in metadata.values()})
    assert metadata["source"] == "policy.txt" and ":\\" not in metadata["source"]


@pytest.mark.asyncio
async def test_stage_defaults_to_processing_and_not_current(vector_store):
    await vector_store.stage_document_version("document-a", 1, chunks(), "hash-1")
    assert all(not row["metadata"]["is_current"] for row in vector_store._store.rows.values())
    assert {row["metadata"]["status"] for row in vector_store._store.rows.values()} == {"processing"}


@pytest.mark.asyncio
async def test_activate_makes_staged_version_current_and_ready(vector_store):
    await stage_and_activate(vector_store)
    assert all(row["metadata"]["is_current"] for row in vector_store._store.rows.values())
    assert {row["metadata"]["status"] for row in vector_store._store.rows.values()} == {"ready"}


@pytest.mark.asyncio
async def test_activate_deactivates_old_version(vector_store):
    await stage_and_activate(vector_store, version=1)
    await vector_store.stage_document_version("document-a", 2, chunks(), "hash-2")
    await vector_store.activate_document_version("document-a", 2)
    assert not any(row["metadata"]["is_current"] for key, row in vector_store._store.rows.items() if ":v1:" in key)
    assert all(row["metadata"]["is_current"] for key, row in vector_store._store.rows.items() if ":v2:" in key)


@pytest.mark.asyncio
async def test_deactivate_removes_version_from_current_set(vector_store):
    await stage_and_activate(vector_store)
    assert await vector_store.deactivate_document_version("document-a", 1) == 2
    assert not any(row["metadata"]["is_current"] for row in vector_store._store.rows.values())


@pytest.mark.asyncio
async def test_delete_version_is_precise(vector_store):
    await stage_and_activate(vector_store, version=1)
    await vector_store.stage_document_version("document-a", 2, chunks(), "hash-2")
    assert await vector_store.delete_document_version("document-a", 2) == 2
    assert vector_store.count_document_version("document-a", 1) == 2


@pytest.mark.asyncio
async def test_delete_document_removes_all_its_versions_only(vector_store):
    await stage_and_activate(vector_store, "document-a", 1)
    await vector_store.stage_document_version("document-a", 2, chunks(), "hash-2")
    await stage_and_activate(vector_store, "document-b", 1)
    assert await vector_store.delete_document("document-a") == 4
    assert vector_store._store.count() == 2


@pytest.mark.asyncio
async def test_other_documents_are_untouched_by_version_delete(vector_store):
    await stage_and_activate(vector_store, "document-a", 1)
    await stage_and_activate(vector_store, "document-b", 1)
    await vector_store.delete_document_version("document-a", 1)
    assert vector_store.count_document_version("document-b", 1) == 2


@pytest.mark.asyncio
async def test_partial_stage_failure_compensates_only_written_vectors(vector_store):
    vector_store._store.fail_upsert_after = 1
    with pytest.raises(RuntimeError, match="partial write"):
        await vector_store.stage_document_version("document-a", 1, chunks(), "hash-1")
    assert vector_store._store.count() == 0


@pytest.mark.asyncio
async def test_activate_failure_restores_old_current_version(vector_store):
    await stage_and_activate(vector_store, version=1)
    await vector_store.stage_document_version("document-a", 2, chunks(), "hash-2")
    vector_store._store.fail_update_once_at = vector_store._store.update_calls + 2
    with pytest.raises(RuntimeError, match="activation failure"):
        await vector_store.activate_document_version("document-a", 2)
    old = [row for key, row in vector_store._store.rows.items() if ":v1:" in key]
    new = [row for key, row in vector_store._store.rows.items() if ":v2:" in key]
    assert all(row["metadata"]["is_current"] for row in old)
    assert not any(row["metadata"]["is_current"] for row in new)


@pytest.mark.asyncio
async def test_inactive_version_is_not_retrieved(vector_store):
    await stage_and_activate(vector_store, version=1)
    await vector_store.stage_document_version("document-a", 2, chunks(), "hash-2")
    vector_store._store.query_ids = ["document-a:v2:c0", "document-a:v1:c0"]
    found = await vector_store.search("question", top_k=2)
    assert [row[0]["metadata"]["document_version"] for row in found] == [1]


@pytest.mark.asyncio
async def test_failed_and_deleted_versions_are_not_retrieved(vector_store):
    await stage_and_activate(vector_store)
    for status in ("failed", "deleted"):
        for row in vector_store._store.rows.values():
            row["metadata"].update(status=status, is_current=False)
        assert await vector_store.search("question", top_k=2) == []
        for row in vector_store._store.rows.values():
            row["metadata"].update(status="ready", is_current=True)


@pytest.mark.asyncio
async def test_legacy_vectors_remain_retrievable(vector_store):
    vector_store._store.rows["legacy#0"] = {
        "document": "legacy content",
        "metadata": {"doc_id": "old", "source": r"C:\old\legacy.txt"},
        "embedding": [0.0, 0.0, 0.0],
    }
    found = await vector_store.search("question")
    assert found[0][0]["source"] == "legacy.txt"


@pytest.mark.asyncio
async def test_current_vector_is_retrievable_with_document_provenance(vector_store):
    await stage_and_activate(vector_store)
    found = await vector_store.search("question", top_k=1)
    assert found[0][0]["metadata"]["document_id"] == "document-a"
    assert found[0][0]["metadata"]["document_version"] == 1


@pytest.mark.asyncio
async def test_search_source_never_contains_an_absolute_path(vector_store):
    await stage_and_activate(vector_store)
    found = await vector_store.search("question", top_k=1)
    assert found[0][0]["source"] == "policy.txt"


@pytest.mark.asyncio
async def test_top_k_is_applied_after_filtering_with_bounded_overfetch(vector_store):
    await stage_and_activate(vector_store, "document-a", 1)
    await vector_store.stage_document_version("document-a", 2, chunks(), "hash-2")
    await stage_and_activate(vector_store, "document-b", 1)
    vector_store._store.query_ids = [
        "document-a:v2:c0",
        "document-a:v2:c1",
        "document-a:v1:c0",
        "document-b:v1:c0",
    ]
    found = await vector_store.search("question", top_k=2)
    assert len(found) == 2
    assert vector_store._store.last_query_n_results == 10 <= vector_store.SEARCH_MAX_CANDIDATES


@pytest.mark.asyncio
async def test_legacy_add_chunks_remains_compatible(vector_store):
    assert await vector_store.add_chunks(chunks()) == 2
    row = vector_store._store.rows["legacy-doc#chunk-0"]
    assert row["metadata"]["legacy"] is True and row["metadata"]["source"] == "policy.txt"


@pytest.mark.asyncio
async def test_empty_staged_version_cannot_be_activated(vector_store):
    with pytest.raises(ValueError, match="no staged vectors"):
        await vector_store.activate_document_version("document-a", 1)
