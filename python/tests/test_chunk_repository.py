"""In-memory, fake-only lifecycle tests for the Mongo chunk catalog."""

from __future__ import annotations

from copy import deepcopy

import pytest

from agents.doc_parser_agent import DocType, StructuredBlock
from services.chunk_models import build_parent_child_chunks
from services.chunk_repository import ChunkRepository


def _matches(row, query):
    return all(row.get(key) == value for key, value in query.items())


class Cursor(list):
    def sort(self, key, direction=1):
        if isinstance(key, list):
            for field, order in reversed(key):
                super().sort(key=lambda item: item.get(field), reverse=order < 0)
        else:
            super().sort(key=lambda item: item.get(key), reverse=direction < 0)
        return self


class Result:
    def __init__(self, count): self.modified_count = self.deleted_count = count


class Collection:
    def __init__(self): self.rows = []
    def create_index(self, *_args, **_kwargs): return "ok"
    def count_documents(self, query): return sum(_matches(row, query) for row in self.rows)
    def find(self, query, projection=None):
        rows = [deepcopy(row) for row in self.rows if _matches(row, query)]
        if projection == {"_id": 0}:
            for row in rows: row.pop("_id", None)
        return Cursor(rows)
    def find_one(self, query, projection=None):
        rows = self.find(query, projection)
        return rows[0] if rows else None
    def update_one(self, query, update, upsert=False):
        row = next((item for item in self.rows if _matches(item, query)), None)
        if row is None and upsert:
            row = dict(query); row["_id"] = len(self.rows) + 1; self.rows.append(row)
            row.update(update.get("$setOnInsert", {}))
        if row is not None: row.update(update.get("$set", {}))
        return Result(1 if row else 0)
    def update_many(self, query, update):
        selected = [row for row in self.rows if _matches(row, query)]
        for row in selected: row.update(update.get("$set", {}))
        return Result(len(selected))
    def delete_many(self, query):
        selected = [row for row in self.rows if _matches(row, query)]
        self.rows = [row for row in self.rows if not _matches(row, query)]
        return Result(len(selected))


class Database:
    def __init__(self): self.collection = Collection()
    def __getitem__(self, name): assert name == "document_chunks"; return self.collection


def bundle(document_id="doc-a", version=1):
    return build_parent_child_chunks(
        [StructuredBlock("一二三四五六七八九十" * 3, DocType.TEXT)], document_id=document_id,
        document_version=version, content_hash="a" * 64, source="safe.txt",
        parent_target_tokens=12, parent_max_tokens=18, child_target_tokens=5, child_overlap_tokens=1,
    )


@pytest.mark.asyncio
async def test_stage_activate_and_version_queries_are_exact():
    repository = ChunkRepository(Database()); repository.ensure_indexes()
    v1 = bundle(version=1)
    await repository.stage_version("doc-a", 1, v1.parents, v1.children)
    assert repository.count_version("doc-a", 1) == len(v1.parents) + len(v1.children)
    assert all(row["status"] == "processing" and not row["is_current"] for row in repository.collection.rows)
    await repository.activate_version("doc-a", 1)
    parent = repository.get_parent("doc-a", 1, v1.parents[0].parent_chunk_id)
    assert parent and parent["is_current"] and parent["status"] == "ready"
    assert len(repository.list_children("doc-a", 1, v1.parents[0].parent_chunk_id)) > 0
    assert repository.get_parent("doc-a", 2, v1.parents[0].parent_chunk_id) is None


@pytest.mark.asyncio
async def test_activation_deactivation_and_deletes_protect_other_versions_and_documents():
    repository = ChunkRepository(Database())
    v1, v2, other = bundle(version=1), bundle(version=2), bundle(document_id="doc-b", version=1)
    await repository.stage_version("doc-a", 1, v1.parents, v1.children); await repository.activate_version("doc-a", 1)
    await repository.stage_version("doc-a", 2, v2.parents, v2.children); await repository.activate_version("doc-a", 2)
    await repository.stage_version("doc-b", 1, other.parents, other.children); await repository.activate_version("doc-b", 1)
    assert not any(row["is_current"] for row in repository.collection.rows if row["document_id"] == "doc-a" and row["document_version"] == 1)
    await repository.delete_version("doc-a", 2)
    assert repository.count_version("doc-a", 1) == len(v1.parents) + len(v1.children)
    assert repository.count_version("doc-b", 1) == len(other.parents) + len(other.children)
    await repository.delete_document("doc-a")
    assert repository.count_version("doc-b", 1) == len(other.parents) + len(other.children)
