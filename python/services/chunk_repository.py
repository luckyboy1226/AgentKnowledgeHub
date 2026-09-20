"""Mongo catalog for Parent–Child chunk lifecycle records.

Only the document coordinator calls this repository. It never creates a Mongo
client, and every mutation is constrained by a document ID plus version.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from services.chunk_models import ChildChunk, ParentChunk
from services.document_registry import RegistryError


def _now() -> datetime:
    return datetime.now(UTC)


def _safe_source(value: object) -> str:
    return str(value or "").replace("\\", "/").rsplit("/", 1)[-1][:180]


class ChunkRepository:
    """Version-aware Parent/Child catalog backed by ``document_chunks``."""

    def __init__(self, database: Any) -> None:
        self.collection = database["document_chunks"]

    def ensure_indexes(self) -> None:
        try:
            self.collection.create_index(
                [("document_id", 1), ("document_version", 1), ("kind", 1), ("chunk_id", 1)], unique=True
            )
            self.collection.create_index(
                [("document_id", 1), ("document_version", 1), ("parent_chunk_id", 1), ("kind", 1)]
            )
            self.collection.create_index(
                [("document_id", 1), ("is_current", 1), ("status", 1), ("kind", 1)]
            )
        except Exception as exc:  # pragma: no cover - driver-specific failures
            raise RegistryError("Chunk catalog indexes are unavailable") from exc

    @staticmethod
    def _record(kind: str, chunk: ParentChunk | ChildChunk) -> dict[str, Any]:
        now = _now()
        record = {
            "kind": kind,
            "document_id": str(chunk.document_id),
            "document_version": int(chunk.document_version),
            "parent_chunk_id": str(chunk.parent_chunk_id),
            "chunk_id": str(chunk.parent_chunk_id if kind == "parent" else chunk.chunk_id),
            "content": str(chunk.content),
            "section_title": str(chunk.section_title) if chunk.section_title else None,
            "page_number": int(chunk.page_number) if chunk.page_number is not None else None,
            "table_id": str(chunk.table_id) if chunk.table_id else None,
            "estimated_token_count": int(chunk.estimated_token_count),
            "content_hash": str(chunk.content_hash),
            "status": "processing",
            "is_current": False,
            "source": _safe_source(chunk.source),
            "created_at": now,
            "updated_at": now,
            "metadata": dict(chunk.metadata),
        }
        if kind == "parent":
            record["parent_index"] = int(chunk.parent_index)
        else:
            assert isinstance(chunk, ChildChunk)
            record["child_chunk_id"] = str(chunk.chunk_id)
            record["chunk_index"] = int(chunk.chunk_index)
            record["child_index_in_parent"] = int(chunk.child_index_in_parent)
        return record

    async def stage_version(
        self, document_id: str, version: int, parents: list[ParentChunk], children: list[ChildChunk]
    ) -> int:
        if not parents or not children:
            raise ValueError("Parent–Child catalog requires parent and child records")
        if any(chunk.document_id != document_id or int(chunk.document_version) != int(version) for chunk in [*parents, *children]):
            raise ValueError("Chunk catalog identity does not match the staged version")
        try:
            for parent in parents:
                record = self._record("parent", parent)
                mutable = {**record, "updated_at": _now()}
                mutable.pop("created_at", None)
                self.collection.update_one(
                    {"document_id": document_id, "document_version": int(version), "kind": "parent", "chunk_id": parent.parent_chunk_id},
                    {"$set": mutable, "$setOnInsert": {"created_at": record["created_at"]}},
                    upsert=True,
                )
            for child in children:
                record = self._record("child", child)
                mutable = {**record, "updated_at": _now()}
                mutable.pop("created_at", None)
                self.collection.update_one(
                    {"document_id": document_id, "document_version": int(version), "kind": "child", "chunk_id": child.chunk_id},
                    {"$set": mutable, "$setOnInsert": {"created_at": record["created_at"]}},
                    upsert=True,
                )
            return len(parents) + len(children)
        except Exception as exc:
            raise RegistryError("Chunk catalog staging failed") from exc

    def count_version(self, document_id: str, version: int) -> int:
        return int(self.collection.count_documents({"document_id": str(document_id), "document_version": int(version)}))

    async def activate_version(self, document_id: str, version: int) -> int:
        document_id, version = str(document_id), int(version)
        rows = list(self.collection.find({"document_id": document_id, "document_version": version}, {"_id": 0}))
        if not rows:
            raise ValueError("Cannot activate a version without catalog chunks")
        old_rows = list(self.collection.find({"document_id": document_id, "is_current": True}))
        now = _now()
        try:
            self.collection.update_many({"document_id": document_id, "is_current": True}, {"$set": {"is_current": False, "updated_at": now}})
            self.collection.update_many(
                {"document_id": document_id, "document_version": version},
                {"$set": {"is_current": True, "status": "ready", "updated_at": now}},
            )
            return len(rows)
        except Exception:
            # Exact restoration mirrors the vector/graph activation contract.
            self.collection.update_many({"document_id": document_id, "document_version": version}, {"$set": {"is_current": False, "status": "processing", "updated_at": _now()}})
            for row in old_rows:
                self.collection.update_one({"_id": row.get("_id")}, {"$set": {"is_current": True, "status": row.get("status", "ready"), "updated_at": _now()}})
            raise

    async def deactivate_version(self, document_id: str, version: int) -> int:
        result = self.collection.update_many(
            {"document_id": str(document_id), "document_version": int(version)},
            {"$set": {"is_current": False, "updated_at": _now()}},
        )
        return int(getattr(result, "modified_count", 0))

    async def delete_version(self, document_id: str, version: int) -> int:
        result = self.collection.delete_many({"document_id": str(document_id), "document_version": int(version)})
        return int(getattr(result, "deleted_count", 0))

    async def delete_document(self, document_id: str) -> int:
        result = self.collection.delete_many({"document_id": str(document_id)})
        return int(getattr(result, "deleted_count", 0))

    def get_parent(self, document_id: str, version: int, parent_chunk_id: str) -> dict[str, Any] | None:
        return self.collection.find_one(
            {"document_id": str(document_id), "document_version": int(version), "kind": "parent", "parent_chunk_id": str(parent_chunk_id)},
            {"_id": 0},
        )

    def list_children(self, document_id: str, version: int, parent_chunk_id: str) -> list[dict[str, Any]]:
        return list(self.collection.find(
            {"document_id": str(document_id), "document_version": int(version), "kind": "child", "parent_chunk_id": str(parent_chunk_id)},
            {"_id": 0},
        ).sort("chunk_index", 1))

    def list_current_children(self, document_id: str | None = None) -> list[dict[str, Any]]:
        query: dict[str, Any] = {"kind": "child", "status": "ready", "is_current": True}
        if document_id is not None:
            query["document_id"] = str(document_id)
        return list(self.collection.find(query, {"_id": 0}).sort([("document_id", 1), ("chunk_index", 1)]))
