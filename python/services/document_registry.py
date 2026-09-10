"""Mongo-backed logical document and document-version registry.

This module intentionally owns only MongoDB metadata. Vector, graph and QA
storage are deliberately not touched until their version-aware updates are
implemented in a later S3 increment.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime, timezone
from typing import Any


VALID_STATUSES = {"processing", "ready", "failed", "deleting", "deleted"}
ALLOWED_TRANSITIONS = {
    "processing": {"ready", "failed"},
    "ready": {"processing", "deleting"},
    "failed": {"processing"},
    "deleting": {"deleted", "failed"},
    "deleted": set(),
}
_SECRET_PATTERNS = (
    r"mongodb(?:\+srv)?://[^\s]+",
    r"Bearer\s+\S+",
    r"(?:api[_-]?key|password|token)\s*[=:]\s*\S+",
)


class RegistryError(RuntimeError):
    """A safe, user-facing registry error."""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def safe_key(value: str) -> str:
    """Normalize a logical key or filename without retaining path components."""
    normalized = re.sub(r"[\\/\x00]", "_", value).strip()
    return normalized[:255] or "unnamed"


def safe_error(error: Exception | str) -> str:
    """Keep a short diagnostic while redacting common secret-bearing fragments."""
    message = str(error)
    for pattern in _SECRET_PATTERNS:
        message = re.sub(pattern, "[redacted]", message, flags=re.IGNORECASE)
    return message[:300]


class DocumentRegistry:
    """Repository for stable logical documents and their immutable versions."""

    def __init__(self, database: Any):
        self.documents = database["documents"]
        self.versions = database["document_versions"]
        # The coordinator receives these collection handles through this
        # repository; it never creates an independent Mongo client.
        self.operations = database["document_operations"]
        self.operation_locks = database["document_operation_locks"]

    def ensure_indexes(self) -> None:
        """Install only the two uniqueness guarantees required by this phase."""
        try:
            self.documents.create_index(
                [("namespace", 1), ("logical_key", 1)], unique=True
            )
            self.versions.create_index(
                [("document_id", 1), ("version", 1)], unique=True
            )
            self.operations.create_index([("operation_id", 1)], unique=True)
            self.operations.create_index([("status", 1), ("updated_at", 1)])
            self.operation_locks.create_index([("document_id", 1)], unique=True)
            self.operation_locks.create_index([("expires_at", 1)])
        except Exception as exc:  # pragma: no cover - driver-specific error shape
            raise RegistryError("Document registry indexes are unavailable") from exc

    def find(self, document_id: str) -> dict[str, Any] | None:
        return self.documents.find_one({"document_id": document_id}, {"_id": 0})

    def find_by_key(self, namespace: str, logical_key: str) -> dict[str, Any] | None:
        return self.documents.find_one(
            {"namespace": safe_key(namespace), "logical_key": safe_key(logical_key)},
            {"_id": 0},
        )

    def list_documents(self, namespace: str | None = None) -> list[dict[str, Any]]:
        """Return registry summaries without consulting the legacy upload directory."""
        query: dict[str, Any] = {}
        if namespace is not None:
            query["namespace"] = safe_key(namespace)
        return list(self.documents.find(query, {"_id": 0}).sort("updated_at", -1))

    def versions_for(self, document_id: str) -> list[dict[str, Any]]:
        return list(
            self.versions.find({"document_id": document_id}, {"_id": 0}).sort(
                "version", 1
            )
        )

    def reserve(
        self,
        filename: str,
        content: bytes,
        logical_key: str | None = None,
        namespace: str = "default",
    ) -> tuple[dict[str, Any], dict[str, Any], bool]:
        """Reserve version one or the next immutable version.

        The returned ``changed`` flag is false for an existing content hash;
        this makes a retry idempotent even while that version is processing or
        failed. Retrying a failed version is an explicit later API operation.
        """
        filename = safe_key(filename)
        namespace = safe_key(namespace)
        logical_key = safe_key(logical_key or filename)
        content_hash = hashlib.sha256(content).hexdigest()
        now = utcnow()

        try:
            document = self.find_by_key(namespace, logical_key)
            if document is None:
                document_id = str(uuid.uuid4())
                document = {
                    "document_id": document_id,
                    "namespace": namespace,
                    "logical_key": logical_key,
                    "filename": filename,
                    "current_version": None,
                    "status": "processing",
                    "created_at": now,
                    "updated_at": now,
                    "deleted_at": None,
                }
                self.documents.insert_one(document)
                version = 1
            else:
                document_id = document["document_id"]
                existing = self.versions.find_one(
                    {"document_id": document_id, "content_hash": content_hash},
                    {"_id": 0},
                )
                if existing is not None:
                    return document, existing, False
                latest = self.versions.find_one(
                    {"document_id": document_id}, sort=[("version", -1)]
                )
                version = int(latest["version"]) + 1 if latest else 1
                self.documents.update_one(
                    {"document_id": document_id},
                    {"$set": {"status": "processing", "updated_at": now, "filename": filename}},
                )

            record = {
                "document_id": document_id,
                "version": version,
                "content_hash": content_hash,
                "filename": filename,
                "status": "processing",
                "is_current": False,
                "chunk_count": 0,
                "error_summary": None,
                "created_at": now,
                "updated_at": now,
                "ready_at": None,
            }
            self.versions.insert_one(record)
            reserved = self.find(document_id)
            if reserved is None:
                raise RegistryError("Document registry reservation was not persisted")
            return reserved, record, True
        except RegistryError:
            raise
        except Exception as exc:
            raise RegistryError("Document version reservation failed") from exc

    def create_next_version(
        self, document_id: str, filename: str, content: bytes
    ) -> tuple[dict[str, Any], dict[str, Any], bool]:
        """Create the next version for an existing logical document."""
        document = self.find(document_id)
        if document is None:
            raise RegistryError("Document not found")
        return self.reserve(
            filename=filename,
            content=content,
            logical_key=document["logical_key"],
            namespace=document["namespace"],
        )

    def transition(
        self,
        document_id: str,
        version: int,
        target: str,
        *,
        chunk_count: int = 0,
        error: Exception | str | None = None,
    ) -> dict[str, Any]:
        """Move one version through an allowed state transition."""
        try:
            row = self.versions.find_one({"document_id": document_id, "version": version})
            if row is None:
                raise RegistryError("Document version not found")
            if target not in VALID_STATUSES or target not in ALLOWED_TRANSITIONS[row["status"]]:
                raise RegistryError("Invalid document version transition")

            now = utcnow()
            version_update: dict[str, Any] = {"status": target, "updated_at": now}
            document_update: dict[str, Any] = {"status": target, "updated_at": now}

            if target == "ready":
                self.versions.update_many(
                    {"document_id": document_id, "is_current": True},
                    {"$set": {"is_current": False}},
                )
                version_update.update(
                    is_current=True,
                    chunk_count=chunk_count,
                    ready_at=now,
                    error_summary=None,
                )
                document_update["current_version"] = version
            elif target == "failed":
                version_update["error_summary"] = safe_error(error or "processing failed")
                document = self.find(document_id)
                if document and document.get("current_version") is not None:
                    # A failed replacement must not hide the prior ready version.
                    document_update["status"] = "ready"
            elif target == "deleted":
                document_update["deleted_at"] = now

            self.versions.update_one(
                {"document_id": document_id, "version": version}, {"$set": version_update}
            )
            self.documents.update_one({"document_id": document_id}, {"$set": document_update})
            updated = self.versions.find_one(
                {"document_id": document_id, "version": version}, {"_id": 0}
            )
            if updated is None:
                raise RegistryError("Document version transition was not persisted")
            return updated
        except RegistryError:
            raise
        except Exception as exc:
            raise RegistryError("Document version transition failed") from exc

    def mark_deleting(self, document_id: str, version: int | None = None) -> dict[str, Any]:
        document = self.find(document_id)
        if document is None:
            raise RegistryError("Document not found")
        target_version = version if version is not None else document.get("current_version")
        if target_version is None:
            raise RegistryError("Document has no ready version to delete")
        return self.transition(document_id, int(target_version), "deleting")

    def mark_deleted(self, document_id: str, version: int) -> dict[str, Any]:
        return self.transition(document_id, version, "deleted")
