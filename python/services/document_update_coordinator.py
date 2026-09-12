"""Injected, recoverable S3 Saga coordinator; it is not wired to HTTP ingestion."""

from __future__ import annotations

import uuid
import re
import hashlib
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Protocol

from agents.doc_parser_agent import DocumentChunk
from agents.knowledge_extract_agent import Entity, Relation
from services.document_registry import RegistryError, safe_error, safe_key, utcnow
from services.processing_errors import classify_safe_processing_error


ACTIVE_OPERATION_STATUSES = {"pending", "running", "compensating"}
CONTENT_HASH_RE = re.compile(r"^[a-f0-9]{64}$", re.IGNORECASE)
INCOMPLETE_OPERATION_STATUSES = {
    "running",
    "compensating",
    "cleanup_pending",
    "needs_reconciliation",
}


class DocumentProcessor(Protocol):
    async def prepare(
        self,
        *,
        content: bytes,
        filename: str,
        document_id: str,
        version: int,
        content_hash: str,
        operation_id: str,
    ) -> "PreparedDocument": ...


@dataclass(frozen=True)
class PreparedDocument:
    """Prepared values only live in memory and are deliberately never journaled."""

    chunks: list[DocumentChunk] = field(default_factory=list)
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    document_id: str = ""
    document_version: int = 0
    content_hash: str = ""
    source: str = ""
    processing_metadata: dict[str, Any] = field(default_factory=dict)


class OperationBusyError(RuntimeError):
    pass


class OperationIdentityConflictError(RegistryError):
    """A caller reused an operation ID for a different durable request."""

    pass


class OperationJournal:
    """Mongo collection adapter for durable Saga state and document-scoped leases."""

    def __init__(self, operations: Any, locks: Any):
        self.operations = operations
        self.locks = locks

    def create(self, record: dict[str, Any]) -> dict[str, Any]:
        existing = self.get(record["operation_id"])
        if existing:
            return existing
        self.operations.insert_one(record)
        return record

    def get(self, operation_id: str) -> dict[str, Any] | None:
        return self.operations.find_one({"operation_id": operation_id}, {"_id": 0})

    def update(self, operation_id: str, **fields: Any) -> dict[str, Any]:
        fields["updated_at"] = utcnow()
        self.operations.update_one({"operation_id": operation_id}, {"$set": fields})
        operation = self.get(operation_id)
        if operation is None:
            raise RegistryError("Operation journal record not found")
        return operation

    def complete_step(self, operation_id: str, step: str, **fields: Any) -> dict[str, Any]:
        operation = self.get(operation_id)
        if operation is None:
            raise RegistryError("Operation journal record not found")
        steps = list(operation.get("completed_steps", []))
        if step not in steps:
            steps.append(step)
        return self.update(operation_id, completed_steps=steps, **fields)

    def complete_compensation(self, operation_id: str, step: str) -> dict[str, Any]:
        operation = self.get(operation_id)
        if operation is None:
            raise RegistryError("Operation journal record not found")
        steps = list(operation.get("compensation_steps", []))
        if step not in steps:
            steps.append(step)
        return self.update(operation_id, compensation_steps=steps)

    def acquire_lock(self, document_id: str, operation_id: str, lease_seconds: int = 300) -> bool:
        """Acquire/renew a durable document lock; stale locks can be taken over."""
        now = utcnow()
        expires_at = now + timedelta(seconds=lease_seconds)
        existing = self.locks.find_one({"document_id": document_id})
        if existing and existing.get("operation_id") not in {operation_id, None}:
            if existing.get("expires_at") and existing["expires_at"] > now:
                return False
            self.locks.update_one(
                {"document_id": document_id, "operation_id": existing["operation_id"]},
                {"$set": {"operation_id": operation_id, "expires_at": expires_at, "updated_at": now}},
            )
            return True
        if existing:
            self.locks.update_one(
                {"document_id": document_id, "operation_id": operation_id},
                {"$set": {"expires_at": expires_at, "updated_at": now}},
            )
            return True
        try:
            self.locks.insert_one(
                {
                    "document_id": document_id,
                    "operation_id": operation_id,
                    "expires_at": expires_at,
                    "created_at": now,
                    "updated_at": now,
                }
            )
            return True
        except Exception:
            # A concurrent unique insert won. Read the durable owner rather
            # than assuming this process owns the lock.
            owner = self.locks.find_one({"document_id": document_id})
            return bool(owner and owner.get("operation_id") == operation_id)

    def release_lock(self, document_id: str, operation_id: str) -> bool:
        result = self.locks.delete_one({"document_id": document_id, "operation_id": operation_id})
        return bool(getattr(result, "deleted_count", result if isinstance(result, int) else 0))

    def incomplete(self) -> list[dict[str, Any]]:
        return list(self.operations.find({"status": {"$in": list(INCOMPLETE_OPERATION_STATUSES)}}, {"_id": 0}))


class DocumentUpdateCoordinator:
    """S3 Saga coordinator, intentionally isolated from current upload routes."""

    def __init__(
        self,
        registry: Any,
        vector_store: Any,
        knowledge_graph: Any,
        processor: DocumentProcessor,
        journal: OperationJournal | Any | None = None,
        *,
        lease_seconds: int = 300,
    ) -> None:
        self.registry = registry
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.processor = processor
        self.journal = journal or OperationJournal(registry.operations, registry.operation_locks)
        self.lease_seconds = lease_seconds

    @staticmethod
    def _operation_record(
        operation_id: str,
        operation_type: str,
        document_id: str,
        version: int | None,
        previous_current_version: int | None,
        content_hash: str | None = None,
        source: str | None = None,
        namespace: str | None = None,
        logical_key: str | None = None,
    ) -> dict[str, Any]:
        now = utcnow()
        return {
            "operation_id": operation_id,
            "operation_type": operation_type,
            "document_id": document_id,
            "version": version,
            "previous_current_version": previous_current_version,
            # Only a canonical SHA-256 may enter durable journal state. This
            # prevents a faulty caller from turning this field into document text.
            "content_hash": content_hash if content_hash and CONTENT_HASH_RE.fullmatch(content_hash) else None,
            "source": source,
            "namespace": safe_key(namespace) if namespace is not None else None,
            "logical_key": safe_key(logical_key) if logical_key is not None else None,
            "status": "running",
            "completed_steps": ["mongo_reserved"],
            "compensation_steps": [],
            "vector_ids": [],
            "graph_evidence_keys": [],
            "error_summary": None,
            "error_phase": None,
            "error_category": None,
            "error_type": None,
            "chunk_index": None,
            "created_at": now,
            "updated_at": now,
        }

    @staticmethod
    def _content_hash(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def _assert_same_operation(
        existing: dict[str, Any],
        *,
        operation_type: str,
        document_id: str | None = None,
        content_hash: str | None = None,
        namespace: str | None = None,
        logical_key: str | None = None,
    ) -> None:
        """Fail closed when an operation UUID is reused for another request.

        The journal is the durable idempotency boundary.  Returning an existing
        operation is safe only when its immutable request identity matches; a
        UUID must never silently become an alias for another upload.
        """
        expected = {
            "operation_type": operation_type,
            "document_id": document_id,
            "content_hash": content_hash,
            "namespace": safe_key(namespace) if namespace is not None else None,
            "logical_key": safe_key(logical_key) if logical_key is not None else None,
        }
        for key, value in expected.items():
            if value is not None and existing.get(key) != value:
                raise OperationIdentityConflictError("Operation ID is already bound to another request")

    async def create_document_version(
        self,
        *,
        filename: str,
        content: bytes,
        logical_key: str | None = None,
        namespace: str = "default",
        operation_id: str | None = None,
    ) -> dict[str, Any]:
        content_hash = self._content_hash(content)
        normalized_namespace = safe_key(namespace)
        normalized_logical_key = safe_key(logical_key or filename)
        if operation_id:
            existing = self.journal.get(operation_id)
            if existing:
                self._assert_same_operation(
                    existing,
                    operation_type="create",
                    content_hash=content_hash,
                    namespace=normalized_namespace,
                    logical_key=normalized_logical_key,
                )
                return {**existing, "changed": True}
        document, version_record, changed = self.registry.reserve(
            filename, content, logical_key, namespace
        )
        if not changed:
            return self._record_unchanged(
                operation_id, "create", document, version_record, filename,
                namespace=normalized_namespace, logical_key=normalized_logical_key,
            )
        return await self._run_reserved_version(
            document=document,
            version_record=version_record,
            filename=filename,
            content=content,
            operation_type="create",
            operation_id=operation_id,
            namespace=normalized_namespace,
            logical_key=normalized_logical_key,
        )

    async def update_document(
        self,
        document_id: str,
        *,
        filename: str,
        content: bytes,
        operation_id: str | None = None,
    ) -> dict[str, Any]:
        content_hash = self._content_hash(content)
        if operation_id:
            existing = self.journal.get(operation_id)
            if existing:
                self._assert_same_operation(
                    existing,
                    operation_type="update",
                    document_id=document_id,
                    content_hash=content_hash,
                )
                return {**existing, "changed": True}
        operation_id = operation_id or str(uuid.uuid4())
        if not self.journal.acquire_lock(document_id, operation_id, self.lease_seconds):
            raise OperationBusyError("Another document operation is active")
        try:
            document, version_record, changed = self.registry.create_next_version(
                document_id, filename, content
            )
            if not changed:
                self.journal.release_lock(document_id, operation_id)
                return self._record_unchanged(
                    operation_id, "update", document, version_record, filename,
                    namespace=document.get("namespace"), logical_key=document.get("logical_key"),
                )
            return await self._run_reserved_version(
                document=document,
                version_record=version_record,
                filename=filename,
                content=content,
                operation_type="update",
                operation_id=operation_id,
                namespace=document.get("namespace"),
                logical_key=document.get("logical_key"),
            )
        except Exception:
            # _run_reserved_version owns a reserved version and records its
            # outcome. A reservation error happens before that responsibility.
            self.journal.release_lock(document_id, operation_id)
            raise

    def _record_unchanged(
        self,
        operation_id: str | None,
        operation_type: str,
        document: dict[str, Any],
        version_record: dict[str, Any],
        filename: str,
        *,
        namespace: str | None = None,
        logical_key: str | None = None,
    ) -> dict[str, Any]:
        operation_id = operation_id or str(uuid.uuid4())
        record = self.journal.create(
            self._operation_record(
                operation_id,
                operation_type,
                document["document_id"],
                int(version_record["version"]),
                document.get("current_version"),
                version_record.get("content_hash"),
                filename,
                namespace,
                logical_key,
            )
        )
        if record.get("status") != "succeeded":
            self.journal.complete_step(operation_id, "unchanged")
            record = self.journal.update(operation_id, status="succeeded", error_summary=None)
        return {**record, "changed": False}

    async def _run_reserved_version(
        self,
        *,
        document: dict[str, Any],
        version_record: dict[str, Any],
        filename: str,
        content: bytes,
        operation_type: str,
        operation_id: str | None,
        namespace: str | None = None,
        logical_key: str | None = None,
    ) -> dict[str, Any]:
        document_id = document["document_id"]
        version = int(version_record["version"])
        operation_id = operation_id or str(uuid.uuid4())
        if not self.journal.acquire_lock(document_id, operation_id, self.lease_seconds):
            self._mark_reserved_version_failed(document_id, version, "Another document operation is active")
            raise OperationBusyError("Another document operation is active")

        operation = self.journal.create(
            self._operation_record(
                operation_id,
                operation_type,
                document_id,
                version,
                document.get("current_version"),
                version_record.get("content_hash"),
                filename,
                namespace,
                logical_key,
            )
        )
        if operation.get("status") == "succeeded":
            self.journal.release_lock(document_id, operation_id)
            return {**operation, "changed": True}

        try:
            prepared = await self.processor.prepare(
                content=content,
                filename=filename,
                document_id=document_id,
                version=version,
                content_hash=version_record["content_hash"],
                operation_id=operation_id,
            )
            self.journal.complete_step(
                operation_id,
                "prepared",
                processing_metadata=dict(prepared.processing_metadata),
            )

            await self.vector_store.stage_document_version(
                document_id,
                version,
                prepared.chunks,
                version_record["content_hash"],
                source=filename,
            )
            vector_ids = self.vector_store.list_document_vector_ids(document_id, version)
            if len(vector_ids) != len(prepared.chunks):
                raise RuntimeError("Vector stage count does not match prepared chunks")
            self.journal.complete_step(operation_id, "vector_staged", vector_ids=vector_ids)

            await self.knowledge_graph.stage_document_version(
                document_id,
                version,
                version_record["content_hash"],
                filename,
                prepared.entities,
                prepared.relations,
            )
            evidence = await self.knowledge_graph.list_document_evidence(document_id, version)
            if len(evidence) != len(prepared.relations):
                raise RuntimeError("Graph stage count does not match prepared relations")
            evidence_keys = [row.get("evidence_key") for row in evidence if row.get("evidence_key")]
            self.journal.complete_step(
                operation_id, "graph_staged", graph_evidence_keys=evidence_keys
            )

            await self.vector_store.activate_document_version(document_id, version)
            self.journal.complete_step(operation_id, "vector_activated")
            await self.knowledge_graph.activate_document_version(document_id, version)
            self.journal.complete_step(operation_id, "graph_activated")
            self.registry.transition(
                document_id, version, "ready", chunk_count=len(prepared.chunks)
            )
            self.journal.complete_step(operation_id, "mongo_ready")
            operation = self.journal.update(operation_id, status="succeeded", error_summary=None)
            return {**operation, "changed": True}
        except Exception as error:
            await self._compensate(operation_id, error)
            raise
        finally:
            operation = self.journal.get(operation_id)
            if operation and operation.get("status") in {"succeeded", "failed", "needs_reconciliation"}:
                self.journal.release_lock(document_id, operation_id)

    def _mark_reserved_version_failed(self, document_id: str, version: int, error: Exception | str) -> None:
        try:
            self.registry.transition(document_id, version, "failed", error=error)
        except Exception:
            pass

    async def _compensate(self, operation_id: str, original_error: Exception) -> None:
        operation = self.journal.get(operation_id)
        if operation is None:
            return
        failure = classify_safe_processing_error(original_error, phase=self._failure_phase(original_error))
        self.journal.update(
            operation_id,
            status="compensating",
            error_summary=safe_error(original_error),
            **failure.as_dict(),
        )
        completed = set(operation.get("completed_steps", []))
        document_id = operation["document_id"]
        version = int(operation["version"])
        previous = operation.get("previous_current_version")
        failures: list[Exception] = []

        async def compensate(step: str, action: Any) -> None:
            if step in operation.get("compensation_steps", []):
                return
            try:
                value = action()
                if hasattr(value, "__await__"):
                    await value
                self.journal.complete_compensation(operation_id, step)
            except Exception as exc:  # preserve original business error outside this routine
                failures.append(exc)

        if "graph_activated" in completed:
            await compensate("graph_deactivated", lambda: self.knowledge_graph.deactivate_document_version(document_id, version))
        if "vector_activated" in completed:
            await compensate("vector_deactivated", lambda: self.vector_store.deactivate_document_version(document_id, version))
        if "graph_staged" in completed:
            await compensate("graph_stage_deleted", lambda: self.knowledge_graph.delete_document_version(document_id, version))
        if "vector_staged" in completed:
            await compensate("vector_stage_deleted", lambda: self.vector_store.delete_document_version(document_id, version))
        if previous is not None and "vector_activated" in completed:
            await compensate("previous_vector_restored", lambda: self.vector_store.activate_document_version(document_id, previous))
        if previous is not None and "graph_activated" in completed:
            await compensate("previous_graph_restored", lambda: self.knowledge_graph.activate_document_version(document_id, previous))

        if failures:
            self.journal.update(
                operation_id,
                status="needs_reconciliation",
                error_summary=safe_error(failures[0]),
            )
            return
        self._mark_reserved_version_failed(document_id, version, original_error)
        self.journal.update(
            operation_id,
            status="failed",
            error_summary=safe_error(original_error),
            **failure.as_dict(),
        )

    @staticmethod
    def _failure_phase(error: Exception) -> str:
        """Map our processing boundary wrappers without changing their semantics."""
        from services.document_processor import (
            DocumentParseError,
            InvalidExtractionResult,
            KnowledgeExtractionError,
            ProcessingTimeoutError,
        )

        if isinstance(error, DocumentParseError):
            return "parse"
        if isinstance(error, KnowledgeExtractionError):
            return "extract"
        if isinstance(error, InvalidExtractionResult):
            return "normalize"
        if isinstance(error, ProcessingTimeoutError):
            phase = getattr(error, "safe_processing_phase", "unknown")
            return phase if phase in {"parse", "extract"} else "unknown"
        return "unknown"

    async def delete_document(self, document_id: str, operation_id: str | None = None) -> dict[str, Any]:
        if operation_id:
            existing = self.journal.get(operation_id)
            if existing:
                return await self.recover_operation(operation_id) or existing
        document = self.registry.find(document_id)
        if document is None or document.get("current_version") is None:
            raise RegistryError("Document has no current version to delete")
        if document.get("status") == "deleted":
            raise RegistryError("Document is already deleted")
        version = int(document["current_version"])
        operation_id = operation_id or str(uuid.uuid4())
        if not self.journal.acquire_lock(document_id, operation_id, self.lease_seconds):
            raise OperationBusyError("Another document operation is active")
        operation = self.journal.create(
            self._operation_record(operation_id, "delete", document_id, version, version)
        )
        try:
            self.registry.mark_deleting(document_id, version)
            self.journal.complete_step(operation_id, "mongo_deleting")
            await self.vector_store.deactivate_document_version(document_id, version)
            self.journal.complete_step(operation_id, "vector_deactivated")
            await self.knowledge_graph.deactivate_document_version(document_id, version)
            self.journal.complete_step(operation_id, "graph_deactivated")
            self.registry.mark_deleted(document_id, version)
            self.journal.complete_step(operation_id, "mongo_deleted")
            try:
                await self.vector_store.delete_document(document_id)
                self.journal.complete_step(operation_id, "vector_cleaned")
                await self.knowledge_graph.delete_document(document_id)
                self.journal.complete_step(operation_id, "graph_cleaned")
            except Exception as cleanup_error:
                return self.journal.update(
                    operation_id,
                    status="cleanup_pending",
                    error_summary=safe_error(cleanup_error),
                )
            return self.journal.update(operation_id, status="succeeded", error_summary=None)
        finally:
            operation = self.journal.get(operation_id)
            if operation and operation.get("status") in {"succeeded", "cleanup_pending", "failed"}:
                self.journal.release_lock(document_id, operation_id)

    async def recover_operation(self, operation_id: str) -> dict[str, Any] | None:
        operation = self.journal.get(operation_id)
        if operation is None or operation.get("status") == "succeeded":
            return operation
        document_id = operation["document_id"]
        if not self.journal.acquire_lock(document_id, operation_id, self.lease_seconds):
            raise OperationBusyError("Operation is owned by another active worker")
        try:
            if operation.get("operation_type") == "delete" and operation.get("status") == "cleanup_pending":
                try:
                    await self.vector_store.delete_document(document_id)
                    self.journal.complete_step(operation_id, "vector_cleaned")
                    await self.knowledge_graph.delete_document(document_id)
                    self.journal.complete_step(operation_id, "graph_cleaned")
                    return self.journal.update(operation_id, status="succeeded", error_summary=None)
                except Exception as error:
                    return self.journal.update(
                        operation_id, status="cleanup_pending", error_summary=safe_error(error)
                    )
            await self._compensate(operation_id, RuntimeError(operation.get("error_summary") or "recovery"))
            return self.journal.get(operation_id)
        finally:
            operation = self.journal.get(operation_id)
            if operation and operation.get("status") in {"succeeded", "failed", "needs_reconciliation", "cleanup_pending"}:
                self.journal.release_lock(document_id, operation_id)

    async def reconcile_incomplete_operations(self) -> list[dict[str, Any] | None]:
        """Explicit recovery entrypoint; callers decide when to run it after restart."""
        recovered: list[dict[str, Any] | None] = []
        now = utcnow()
        for operation in self.journal.incomplete():
            if operation.get("status") == "running":
                lock = getattr(self.journal, "locks", None)
                if lock is not None:
                    owner = lock.find_one({"document_id": operation["document_id"]})
                    if owner and owner.get("expires_at") and owner["expires_at"] > now:
                        continue
            recovered.append(await self.recover_operation(operation["operation_id"]))
        return recovered
