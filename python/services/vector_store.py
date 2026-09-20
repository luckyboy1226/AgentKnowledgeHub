"""Vector storage with legacy compatibility and version-aware Chroma helpers."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import re
from typing import Any
from uuid import UUID

from agents.doc_parser_agent import DocumentChunk
from config import settings
from providers.embeddings import EmbeddingProvider, EmbeddingProviderError


class VectorStoreService:
    """Unified vector interface; version lifecycle helpers currently require Chroma."""

    COLLECTION_NAME = "knowledge_chunks"
    SEARCH_OVERFETCH_FACTOR = 5
    SEARCH_MAX_CANDIDATES = 50

    _COLLECTION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,127}$")

    def __init__(
        self,
        embeddings: EmbeddingProvider,
        *,
        collection_name: str | None = None,
        embedding_space_id: str | None = None,
    ) -> None:
        self.embeddings = embeddings
        self._store: Any = None
        self._backend = settings.vector_store_type
        self.collection_name = collection_name or settings.chroma_collection_name or self.COLLECTION_NAME
        if not self._COLLECTION_NAME.fullmatch(self.collection_name):
            raise ValueError("Invalid Chroma collection name")
        self.embedding_space_id = embedding_space_id or settings.resolved_embedding_space_id

    def _identity_metadata(self) -> dict[str, Any]:
        return {
            "hnsw:space": "cosine",
            "embedding_provider": self.embeddings.provider_name,
            "embedding_model": self.embeddings.model_name,
            "embedding_dimensions": int(self.embeddings.dimensions),
            "embedding_space_id": self.embedding_space_id,
            "schema_version": 1,
        }

    def _validate_collection_identity(self, metadata: dict[str, Any] | None) -> None:
        metadata = dict(metadata or {})
        expected = self._identity_metadata()
        # The original collection predates identity metadata.  It remains
        # readable only for its matching legacy space; it is never a target
        # for a new configured model/dimension.
        identity_keys = set(expected) - {"hnsw:space", "schema_version"}
        if not identity_keys.intersection(metadata):
            if self.collection_name == self.COLLECTION_NAME and self.embeddings.dimensions == 1536:
                return
            raise EmbeddingProviderError("legacy_collection_write_refused")
        if metadata.get("embedding_provider") != expected["embedding_provider"]:
            raise EmbeddingProviderError("collection_identity_mismatch")
        if metadata.get("embedding_model") != expected["embedding_model"]:
            raise EmbeddingProviderError("embedding_model_mismatch")
        if metadata.get("embedding_dimensions") != expected["embedding_dimensions"]:
            raise EmbeddingProviderError("embedding_dimension_mismatch")
        if metadata.get("embedding_space_id") != expected["embedding_space_id"]:
            raise EmbeddingProviderError("embedding_space_mismatch")
        if metadata.get("hnsw:space") != "cosine":
            raise EmbeddingProviderError("embedding_space_mismatch")

    # ── initialization ───────────────────────────────────────

    async def init(self) -> None:
        if self._backend == "chroma":
            await self._init_chroma()
        else:
            await self._init_pgvector()

    async def _init_chroma(self) -> None:
        import chromadb

        # Chroma's HTTP client can resolve ``localhost`` to IPv6 on Windows.
        # The Compose port is bound on IPv4, so make the local default explicit
        # while leaving a configured remote host untouched.
        client = chromadb.HttpClient(
            host=self.chroma_http_host(settings.chroma_host), port=settings.chroma_port
        )
        self._store = client.get_or_create_collection(name=self.collection_name, metadata=self._identity_metadata())
        self._validate_collection_identity(getattr(self._store, "metadata", None))

    @staticmethod
    def chroma_http_host(host: str) -> str:
        """Return a deterministic IPv4 endpoint for the local Compose service."""
        normalized = str(host).strip().lower()
        # On Windows, environment resolution can produce the IPv6 loopback
        # literal even when Compose publishes Chroma only on IPv4.
        if normalized in {"localhost", "::1", "[::1]"}:
            return "127.0.0.1"
        return str(host)

    async def _init_pgvector(self) -> None:
        from langchain_community.vectorstores import PGVector

        self._store = PGVector(
            connection_string=settings.pgvector_dsn,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )

    # ── legacy CRUD ──────────────────────────────────────────

    async def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Store legacy path-derived chunks without changing existing callers."""
        if not chunks:
            return 0

        texts = [chunk.content for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [
            {
                "doc_id": str(chunk.doc_id),
                "doc_type": str(chunk.doc_type.value),
                "source": self.safe_source(chunk.metadata.get("source", "")),
                "chunk_index": int(chunk.chunk_index),
                "legacy": True,
            }
            for chunk in chunks
        ]

        if self._backend == "chroma":
            vectors = await self.embeddings.aembed_documents(texts)
            self._store.upsert(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
        else:
            await self._store.aadd_texts(texts=texts, metadatas=metadatas, ids=ids)
        return len(chunks)

    async def delete_by_doc_id(self, doc_id: str) -> int:
        """Maintenance-only cleanup for pre-S3 legacy parser IDs.

        Formal document deletion must use ``DocumentUpdateCoordinator`` and
        ``delete_document``. No API, Coordinator, or future CDC adapter may
        use this path because it lacks registry/version Saga semantics.
        """
        if self._backend == "chroma":
            existing = self._store.get(where={"doc_id": str(doc_id)}, include=[])
            ids = existing.get("ids", [])
            if ids:
                self._store.delete(ids=ids)
            return len(ids)
        return 0

    # ── version-aware Chroma lifecycle ───────────────────────

    @staticmethod
    def vector_id(document_id: str | UUID, version: int, chunk_index: int) -> str:
        """Return the deterministic vector identity for one document chunk."""
        return f"{str(document_id)}:v{int(version)}:c{int(chunk_index)}"

    @staticmethod
    def scalar_metadata(value: Any) -> str | int | float | bool:
        """Convert values to the scalar metadata types accepted by Chroma 1.5.9."""
        if isinstance(value, bool):
            return value
        if isinstance(value, (str, int, float)):
            return value
        if isinstance(value, (Path, UUID, datetime, date)):
            return str(value)
        return str(value)

    @staticmethod
    def safe_source(source: Any) -> str:
        """Return a display filename, never a local absolute path."""
        text = str(source or "").replace("\\", "/")
        return text.rsplit("/", 1)[-1]

    def _require_chroma(self) -> None:
        if self._backend != "chroma" or self._store is None:
            raise NotImplementedError("Document version vector operations require an initialized Chroma store")

    def _version_records(self, document_id: str, version: int | None = None) -> list[tuple[str, dict[str, Any]]]:
        """Return only exact records for a versioned document, never collection-wide rows."""
        self._require_chroma()
        result = self._store.get(
            where={"document_id": str(document_id)}, include=["metadatas"]
        )
        ids = result.get("ids", [])
        metadatas = result.get("metadatas", [])
        records: list[tuple[str, dict[str, Any]]] = []
        for vector_id, metadata in zip(ids, metadatas):
            metadata = metadata or {}
            if version is None or metadata.get("document_version") == int(version):
                records.append((str(vector_id), metadata))
        return records

    def _update_metadata(self, vector_id: str, metadata: dict[str, Any]) -> None:
        # Chroma 1.5.9 supports update(ids, metadatas). It merges metadata with
        # the existing record, so lifecycle updates do not erase provenance.
        self._store.update(ids=[vector_id], metadatas=[metadata])

    async def stage_document_version(
        self,
        document_id: str | UUID,
        version: int,
        chunks: list[DocumentChunk],
        content_hash: str,
        *,
        source: str | Path | None = None,
    ) -> int:
        """Idempotently write a non-current, processing Chroma document version.

        Chunks are written one by one after embedding so the IDs that reached
        Chroma are known. If an upsert partially fails, only those exact IDs are
        removed; old versions and unrelated documents are untouched.
        """
        self._require_chroma()
        if not chunks:
            return 0

        document_id_text = str(document_id)
        version_number = int(version)
        texts = [chunk.content for chunk in chunks]
        vectors = await self.embeddings.aembed_documents(texts)
        if len(vectors) != len(chunks):
            raise EmbeddingProviderError("Document chunk embedding count does not match input chunks.")

        staged_ids = [
            self.vector_id(document_id_text, version_number, chunk.chunk_index)
            for chunk in chunks
        ]
        existing_ids = set(self._store.get(ids=staged_ids, include=[]).get("ids", []))
        newly_written_ids: list[str] = []
        try:
            for chunk, vector in zip(chunks, vectors):
                vector_id = self.vector_id(document_id_text, version_number, chunk.chunk_index)
                metadata = {
                    "document_id": self.scalar_metadata(document_id_text),
                    "document_version": self.scalar_metadata(version_number),
                    "chunk_id": vector_id,
                    "chunk_index": self.scalar_metadata(int(chunk.chunk_index)),
                    "content_hash": self.scalar_metadata(content_hash),
                    "source": self.safe_source(source or chunk.metadata.get("source", "")),
                    "is_current": False,
                    "status": "processing",
                    "doc_type": str(chunk.doc_type.value),
                }
                # Parent–Child ingestion is feature-gated upstream. These
                # optional scalar fields preserve the exact child provenance
                # without putting parent content into Chroma.
                for key in (
                    "parent_chunk_id",
                    "section_title",
                    "page_number",
                    "table_id",
                    "estimated_token_count",
                ):
                    value = (chunk.metadata or {}).get(key)
                    if value is not None and value != "":
                        metadata[key] = self.scalar_metadata(value)
                self._store.upsert(
                    ids=[vector_id],
                    embeddings=[vector],
                    documents=[chunk.content],
                    metadatas=[metadata],
                )
                if vector_id not in existing_ids:
                    newly_written_ids.append(vector_id)
        except Exception:
            if newly_written_ids:
                try:
                    self._store.delete(ids=newly_written_ids)
                except Exception:
                    # The original write error remains the actionable failure.
                    pass
            raise
        return len(chunks)

    async def activate_document_version(self, document_id: str | UUID, version: int) -> int:
        """Promote one fully staged version and deactivate its previous current version.

        Chroma has no multi-document transaction. Old vectors are deactivated
        first, then the new vectors are made current. On any failure, the helper
        deactivates already-promoted new vectors and restores already-deactivated
        old vectors before re-raising the original exception.
        """
        document_id_text = str(document_id)
        version_number = int(version)
        new_ids = self.list_document_vector_ids(document_id_text, version_number)
        if not new_ids:
            raise ValueError("Cannot activate a document version with no staged vectors")

        all_records = self._version_records(document_id_text)
        old_ids = [
            vector_id
            for vector_id, metadata in all_records
            if metadata.get("document_version") != version_number and metadata.get("is_current") is True
        ]
        deactivated_old: list[str] = []
        activated_new: list[str] = []
        try:
            for vector_id in old_ids:
                self._update_metadata(vector_id, {"is_current": False})
                deactivated_old.append(vector_id)
            for vector_id in new_ids:
                self._update_metadata(vector_id, {"is_current": True, "status": "ready"})
                activated_new.append(vector_id)
        except Exception:
            for vector_id in activated_new:
                try:
                    self._update_metadata(vector_id, {"is_current": False, "status": "processing"})
                except Exception:
                    pass
            for vector_id in deactivated_old:
                try:
                    self._update_metadata(vector_id, {"is_current": True, "status": "ready"})
                except Exception:
                    pass
            raise
        return len(new_ids)

    async def deactivate_document_version(self, document_id: str | UUID, version: int) -> int:
        """Make one version ineligible for retrieval without deleting it."""
        records = self._version_records(str(document_id), int(version))
        for vector_id, _ in records:
            self._update_metadata(vector_id, {"is_current": False})
        return len(records)

    async def delete_document_version(self, document_id: str | UUID, version: int) -> int:
        """Precisely remove one document version, never an entire collection."""
        ids = self.list_document_vector_ids(str(document_id), int(version))
        if ids:
            self._store.delete(ids=ids)
        return len(ids)

    async def delete_document(self, document_id: str | UUID) -> int:
        """Precisely remove all versioned vectors for one logical document."""
        ids = self.list_document_vector_ids(str(document_id), version=None)
        if ids:
            self._store.delete(ids=ids)
        return len(ids)

    def count_document_version(self, document_id: str | UUID, version: int) -> int:
        return len(self.list_document_vector_ids(document_id, version))

    def list_document_vector_ids(
        self, document_id: str | UUID, version: int | None
    ) -> list[str]:
        return [vector_id for vector_id, _ in self._version_records(str(document_id), version)]

    # ── retrieval ────────────────────────────────────────────

    @staticmethod
    def _is_retrievable(metadata: dict[str, Any]) -> bool:
        """Accept legacy rows, but require explicit ready/current state for S3 rows."""
        versioned_keys = {
            "document_id",
            "document_version",
            "chunk_id",
            "content_hash",
            "is_current",
            "status",
        }
        if metadata.get("legacy") is True:
            return True
        if not any(key in metadata for key in versioned_keys):
            return True
        return metadata.get("is_current") is True and metadata.get("status") == "ready"

    @staticmethod
    def _in_evaluation_scope(metadata: dict[str, Any], allowed_document_ids: frozenset[str]) -> bool:
        """Scoped evaluation never treats legacy/missing provenance as eligible."""
        document_id = metadata.get("document_id")
        version = metadata.get("document_version")
        return str(document_id) in allowed_document_ids and isinstance(version, int) and version >= 1

    async def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        allowed_document_ids: frozenset[str] | None = None,
        scope_diagnostics: dict[str, int] | None = None,
    ) -> list[tuple[dict, float]]:
        """Search current S3 rows plus legacy rows, with bounded compatibility over-fetch."""
        if top_k <= 0:
            return []
        if self._backend == "chroma":
            q_vec = await self.embeddings.aembed_query(query)
            if len(q_vec) != self.embeddings.dimensions:
                raise EmbeddingProviderError("Query embedding dimension does not match provider configuration.")
            candidate_limit = min(
                max(int(top_k) * self.SEARCH_OVERFETCH_FACTOR, int(top_k)),
                self.SEARCH_MAX_CANDIDATES,
            )
            results = self._store.query(
                query_embeddings=[q_vec],
                n_results=candidate_limit,
                include=["documents", "metadatas", "distances"],
            )
            out: list[tuple[dict, float]] = []
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]
            for doc, metadata, distance in zip(docs, metas, dists):
                metadata = dict(metadata or {})
                if not self._is_retrievable(metadata):
                    continue
                if allowed_document_ids is not None and not self._in_evaluation_scope(
                    metadata, allowed_document_ids
                ):
                    if scope_diagnostics is not None:
                        scope_diagnostics["vector_scope_rejected_count"] = (
                            scope_diagnostics.get("vector_scope_rejected_count", 0) + 1
                        )
                    continue
                metadata["source"] = self.safe_source(metadata.get("source", ""))
                out.append(
                    (
                        {
                            "content": doc,
                            "source": metadata["source"],
                            "metadata": metadata,
                        },
                        1.0 - distance,
                    )
                )
                if len(out) == top_k:
                    break
            return out

        results = await self._store.asimilarity_search_with_score(query, k=top_k)
        output = [
            (
                {
                    "content": doc.page_content,
                    "source": self.safe_source(doc.metadata.get("source", "")),
                    "metadata": doc.metadata,
                },
                score,
            )
            for doc, score in results
        ]
        if allowed_document_ids is None:
            return output
        scoped_output: list[tuple[dict, float]] = []
        for record, score in output:
            metadata = dict(record.get("metadata") or {})
            if not self._in_evaluation_scope(metadata, allowed_document_ids):
                if scope_diagnostics is not None:
                    scope_diagnostics["vector_scope_rejected_count"] = (
                        scope_diagnostics.get("vector_scope_rejected_count", 0) + 1
                    )
                continue
            scoped_output.append((record, score))
            if len(scoped_output) == top_k:
                break
        return scoped_output

    async def get_stats(self) -> dict:
        """Return a small datastore health/statistics payload."""
        if self._backend == "chroma":
            # Readiness calls this method.  Verify identity before exposing a
            # healthy vector dependency so a changed provider/model/dimension
            # cannot start against an incompatible collection.
            self._validate_collection_identity(getattr(self._store, "metadata", None))
            count = self._store.count()
            return {"backend": "chroma", "total_vectors": count, "collection": self.collection_name, "embedding_space": self.embedding_space_id}
        return {"backend": "pgvector", "collection": self.collection_name, "embedding_space": self.embedding_space_id}
