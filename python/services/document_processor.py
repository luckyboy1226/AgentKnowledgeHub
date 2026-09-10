"""Safe adapter from uploaded bytes to the Saga's in-memory document artifact.

This module deliberately has no database or provider factory dependency.  The
caller injects already-configured parser and extractor instances, so the
existing parser/extractor prompts and provider lifecycle remain authoritative.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
import time
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

from agents.doc_parser_agent import DocType, DocumentChunk
from agents.knowledge_extract_agent import Entity, ExtractionResult, Relation
from services.document_update_coordinator import PreparedDocument


logger = logging.getLogger(__name__)
_WINDOWS_RESERVED = re.compile(r'[<>:"|?*]')
_ABSOLUTE_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|[\\/]{1,2})")
_SECRET_METADATA_KEY = re.compile(r"(?:api[_-]?key|authorization|password|secret|token)", re.IGNORECASE)


class UnsupportedDocumentType(ValueError):
    """The injected parser does not support the supplied safe filename."""


class DocumentParseError(RuntimeError):
    """A parser failure with the original exception retained as ``__cause__``."""


class EmptyDocumentError(ValueError):
    """The parser produced no usable chunks."""


class KnowledgeExtractionError(RuntimeError):
    """An extractor failure with the original exception retained as ``__cause__``."""


class InvalidExtractionResult(ValueError):
    """The parser or extractor returned an unsafe or internally inconsistent result."""


class ProcessingTimeoutError(TimeoutError):
    """An upstream parser/extractor timeout; no retry is performed here."""


class Parser(Protocol):
    async def parse(self, file_path: str) -> list[DocumentChunk]: ...


class Extractor(Protocol):
    async def extract(self, chunks: list[DocumentChunk]) -> list[ExtractionResult]: ...


AuditSink = Callable[[dict[str, Any]], None]


def safe_filename(filename: str) -> str:
    """Return a display-only name, never a path, drive, UNC reference, or traversal."""
    value = str(filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    value = _WINDOWS_RESERVED.sub("_", value)
    value = re.sub(r"\.{2,}", "_", value).strip(" .")
    value = re.sub(r"\s+", " ", value).strip()
    return value[:180] if value and value.strip("_") else "unnamed"


def _json_safe(value: Any) -> Any:
    """Convert metadata to scalar/list/dict JSON values without retaining paths."""
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        return safe_filename(value) if _ABSOLUTE_PATH.match(value) else value
    if isinstance(value, Path):
        return safe_filename(value.name)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key)[:100]: "[redacted]" if _SECRET_METADATA_KEY.search(str(key)) else _json_safe(item)
            for key, item in value.items()
        }
    rendered = str(value)
    return safe_filename(rendered) if _ABSOLUTE_PATH.match(rendered) else rendered


class DocumentProcessorAdapter:
    """Adapt existing parsing and extraction Agents to ``DocumentProcessor.prepare``.

    The adapter creates one file below ``temp_root`` because ``DocParserAgent``
    currently has a path-based public API.  It only gives the parser that
    controlled path, normalizes the returned chunks to the stable registry ID,
    and removes the file in ``finally`` on every outcome.
    """

    def __init__(
        self,
        parser: Parser,
        extractor: Extractor,
        *,
        temp_root: str | Path,
        max_chunks: int = 10_000,
        max_entities: int = 20_000,
        max_relations: int = 20_000,
        audit_sink: AuditSink | None = None,
    ) -> None:
        if min(max_chunks, max_entities, max_relations) < 1:
            raise ValueError("Document processor limits must be positive")
        self.parser = parser
        self.extractor = extractor
        self.temp_root = Path(temp_root).resolve()
        self.max_chunks = max_chunks
        self.max_entities = max_entities
        self.max_relations = max_relations
        self.audit_sink = audit_sink

    async def prepare(
        self,
        *,
        content: bytes,
        filename: str,
        document_id: str,
        version: int,
        content_hash: str,
        operation_id: str,
        doc_type: DocType | None = None,
    ) -> PreparedDocument:
        """Prepare a validated in-memory artifact; no storage writes or retries occur."""
        source = safe_filename(filename)
        self._validate_inputs(content, source, document_id, version, content_hash, operation_id)
        temp_path = self._write_temp_file(content, source)
        try:
            parse_started = time.monotonic()
            try:
                parsed_chunks = await self.parser.parse(str(temp_path))
            except asyncio.TimeoutError as exc:
                self._audit(operation_id, "parse", success=False, error_type=type(exc).__name__)
                raise ProcessingTimeoutError("Document parsing timed out") from exc
            except Exception as exc:
                self._audit(operation_id, "parse", success=False, error_type=type(exc).__name__)
                raise DocumentParseError("Document parsing failed") from exc
            parse_elapsed_ms = round((time.monotonic() - parse_started) * 1000, 3)
            chunks = self._normalize_chunks(parsed_chunks, document_id, version, content_hash, source, doc_type)
            self._audit(operation_id, "parse", success=True, elapsed_ms=parse_elapsed_ms, chunk_count=len(chunks))

            extract_started = time.monotonic()
            try:
                extraction = await self.extractor.extract(chunks)
            except asyncio.TimeoutError as exc:
                self._audit(operation_id, "extract", success=False, error_type=type(exc).__name__)
                raise ProcessingTimeoutError("Knowledge extraction timed out") from exc
            except Exception as exc:
                self._audit(operation_id, "extract", success=False, error_type=type(exc).__name__)
                raise KnowledgeExtractionError("Knowledge extraction failed") from exc
            extract_elapsed_ms = round((time.monotonic() - extract_started) * 1000, 3)
            entities, relations = self._normalize_extraction(extraction)
            self._audit(
                operation_id,
                "extract",
                success=True,
                elapsed_ms=extract_elapsed_ms,
                entity_count=len(entities),
                relation_count=len(relations),
            )
            return PreparedDocument(
                chunks=chunks,
                entities=entities,
                relations=relations,
                document_id=document_id,
                document_version=version,
                content_hash=content_hash,
                source=source,
                processing_metadata={
                    "operation_id": operation_id,
                    "parse_elapsed_ms": parse_elapsed_ms,
                    "extraction_elapsed_ms": extract_elapsed_ms,
                    "chunk_count": len(chunks),
                    "entity_count": len(entities),
                    "relation_count": len(relations),
                },
            )
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError as exc:
                # Do not hide a parser/extractor failure, and do not expose a path.
                logger.warning("document processor temporary-file cleanup failed: %s", type(exc).__name__)

    def _write_temp_file(self, content: bytes, source: str) -> Path:
        self.temp_root.mkdir(parents=True, exist_ok=True)
        suffix = Path(source).suffix.lower()
        if suffix and len(suffix) > 12:
            suffix = ""
        descriptor, raw_path = tempfile.mkstemp(prefix="document-", suffix=suffix, dir=str(self.temp_root))
        path = Path(raw_path).resolve()
        try:
            if self.temp_root not in path.parents:
                raise InvalidExtractionResult("Temporary file escaped the configured directory")
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                stream.write(content)
            return path
        except Exception:
            if descriptor != -1:
                os.close(descriptor)
            Path(raw_path).unlink(missing_ok=True)
            raise

    @staticmethod
    def _validate_inputs(
        content: bytes,
        source: str,
        document_id: str,
        version: int,
        content_hash: str,
        operation_id: str,
    ) -> None:
        if not isinstance(content, bytes) or not content:
            raise EmptyDocumentError("Document content is empty")
        if not source or not document_id or not content_hash or not operation_id or int(version) < 1:
            raise InvalidExtractionResult("Document processing identity is incomplete")

    def _normalize_chunks(
        self,
        parsed_chunks: list[DocumentChunk],
        document_id: str,
        version: int,
        content_hash: str,
        source: str,
        requested_doc_type: DocType | None,
    ) -> list[DocumentChunk]:
        if not parsed_chunks:
            raise EmptyDocumentError("Document parsing produced no chunks")
        if len(parsed_chunks) > self.max_chunks:
            raise InvalidExtractionResult("Document produced too many chunks")
        ordered = sorted(parsed_chunks, key=lambda chunk: chunk.chunk_index)
        if [chunk.chunk_index for chunk in ordered] != list(range(len(ordered))):
            raise InvalidExtractionResult("Chunk indexes must be unique and contiguous")
        normalized: list[DocumentChunk] = []
        for index, chunk in enumerate(ordered):
            text = str(chunk.content or "").strip()
            if not text:
                raise EmptyDocumentError("Document contains an empty chunk")
            chunk_type = requested_doc_type or chunk.doc_type
            try:
                chunk_type = DocType(chunk_type)
            except ValueError as exc:
                raise UnsupportedDocumentType("Unsupported document type") from exc
            metadata = _json_safe(chunk.metadata or {})
            if not isinstance(metadata, dict):
                raise InvalidExtractionResult("Chunk metadata must be an object")
            metadata.update(
                {
                    "source": source,
                    "document_id": document_id,
                    "document_version": int(version),
                    "content_hash": content_hash,
                    "chunk_index": index,
                }
            )
            # JSON serialization is an explicit contract for downstream metadata.
            json.dumps(metadata, ensure_ascii=False, sort_keys=True)
            normalized.append(
                DocumentChunk(text, document_id, index, chunk_type, metadata, embedding=None)
            )
        return normalized

    def _normalize_extraction(
        self, results: list[ExtractionResult]
    ) -> tuple[list[Entity], list[Relation]]:
        entities_by_key: dict[tuple[str, str], Entity] = {}
        raw_relations: list[Relation] = []
        for result in results or []:
            for entity in result.entities:
                name = str(entity.name or "").strip()
                entity_type = str(entity.type or "").strip()
                if not name or not entity_type:
                    raise InvalidExtractionResult("Entity identity is incomplete")
                key = (name, entity_type)
                if key not in entities_by_key:
                    properties = _json_safe(entity.properties or {})
                    if not isinstance(properties, dict):
                        raise InvalidExtractionResult("Entity properties must be an object")
                    entities_by_key[key] = Entity(
                        name=name,
                        type=entity_type,
                        description=str(entity.description or ""),
                        properties=properties,
                    )
            raw_relations.extend(result.relations)

        if len(entities_by_key) > self.max_entities or len(raw_relations) > self.max_relations:
            raise InvalidExtractionResult("Extraction result exceeds configured limits")
        entity_names = {entity.name for entity in entities_by_key.values()}
        relations_by_key: dict[tuple[str, str, str, str], Relation] = {}
        for relation in raw_relations:
            head, tail = str(relation.head or "").strip(), str(relation.tail or "").strip()
            if not head or not tail or head not in entity_names or tail not in entity_names:
                raise InvalidExtractionResult("Relation endpoint is missing from entities")
            # Keep the graph service's established relation identifier policy.
            from services.knowledge_graph import KnowledgeGraphService

            predicate = KnowledgeGraphService.safe_relationship_type(str(relation.relation or ""))
            properties = _json_safe(relation.properties or {})
            if not isinstance(properties, dict):
                raise InvalidExtractionResult("Relation properties must be an object")
            property_key = json.dumps(properties, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            key = (head, predicate, tail, property_key)
            if key not in relations_by_key:
                relations_by_key[key] = Relation(
                    head=head,
                    relation=predicate,
                    tail=tail,
                    confidence=float(relation.confidence),
                    properties=properties,
                )
        entities = [entities_by_key[key] for key in sorted(entities_by_key)]
        relations = [relations_by_key[key] for key in sorted(relations_by_key)]
        return entities, relations

    def _audit(self, operation_id: str, phase: str, **fields: Any) -> None:
        event = {"operation_id": operation_id, "phase": phase, **fields}
        if self.audit_sink:
            self.audit_sink(event)
            return
        logger.info("document processor event=%s", json.dumps(event, sort_keys=True))
