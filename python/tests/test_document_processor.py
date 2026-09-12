"""Offline contract tests for the injected S3 document processor adapter."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import pytest
import httpx

from agents.doc_parser_agent import DocType, DocumentChunk
from agents.knowledge_extract_agent import Entity, ExtractionResult, Relation
from services.document_processor import (
    DocumentParseError,
    DocumentProcessorAdapter,
    EmptyDocumentError,
    InvalidExtractionResult,
    KnowledgeExtractionError,
    ProcessingTimeoutError,
    UnsupportedDocumentType,
    safe_filename,
)


class FakeParser:
    def __init__(self, chunks=None, error=None):
        self.chunks = chunks if chunks is not None else [sample_chunk()]
        self.error = error
        self.paths = []

    async def parse(self, file_path):
        path = Path(file_path)
        self.paths.append(path)
        assert path.exists()
        if self.error:
            raise self.error
        return deepcopy(self.chunks)


class FakeExtractor:
    def __init__(self, results=None, error=None):
        self.results = results if results is not None else [sample_extraction()]
        self.error = error
        self.calls = 0
        self.received_chunks = []

    async def extract(self, chunks):
        self.calls += 1
        self.received_chunks = deepcopy(chunks)
        if self.error:
            raise self.error
        return deepcopy(self.results)


def sample_chunk(index=0, content="Alice owns Project"):
    return DocumentChunk(content, "parser-path-id", index, DocType.TEXT, {"source": "/secret/path.txt"})


def sample_extraction():
    return ExtractionResult(
        entities=[Entity("Alice", "Person"), Entity("Project", "Concept")],
        relations=[Relation("Alice", "owns", "Project", 0.9)],
        events=[],
    )


@pytest.fixture
def setup(tmp_path):
    parser, extractor, events = FakeParser(), FakeExtractor(), []
    processor = DocumentProcessorAdapter(parser, extractor, temp_root=tmp_path / "controlled", audit_sink=events.append)
    return processor, parser, extractor, events, tmp_path / "controlled"


async def prepare(processor, **overrides):
    values = {
        "content": b"Alice owns Project",
        "filename": "report.txt",
        "document_id": "doc-stable",
        "version": 2,
        "content_hash": "a" * 64,
        "operation_id": "op-stable",
    }
    values.update(overrides)
    return await processor.prepare(**values)


@pytest.mark.asyncio
async def test_normal_document_produces_complete_stable_artifact(setup):
    processor, _, _, _, _ = setup
    result = await prepare(processor)
    assert (result.document_id, result.document_version, result.content_hash, result.source) == (
        "doc-stable", 2, "a" * 64, "report.txt"
    )
    assert result.chunks[0].doc_id == "doc-stable" and result.chunks[0].chunk_index == 0
    assert result.relations[0].relation == "OWNS"


@pytest.mark.asyncio
async def test_operation_context_is_only_in_processing_metadata(setup):
    processor, _, _, events, _ = setup
    result = await prepare(processor, operation_id="operation-123")
    assert result.processing_metadata["operation_id"] == "operation-123"
    assert all(event["operation_id"] == "operation-123" for event in events)


def test_safe_filename_removes_paths_drives_and_traversal():
    assert safe_filename(r"..\..\C:\private\report.txt") == "report.txt"
    assert safe_filename("//server/share/secret.pdf") == "secret.pdf"
    assert safe_filename("...") == "unnamed"


@pytest.mark.asyncio
async def test_traversal_filename_never_appears_as_source(setup):
    processor, _, extractor, _, _ = setup
    result = await prepare(processor, filename=r"..\..\C:\vault\safe.txt")
    assert result.source == "safe.txt"
    assert extractor.received_chunks[0].metadata["source"] == "safe.txt"


@pytest.mark.asyncio
async def test_temp_file_is_confined_to_controlled_directory_and_removed(setup):
    processor, parser, _, _, temp_root = setup
    await prepare(processor)
    assert parser.paths[0].parent == temp_root.resolve()
    assert not parser.paths[0].exists() and list(temp_root.glob("*")) == []


@pytest.mark.asyncio
async def test_temp_file_is_removed_after_parse_failure(setup):
    processor, parser, _, _, temp_root = setup
    parser.error = ValueError("bad parser")
    with pytest.raises(DocumentParseError):
        await prepare(processor)
    assert not parser.paths[0].exists() and list(temp_root.glob("*")) == []


@pytest.mark.asyncio
async def test_temp_file_is_removed_after_extract_failure(setup):
    processor, parser, extractor, _, temp_root = setup
    extractor.error = ValueError("bad extraction")
    with pytest.raises(KnowledgeExtractionError):
        await prepare(processor)
    assert parser.paths and not parser.paths[0].exists() and list(temp_root.glob("*")) == []


@pytest.mark.asyncio
async def test_provider_timeout_is_wrapped_and_audited_without_provider_text(setup):
    processor, _, extractor, events, _ = setup
    extractor.error = httpx.ReadTimeout("api_key=fake-key provider-body=do-not-store")
    with pytest.raises(KnowledgeExtractionError) as error:
        await prepare(processor)
    assert isinstance(error.value.__cause__, httpx.ReadTimeout)
    event = events[-1]
    assert event["phase"] == "extract"
    assert event["error_category"] == "provider_timeout"
    assert event["error_type"] == "ReadTimeout"
    assert "fake-key" not in str(event) and "provider-body" not in str(event)


@pytest.mark.asyncio
async def test_normalization_validation_is_audited_with_safe_category(setup):
    processor, _, extractor, events, _ = setup
    extractor.results = [ExtractionResult([Entity("Alice", "Person")], [Relation("", "uses", "Alice")], [])]
    with pytest.raises(InvalidExtractionResult):
        await prepare(processor)
    assert events[-1]["phase"] == "normalize"
    assert events[-1]["error_category"] == "extraction_validation_error"


@pytest.mark.asyncio
async def test_empty_input_is_rejected_before_parser(setup):
    processor, parser, _, _, _ = setup
    with pytest.raises(EmptyDocumentError):
        await prepare(processor, content=b"")
    assert parser.paths == []


@pytest.mark.asyncio
async def test_empty_parser_result_is_rejected(setup):
    processor, _, _, _, _ = setup
    processor.parser.chunks = []
    with pytest.raises(EmptyDocumentError):
        await prepare(processor)


@pytest.mark.asyncio
async def test_empty_chunk_is_rejected(setup):
    processor, _, _, _, _ = setup
    processor.parser.chunks = [sample_chunk(content="  ")]
    with pytest.raises(EmptyDocumentError):
        await prepare(processor)


@pytest.mark.asyncio
async def test_chunk_indexes_must_be_unique_and_contiguous(setup):
    processor, _, _, _, _ = setup
    processor.parser.chunks = [sample_chunk(0), sample_chunk(2)]
    with pytest.raises(InvalidExtractionResult, match="contiguous"):
        await prepare(processor)


@pytest.mark.asyncio
async def test_chunk_order_is_stable_and_metadata_is_versioned(setup):
    processor, _, _, _, _ = setup
    processor.parser.chunks = [sample_chunk(1, "second"), sample_chunk(0, "first")]
    result = await prepare(processor)
    assert [chunk.content for chunk in result.chunks] == ["first", "second"]
    assert result.chunks[0].metadata["document_version"] == 2
    assert result.chunks[0].metadata["content_hash"] == "a" * 64


@pytest.mark.asyncio
async def test_entities_are_stably_deduplicated(setup):
    processor, _, extractor, _, _ = setup
    extraction = sample_extraction()
    extraction.entities.append(Entity("Alice", "Person", "later"))
    extractor.results = [extraction, deepcopy(extraction)]
    result = await prepare(processor)
    assert [(entity.name, entity.type) for entity in result.entities] == [("Alice", "Person"), ("Project", "Concept")]


@pytest.mark.asyncio
async def test_relations_are_stably_deduplicated(setup):
    processor, _, extractor, _, _ = setup
    extraction = sample_extraction()
    extraction.relations.append(Relation("Alice", "owns", "Project", 0.1))
    extractor.results = [extraction, deepcopy(extraction)]
    result = await prepare(processor)
    assert len(result.relations) == 1


@pytest.mark.asyncio
async def test_relation_missing_entity_endpoint_is_dropped_safely(setup):
    processor, _, extractor, audits, _ = setup
    extractor.results = [ExtractionResult([Entity("Alice", "Person")], [Relation("Alice", "uses", "Missing")], [])]
    result = await prepare(processor)
    assert result.relations == []
    assert result.processing_metadata["dropped_relation_count"] == 1
    assert audits[-1]["dropped_relation_count"] == 1


@pytest.mark.asyncio
async def test_predicate_uses_graph_relation_safety_normalization(setup):
    processor, _, extractor, _, _ = setup
    extractor.results[0].relations[0].relation = "works at"
    result = await prepare(processor)
    assert result.relations[0].relation == "WORKS_AT"


@pytest.mark.asyncio
async def test_unsafe_predicate_falls_back_to_related_to(setup):
    processor, _, extractor, _, _ = setup
    extractor.results[0].relations[0].relation = "drop; delete"
    result = await prepare(processor)
    assert result.relations[0].relation == "RELATED_TO"


@pytest.mark.asyncio
async def test_metadata_is_json_safe_without_absolute_paths(setup):
    processor, _, extractor, _, _ = setup
    processor.parser.chunks[0].metadata = {
        "path": Path("C:/private/source.txt"),
        "string_path": r"C:\private\string-source.txt",
        "id": UUID("12345678-1234-5678-1234-567812345678"),
        "when": datetime(2026, 1, 2, tzinfo=timezone.utc),
        "api_key": "must-not-survive",
    }
    extractor.results[0].entities[0].properties = {"path": Path("C:/private/entity")}
    result = await prepare(processor)
    assert result.chunks[0].metadata["path"] == "source.txt"
    assert result.chunks[0].metadata["string_path"] == "string-source.txt"
    assert result.chunks[0].metadata["id"] == "12345678-1234-5678-1234-567812345678"
    assert result.chunks[0].metadata["api_key"] == "[redacted]"
    assert result.entities[0].properties["path"] == "entity"


@pytest.mark.asyncio
async def test_parser_timeout_becomes_timeout_without_retry(setup):
    processor, parser, _, _, _ = setup
    parser.error = asyncio.TimeoutError()
    with pytest.raises(ProcessingTimeoutError):
        await prepare(processor)
    assert len(parser.paths) == 1


@pytest.mark.asyncio
async def test_extractor_timeout_becomes_timeout_without_retry(setup):
    processor, _, extractor, _, _ = setup
    extractor.error = asyncio.TimeoutError()
    with pytest.raises(ProcessingTimeoutError):
        await prepare(processor)
    assert extractor.calls == 1


@pytest.mark.asyncio
async def test_parse_and_extract_errors_preserve_their_cause_type(setup):
    processor, parser, _, _, _ = setup
    parser.error = LookupError("parser")
    with pytest.raises(DocumentParseError) as parse_error:
        await prepare(processor)
    assert isinstance(parse_error.value.__cause__, LookupError)
    processor.parser.error = None
    processor.extractor.error = LookupError("extractor")
    with pytest.raises(KnowledgeExtractionError) as extract_error:
        await prepare(processor)
    assert isinstance(extract_error.value.__cause__, LookupError)


@pytest.mark.asyncio
async def test_processor_does_not_retain_document_body_in_audit_events(setup):
    processor, _, _, events, _ = setup
    await prepare(processor, content=b"unique secret document body")
    assert "unique secret document body" not in str(events)
    assert all("content" not in event for event in events)


@pytest.mark.asyncio
async def test_processor_has_no_storage_dependencies(setup):
    processor, _, _, _, _ = setup
    await prepare(processor)
    assert not any(name in vars(processor) for name in ("registry", "vector_store", "knowledge_graph", "mongo_client"))


@pytest.mark.asyncio
async def test_same_input_has_stable_serializable_result(setup):
    processor, _, _, _, _ = setup
    first = await prepare(processor)
    second = await prepare(processor)
    assert [(chunk.chunk_id, chunk.content, chunk.metadata) for chunk in first.chunks] == [
        (chunk.chunk_id, chunk.content, chunk.metadata) for chunk in second.chunks
    ]
    assert [(relation.head, relation.relation, relation.tail) for relation in first.relations] == [
        (relation.head, relation.relation, relation.tail) for relation in second.relations
    ]


@pytest.mark.asyncio
async def test_chunk_limit_is_enforced(setup):
    processor, _, _, _, _ = setup
    processor.max_chunks = 1
    processor.parser.chunks = [sample_chunk(0), sample_chunk(1)]
    with pytest.raises(InvalidExtractionResult, match="too many"):
        await prepare(processor)


@pytest.mark.asyncio
async def test_entity_and_relation_limits_are_enforced(setup):
    processor, _, extractor, _, _ = setup
    processor.max_entities = 1
    with pytest.raises(InvalidExtractionResult, match="exceeds"):
        await prepare(processor)
    processor.max_entities = 10
    processor.max_relations = 0
    with pytest.raises(InvalidExtractionResult, match="exceeds"):
        await prepare(processor)


@pytest.mark.asyncio
async def test_requested_unsupported_document_type_is_rejected(setup):
    processor, _, _, _, _ = setup
    with pytest.raises(UnsupportedDocumentType):
        await processor.prepare(
            content=b"content", filename="safe.txt", document_id="doc", version=1,
            content_hash="h", operation_id="op", doc_type="not-a-type"
        )
