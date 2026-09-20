"""Fake-only contracts for deterministic Parent–Child artifact creation."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.doc_parser_agent import DocParserAgent, DocType, DocumentChunk, StructuredBlock
from agents.knowledge_extract_agent import Entity, ExtractionResult, Relation
from services.chunk_models import UnicodeTokenEstimator, build_parent_child_chunks
from services.document_processor import DocumentProcessorAdapter


def build(blocks, *, document_id="doc-a", version=1):
    return build_parent_child_chunks(
        blocks,
        document_id=document_id,
        document_version=version,
        content_hash="a" * 64,
        source="safe.txt",
        parent_target_tokens=12,
        parent_max_tokens=18,
        child_target_tokens=5,
        child_overlap_tokens=1,
    )


def test_one_parent_can_generate_multiple_children_with_stable_provenance():
    bundle = build([StructuredBlock("甲乙丙丁戊己庚辛壬癸", DocType.TEXT)])
    assert len(bundle.parents) == 1 and len(bundle.children) > 1
    assert {item.parent_chunk_id for item in bundle.children} == {bundle.parents[0].parent_chunk_id}
    assert [item.chunk_id for item in bundle.children] == [f"doc-a:v1:c{index}" for index in range(len(bundle.children))]


def test_ids_are_deterministic_and_separate_document_versions():
    blocks = [StructuredBlock("中文内容" * 8, DocType.TEXT)]
    first = build(blocks)
    again = build(blocks)
    next_version = build(blocks, version=2)
    other_document = build(blocks, document_id="doc-b")
    assert first == again
    assert first.parents[0].parent_chunk_id != next_version.parents[0].parent_chunk_id
    assert first.children[0].chunk_id != other_document.children[0].chunk_id


def test_markdown_section_and_unicode_estimate_are_preserved():
    blocks = DocParserAgent._markdown_blocks("# 架构\n\n这里是中文正文。\n\n## 约束\n\n不可跨越。", DocType.MARKDOWN)
    bundle = build(blocks)
    assert [item.section_title for item in bundle.parents] == ["架构", "约束"]
    assert UnicodeTokenEstimator().count("中文 ABC def") == 4


def test_text_blocks_prefer_paragraph_boundaries_and_long_sections_fallback():
    blocks = DocParserAgent._text_blocks("第一段。\n\n第二段。\n\n" + "超长" * 20, DocType.TEXT)
    bundle = build_parent_child_chunks(
        blocks, document_id="doc-a", document_version=1, content_hash="a" * 64, source="safe.txt",
        parent_target_tokens=20, parent_max_tokens=30, child_target_tokens=8, child_overlap_tokens=1,
    )
    assert bundle.parents[0].content == "第一段。\n\n第二段。"
    assert all(parent.estimated_token_count <= 30 for parent in bundle.parents)


def test_parent_and_child_metadata_is_compatible_with_document_chunk():
    bundle = build([StructuredBlock("一二三四五六七八九十", DocType.TEXT, section_title="标题", page_number=2)])
    chunk = bundle.children[0].to_document_chunk()
    assert chunk.metadata["parent_chunk_id"] == bundle.parents[0].parent_chunk_id
    assert chunk.metadata["section_title"] == "标题"
    assert chunk.metadata["page_number"] == 2
    assert chunk.chunk_index == 0


class StructuredParser:
    async def parse_structured(self, _path):
        return [StructuredBlock("甲乙丙丁戊己庚辛壬癸" * 3, DocType.TEXT, section_title="测试")]


class Extractor:
    def __init__(self):
        self.chunks = []

    async def extract(self, chunks):
        self.chunks = chunks
        return [ExtractionResult([Entity("甲", "Person"), Entity("乙", "Concept")], [Relation("甲", "related", "乙", 1.0)], [])]


@pytest.mark.asyncio
async def test_processor_feature_flag_produces_parent_catalog_artifacts(tmp_path):
    extractor = Extractor()
    processor = DocumentProcessorAdapter(
        StructuredParser(), extractor, temp_root=tmp_path,
        parent_child_enabled=True, parent_target_tokens=12, parent_max_tokens=18,
        child_target_tokens=5, child_overlap_tokens=1,
    )
    prepared = await processor.prepare(content=b"fixture", filename="safe.txt", document_id="doc-a", version=2, content_hash="a" * 64, operation_id="op-a")
    assert prepared.parent_chunks and prepared.child_chunks
    assert all(chunk.metadata["parent_chunk_id"] for chunk in prepared.chunks)
    assert extractor.chunks == prepared.chunks
    assert prepared.processing_metadata["parent_child_enabled"] is True


class LegacyParser:
    async def parse(self, _path):
        return [DocumentChunk("legacy child", "parser-id", 0, DocType.TEXT, {})]


@pytest.mark.asyncio
async def test_processor_flag_false_keeps_v1_chunk_shape(tmp_path):
    processor = DocumentProcessorAdapter(LegacyParser(), Extractor(), temp_root=tmp_path, parent_child_enabled=False)
    prepared = await processor.prepare(content=b"fixture", filename="safe.txt", document_id="doc-a", version=1, content_hash="a" * 64, operation_id="op-a")
    assert prepared.parent_chunks == [] and prepared.child_chunks == []
    assert "parent_chunk_id" not in prepared.chunks[0].metadata
    assert prepared.processing_metadata["parent_child_enabled"] is False
