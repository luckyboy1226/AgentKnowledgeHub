"""Deterministic Parent–Child chunk artifacts used by ingestion only.

This module deliberately has no provider, database, or retrieval dependency.
``DocumentChunk`` remains the compatibility boundary for the existing vector
and knowledge-extraction paths; ``ChildChunk.to_document_chunk`` bridges the
new artifacts to that established interface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from agents.doc_parser_agent import DocType, DocumentChunk, StructuredBlock


_CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
_WORD = re.compile(r"[A-Za-z0-9_]+")


class UnicodeTokenEstimator:
    """Small deterministic estimate until a provider tokenizer is introduced.

    Each CJK character is counted independently and every contiguous latin,
    numeric, or underscore word counts once. The value is intentionally named
    ``estimated_token_count`` and must not be represented as provider tokens.
    """

    def count(self, text: str) -> int:
        rendered = str(text or "")
        cjk = len(_CJK.findall(rendered))
        without_cjk = _CJK.sub(" ", rendered)
        return cjk + len(_WORD.findall(without_cjk))


@dataclass(frozen=True)
class ParentChunk:
    parent_chunk_id: str
    document_id: str
    document_version: int
    parent_index: int
    content: str
    section_title: str | None
    page_number: int | None
    table_id: str | None
    estimated_token_count: int
    source: str
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChildChunk:
    chunk_id: str
    parent_chunk_id: str
    document_id: str
    document_version: int
    chunk_index: int
    child_index_in_parent: int
    content: str
    doc_type: DocType
    section_title: str | None
    page_number: int | None
    table_id: str | None
    estimated_token_count: int
    source: str
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_document_chunk(self) -> DocumentChunk:
        metadata = dict(self.metadata)
        metadata.update(
            {
                "source": self.source,
                "document_id": self.document_id,
                "document_version": self.document_version,
                "content_hash": self.content_hash,
                "chunk_index": self.chunk_index,
                "parent_chunk_id": self.parent_chunk_id,
                "child_index_in_parent": self.child_index_in_parent,
                "estimated_token_count": self.estimated_token_count,
            }
        )
        if self.section_title:
            metadata["section_title"] = self.section_title
        if self.page_number is not None:
            metadata["page_number"] = self.page_number
        if self.table_id:
            metadata["table_id"] = self.table_id
        return DocumentChunk(self.content, self.document_id, self.chunk_index, self.doc_type, metadata)


@dataclass(frozen=True)
class ChunkBundle:
    parents: list[ParentChunk]
    children: list[ChildChunk]


def _clean_text(value: str) -> str:
    return "\n".join(line.rstrip() for line in str(value or "").strip().splitlines()).strip()


def _cut_to_estimate(text: str, maximum: int, estimator: UnicodeTokenEstimator) -> list[str]:
    """Split only when needed; prefer whitespace/punctuation boundaries."""
    text = _clean_text(text)
    if not text:
        return []
    if estimator.count(text) <= maximum:
        return [text]
    parts: list[str] = []
    remaining = text
    while remaining:
        if estimator.count(remaining) <= maximum:
            parts.append(remaining)
            break
        # Character stepping is deterministic; adjust backwards to a natural
        # boundary without inventing semantic parsing.
        ceiling = min(len(remaining), max(1, maximum))
        cut = ceiling
        while cut > max(1, ceiling - 96) and estimator.count(remaining[:cut]) > maximum:
            cut -= 1
        boundary = max(
            (remaining.rfind(marker, 0, cut) for marker in ("\n", "。", "！", "？", ".", "!", "?", " ")),
            default=-1,
        )
        if boundary >= max(1, cut - 160):
            cut = boundary + 1
        piece = _clean_text(remaining[:cut])
        if not piece:
            cut = max(1, cut)
            piece = remaining[:cut]
        parts.append(piece)
        remaining = remaining[cut:].lstrip()
    return parts


def _paragraph_units(block: StructuredBlock, estimator: UnicodeTokenEstimator, maximum: int) -> Iterable[str]:
    text = _clean_text(block.content)
    if block.doc_type is DocType.TABLE:
        # The parser supplies a header plus bounded row group as one block.
        # Keep it intact unless it alone exceeds the configured parent limit.
        yield from _cut_to_estimate(text, maximum, estimator)
        return
    paragraphs = [part for part in re.split(r"\n\s*\n+", text) if part.strip()]
    for paragraph in paragraphs or [text]:
        yield from _cut_to_estimate(paragraph, maximum, estimator)


def build_parent_child_chunks(
    blocks: Iterable[StructuredBlock],
    *,
    document_id: str,
    document_version: int,
    content_hash: str,
    source: str,
    parent_target_tokens: int = 1000,
    parent_max_tokens: int = 1400,
    child_target_tokens: int = 250,
    child_overlap_tokens: int = 50,
    estimator: UnicodeTokenEstimator | None = None,
) -> ChunkBundle:
    """Build deterministic parent and child artifacts from structural blocks."""
    if min(parent_target_tokens, parent_max_tokens, child_target_tokens) < 1:
        raise ValueError("chunk targets must be positive")
    if parent_target_tokens > parent_max_tokens:
        raise ValueError("parent target cannot exceed parent maximum")
    if child_overlap_tokens < 0 or child_overlap_tokens >= child_target_tokens:
        raise ValueError("child overlap must be non-negative and smaller than child target")
    if not document_id or int(document_version) < 1 or not content_hash:
        raise ValueError("chunk identity is incomplete")
    estimator = estimator or UnicodeTokenEstimator()
    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []
    parent_index = 0
    child_index = 0

    def emit_parent(content: str, block: StructuredBlock) -> None:
        nonlocal parent_index, child_index
        content = _clean_text(content)
        if not content:
            return
        parent_id = f"{document_id}:v{int(document_version)}:p{parent_index}"
        parent = ParentChunk(
            parent_chunk_id=parent_id,
            document_id=document_id,
            document_version=int(document_version),
            parent_index=parent_index,
            content=content,
            section_title=block.section_title,
            page_number=block.page_number,
            table_id=block.table_id,
            estimated_token_count=estimator.count(content),
            source=source,
            content_hash=content_hash,
            metadata=dict(block.metadata),
        )
        parents.append(parent)
        child_parts = _split_children(content, child_target_tokens, child_overlap_tokens, estimator)
        for child_in_parent, child_text in enumerate(child_parts):
            child_id = f"{document_id}:v{int(document_version)}:c{child_index}"
            children.append(ChildChunk(
                chunk_id=child_id,
                parent_chunk_id=parent_id,
                document_id=document_id,
                document_version=int(document_version),
                chunk_index=child_index,
                child_index_in_parent=child_in_parent,
                content=child_text,
                doc_type=block.doc_type,
                section_title=block.section_title,
                page_number=block.page_number,
                table_id=block.table_id,
                estimated_token_count=estimator.count(child_text),
                source=source,
                content_hash=content_hash,
                metadata=dict(block.metadata),
            ))
            child_index += 1
        parent_index += 1

    for block in blocks:
        if not isinstance(block, StructuredBlock):
            raise TypeError("blocks must be StructuredBlock instances")
        buffer: list[str] = []
        buffer_count = 0
        for unit in _paragraph_units(block, estimator, parent_max_tokens):
            unit_count = estimator.count(unit)
            # Do not combine separate sections/pages/table groups. Within a
            # block prefer a target-sized parent but never exceed max.
            if buffer and (buffer_count + unit_count > parent_max_tokens or buffer_count >= parent_target_tokens):
                emit_parent("\n\n".join(buffer), block)
                buffer, buffer_count = [], 0
            buffer.append(unit)
            buffer_count += unit_count
        if buffer:
            emit_parent("\n\n".join(buffer), block)
    if not parents or not children:
        raise ValueError("structured chunking produced no content")
    return ChunkBundle(parents=parents, children=children)


def _split_children(text: str, target: int, overlap: int, estimator: UnicodeTokenEstimator) -> list[str]:
    """Stable child windows; overlap is approximate under the estimator."""
    text = _clean_text(text)
    if estimator.count(text) <= target:
        return [text]
    children: list[str] = []
    start = 0
    while start < len(text):
        # Find the largest substring at/under the target estimate.
        end = min(len(text), start + target)
        while end > start + 1 and estimator.count(text[start:end]) > target:
            end -= 1
        if end < len(text):
            boundary = max((text.rfind(marker, start, end) for marker in ("\n", "。", "！", "？", ".", "!", "?", " ")), default=-1)
            if boundary >= start + max(1, target - 96):
                end = boundary + 1
        item = _clean_text(text[start:end])
        if item:
            children.append(item)
        if end >= len(text):
            break
        next_start = max(start + 1, end - overlap)
        start = next_start
    return children
