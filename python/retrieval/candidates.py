"""Stable, source-safe normalization for retrieval candidates."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal


RetrievalType = Literal["bm25", "vector", "graph"]


def safe_source(value: object) -> str:
    return str(value or "").replace("\\", "/").rsplit("/", 1)[-1][:180]


def _stable_digest(parts: Iterable[object]) -> str:
    material = "\x1f".join(str(part or "") for part in parts)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RetrievalCandidate:
    """Common candidate contract before Phase C fusion/reranking."""

    candidate_id: str
    content: str
    source: str
    document_id: str | None
    document_version: int | None
    chunk_id: str | None
    parent_chunk_id: str | None
    retrieval_type: RetrievalType
    raw_score: float | None
    rank: int | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id or self.retrieval_type not in {"bm25", "vector", "graph"}:
            raise ValueError("Retrieval candidate identity or type is invalid")


def child_candidate_id(metadata: dict[str, Any], content: str) -> str:
    """Prefer durable vector/chunk IDs; make legacy identity explicit."""
    chunk_id = metadata.get("chunk_id") or metadata.get("child_chunk_id")
    if chunk_id:
        return str(chunk_id)
    return "legacy-child:" + _stable_digest((
        metadata.get("document_id"), metadata.get("document_version"),
        metadata.get("source"), content,
    ))


def graph_candidate_id(record: dict[str, Any]) -> str:
    edges = record.get("evidence_edges") if isinstance(record.get("evidence_edges"), list) else []
    evidence_keys = [str(edge.get("evidence_key")) for edge in edges if isinstance(edge, dict) and edge.get("evidence_key")]
    if len(evidence_keys) == 1:
        return "graph:" + evidence_keys[0]
    if evidence_keys:
        return "graph-path:" + _stable_digest((record.get("document_id"), record.get("document_version"), *evidence_keys))
    return "graph-legacy:" + _stable_digest((
        record.get("document_id"), record.get("document_version"), record.get("source"),
        record.get("target"), record.get("relations"),
    ))


def from_vector_result(record: dict[str, Any], score: float | None, rank: int) -> RetrievalCandidate:
    metadata = dict(record.get("metadata") or {})
    content = str(record.get("content") or "")
    return RetrievalCandidate(
        candidate_id=child_candidate_id(metadata, content), content=content,
        source=safe_source(record.get("source") or metadata.get("source")),
        document_id=str(metadata["document_id"]) if metadata.get("document_id") is not None else None,
        document_version=int(metadata["document_version"]) if isinstance(metadata.get("document_version"), int) else None,
        chunk_id=str(metadata.get("chunk_id")) if metadata.get("chunk_id") else None,
        parent_chunk_id=str(metadata.get("parent_chunk_id")) if metadata.get("parent_chunk_id") else None,
        retrieval_type="vector", raw_score=float(score) if score is not None else None,
        rank=int(rank), metadata=metadata,
    )


def _structured_graph_content(record: dict[str, Any]) -> str:
    edges = record.get("evidence_edges") if isinstance(record.get("evidence_edges"), list) else []
    rendered: list[str] = []
    for edge in edges:
        if isinstance(edge, dict):
            rendered.append(f"{edge.get('subject', '')} --{edge.get('predicate', '')}--> {edge.get('object', '')}")
    if rendered:
        return "\n".join(rendered)
    relations = ", ".join(str(value) for value in (record.get("relations") or []))
    return f"{record.get('source', '')} --{relations}--> {record.get('target', '')}".strip()


def from_graph_record(record: dict[str, Any], rank: int, raw_score: float | None = None) -> RetrievalCandidate:
    metadata = dict(record)
    edges = metadata.get("evidence_edges") if isinstance(metadata.get("evidence_edges"), list) else []
    first = next((edge for edge in edges if isinstance(edge, dict)), {})
    document_id = first.get("document_id", metadata.get("document_id"))
    version = first.get("document_version", metadata.get("document_version"))
    source = first.get("source", metadata.get("provenance_source", metadata.get("source", "")))
    return RetrievalCandidate(
        candidate_id=graph_candidate_id(metadata), content=_structured_graph_content(metadata), source=safe_source(source),
        document_id=str(document_id) if document_id is not None else None,
        document_version=int(version) if isinstance(version, int) else None,
        chunk_id=None, parent_chunk_id=None, retrieval_type="graph",
        raw_score=float(raw_score) if raw_score is not None else None, rank=int(rank), metadata=metadata,
    )
