"""Evaluation-only, in-memory provenance trace for GraphRAG evidence.

This module deliberately contains no database, provider, or logging client.
It records safe edge identities and pipeline transitions only when an
evaluation runner explicitly supplies a collector.  Normal QA requests never
create a trace and therefore retain their existing behavior.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable


class GraphTraceStage(str, Enum):
    EXTRACTED = "extracted"
    NORMALIZED = "normalized"
    PERSISTED = "persisted"
    RETRIEVED_RAW = "retrieved_raw"
    SCOPE_ACCEPTED = "scope_accepted"
    SCOPE_REJECTED = "scope_rejected"
    RELEVANCE_SCORED = "relevance_scored"
    RANKED = "ranked"
    ENTERED_FINAL_TOP_K = "entered_final_top_k"
    ENTERED_PROMPT = "entered_prompt"


class GraphTraceRejectionReason(str, Enum):
    INVALID_RELATION = "invalid_relation"
    DANGLING_SUBJECT = "dangling_subject"
    DANGLING_OBJECT = "dangling_object"
    UNSAFE_PREDICATE = "unsafe_predicate"
    MISSING_PROVENANCE = "missing_provenance"
    OUTSIDE_ALLOWLIST = "outside_allowlist"
    INACTIVE = "inactive"
    NOT_READY = "not_ready"
    LEGACY_DISALLOWED = "legacy_disallowed"
    MALFORMED_RECORD = "malformed_record"
    ENTITY_QUERY_MISS = "entity_query_miss"
    PREDICATE_QUERY_MISS = "predicate_query_miss"
    LOW_RELEVANCE = "low_relevance"
    TOP_K_TRUNCATED = "top_k_truncated"
    DUPLICATE_EDGE = "duplicate_edge"
    DIRECTION_MISMATCH = "direction_mismatch"
    ENDPOINT_MISMATCH = "endpoint_mismatch"


_EDGE_FIELDS = (
    "subject", "predicate", "raw_predicate", "object", "direction",
    "document_id", "source", "evidence_key", "relation_semantics_version", "status",
)


def _safe_text(value: object, limit: int = 160) -> str:
    return " ".join(str(value or "").split())[:limit]


def edge_fingerprint(edge: dict[str, Any]) -> str:
    """Stable identity for an edge without retaining document text."""
    material = "\x1f".join(
        _safe_text(edge.get(field)) for field in (
            "subject", "predicate", "object", "direction", "document_id",
            "document_version", "evidence_key",
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class GraphTraceEdge:
    fingerprint: str
    subject: str
    predicate: str
    raw_predicate: str
    object: str
    direction: str
    document_id: str
    document_version: int | None
    evidence_key: str
    source: str
    status: str
    is_current: bool | None
    relation_semantics_version: str
    sources: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "GraphTraceEdge":
        normalized = {field: _safe_text(value.get(field)) for field in _EDGE_FIELDS}
        version = value.get("document_version")
        safe_version = version if isinstance(version, int) and not isinstance(version, bool) else None
        current = value.get("is_current")
        return cls(
            fingerprint=edge_fingerprint(value),
            subject=normalized["subject"], predicate=normalized["predicate"],
            raw_predicate=normalized["raw_predicate"], object=normalized["object"],
            direction=normalized["direction"], document_id=normalized["document_id"],
            document_version=safe_version, evidence_key=normalized["evidence_key"],
            source=normalized["source"].replace("\\", "/").rsplit("/", 1)[-1],
            status=normalized["status"], is_current=current if isinstance(current, bool) else None,
            relation_semantics_version=normalized["relation_semantics_version"],
            sources=(normalized["source"].replace("\\", "/").rsplit("/", 1)[-1],) if normalized["source"] else (),
        )


@dataclass(frozen=True)
class GraphTraceRejection:
    stage: str
    reason: str
    fingerprint: str | None = None
    detail: str | None = None


@dataclass
class GraphTraceQuestion:
    run_id: str
    question_id: str
    mode: str = "graph_rag"
    scope_verified: bool = False
    allowed_document_ids_count: int = 0
    stages: dict[str, list[GraphTraceEdge]] = field(default_factory=dict)
    rejections: list[GraphTraceRejection] = field(default_factory=list)
    relevance_scores: dict[str, float] = field(default_factory=dict)
    rank_positions: dict[str, int] = field(default_factory=dict)


class GraphEvidenceTrace:
    """Per-question trace collector. Instances are never shared globally."""

    def __init__(self, *, run_id: str, question_id: str, scope_verified: bool, allowed_document_ids_count: int) -> None:
        if not scope_verified or int(allowed_document_ids_count) < 1:
            raise ValueError("Graph evidence trace requires a verified nonempty evaluation scope")
        self.question = GraphTraceQuestion(
            run_id=str(run_id), question_id=str(question_id), scope_verified=bool(scope_verified),
            allowed_document_ids_count=int(allowed_document_ids_count),
        )

    @staticmethod
    def edges_from_record(record: dict[str, Any]) -> list[dict[str, Any]]:
        edges = record.get("evidence_edges") if isinstance(record, dict) else None
        return [edge for edge in edges if isinstance(edge, dict)] if isinstance(edges, list) else []

    def record_edges(self, stage: GraphTraceStage | str, edges: Iterable[dict[str, Any]]) -> list[GraphTraceEdge]:
        key = GraphTraceStage(stage).value
        captured_by_fingerprint: dict[str, GraphTraceEdge] = {}
        for raw in edges:
            if not isinstance(raw, dict):
                continue
            edge = GraphTraceEdge.from_mapping(raw)
            previous = captured_by_fingerprint.get(edge.fingerprint)
            if previous is None:
                captured_by_fingerprint[edge.fingerprint] = edge
            else:
                sources = tuple(sorted(set(previous.sources + edge.sources)))
                captured_by_fingerprint[edge.fingerprint] = GraphTraceEdge(
                    **{**asdict(previous), "sources": sources, "source": sources[0] if sources else previous.source}
                )
                self.reject(GraphTraceRejectionReason.DUPLICATE_EDGE, edge=raw, stage=key)
        captured = list(captured_by_fingerprint.values())
        if captured:
            existing = {edge.fingerprint: edge for edge in self.question.stages.setdefault(key, [])}
            for edge in captured:
                previous = existing.get(edge.fingerprint)
                if previous is not None:
                    sources = tuple(sorted(set(previous.sources + edge.sources)))
                    existing[edge.fingerprint] = GraphTraceEdge(
                        **{**asdict(previous), "sources": sources, "source": sources[0] if sources else previous.source}
                    )
                else:
                    existing[edge.fingerprint] = edge
            self.question.stages[key] = list(existing.values())
        return captured

    def record_record(self, stage: GraphTraceStage | str, record: dict[str, Any]) -> list[GraphTraceEdge]:
        return self.record_edges(stage, self.edges_from_record(record))

    def reject(
        self, reason: GraphTraceRejectionReason | str, *, edge: dict[str, Any] | None = None,
        stage: GraphTraceStage | str = GraphTraceStage.SCOPE_REJECTED, detail: str | None = None,
    ) -> None:
        self.question.rejections.append(GraphTraceRejection(
            stage=GraphTraceStage(stage).value, reason=GraphTraceRejectionReason(reason).value,
            fingerprint=edge_fingerprint(edge) if edge else None,
            detail=_safe_text(detail, 80) or None,
        ))

    def score(self, edges: Iterable[dict[str, Any]], score: float) -> None:
        captured = self.record_edges(GraphTraceStage.RELEVANCE_SCORED, edges)
        for edge in captured:
            self.question.relevance_scores[edge.fingerprint] = round(float(score), 6)

    def rank(self, edges: Iterable[dict[str, Any]], position: int) -> None:
        captured = self.record_edges(GraphTraceStage.RANKED, edges)
        for edge in captured:
            self.question.rank_positions[edge.fingerprint] = int(position)

    def to_dict(self) -> dict[str, Any]:
        def serialize(edge: GraphTraceEdge) -> dict[str, Any]:
            value = asdict(edge)
            value["sources"] = list(edge.sources)
            return value
        return {
            "run_id": self.question.run_id,
            "question_id": self.question.question_id,
            "mode": self.question.mode,
            "scope_verified": self.question.scope_verified,
            "allowed_document_ids_count": self.question.allowed_document_ids_count,
            "stages": {
                stage: [serialize(edge) for edge in edges]
                for stage, edges in sorted(self.question.stages.items())
            },
            "rejections": [asdict(item) for item in self.question.rejections],
            "relevance_scores": dict(sorted(self.question.relevance_scores.items())),
            "rank_positions": dict(sorted(self.question.rank_positions.items())),
        }
