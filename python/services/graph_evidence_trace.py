"""Evaluation-only safe provenance trace and optional ingestion journal.

This module deliberately contains no database, provider, or logging client.
It records safe edge identities and pipeline transitions only when an
evaluation runner explicitly supplies a collector.  Normal QA requests never
create a trace and therefore retain their existing behavior.  The optional
JSONL journal is explicitly gated for loopback evaluation ingestion only.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
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
_SAFE_TRACE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,120}$")
_JOURNAL_STAGES = frozenset({
    GraphTraceStage.EXTRACTED.value, GraphTraceStage.NORMALIZED.value,
    GraphTraceStage.PERSISTED.value,
})
_JOURNAL_WRITE_LOCK = threading.Lock()


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


class EvaluationTraceJournal:
    """Run-scoped, JSONL-only ingestion evidence journal.

    It has no provider, database, request ContextVar, or mutable "current
    run" global.  A context is constructed only by the internal evaluation
    upload path and each event carries the full run/operation/document key.
    """

    schema_version = "s4.7b-ingestion-journal-v1"

    def __init__(
        self, *, root: str | Path, run_id: str, operation_id: str,
        fixture_id: str | None = None,
    ) -> None:
        if not _SAFE_TRACE_RUN_ID.fullmatch(str(run_id or "")):
            raise ValueError("invalid evaluation trace run ID")
        self.run_id = str(run_id)
        self.operation_id = self._uuid(operation_id, "operation")
        self.fixture_id = _safe_text(fixture_id, 100) or None
        root_path = Path(root).resolve()
        self.run_dir = (root_path / self.run_id).resolve()
        if root_path not in self.run_dir.parents:
            raise ValueError("evaluation trace path escaped its root")
        self.path = self.run_dir / "graph-ingestion-trace.jsonl"
        self.lock_path = self.run_dir / ".graph-ingestion-trace.lock"

    @staticmethod
    def _uuid(value: object, kind: str) -> str:
        try:
            parsed = uuid.UUID(str(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid evaluation trace {kind} ID") from exc
        return str(parsed)

    @classmethod
    def _safe_event(cls, value: dict[str, Any]) -> dict[str, Any]:
        required = (
            "trace_schema_version", "run_id", "operation_id", "document_id",
            "document_version", "stage", "event_id", "timestamp",
        )
        if not isinstance(value, dict) or any(not value.get(key) for key in required):
            raise ValueError("invalid evaluation trace journal event")
        if value["trace_schema_version"] != cls.schema_version or not _SAFE_TRACE_RUN_ID.fullmatch(str(value["run_id"])):
            raise ValueError("incompatible evaluation trace journal event")
        cls._uuid(value["operation_id"], "operation")
        cls._uuid(value["document_id"], "document")
        if not isinstance(value["document_version"], int) or isinstance(value["document_version"], bool) or value["document_version"] < 1:
            raise ValueError("invalid evaluation trace version")
        if str(value["stage"]) not in _JOURNAL_STAGES | {GraphTraceStage.SCOPE_REJECTED.value}:
            raise ValueError("invalid evaluation trace stage")
        if value.get("event_kind") not in {"edge", "stage_observed", "rejection"}:
            raise ValueError("invalid evaluation trace event kind")
        return value

    def _append(self, event: dict[str, Any]) -> None:
        event = self._safe_event(event)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # The lock protects concurrent document tasks in this process.  Each
        # JSONL record is emitted in a single flushed append; no event depends
        # on process-global run state.
        encoded = (json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        with _JOURNAL_WRITE_LOCK:
            seen: set[str] = set()
            if self.path.exists():
                try:
                    with self.path.open("r", encoding="utf-8") as stream:
                        for line in stream:
                            item = json.loads(line)
                            if isinstance(item, dict) and item.get("event_id"):
                                seen.add(str(item["event_id"]))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise RuntimeError("evaluation trace journal is not readable") from exc
            if event["event_id"] in seen:
                return
            with self.path.open("ab", buffering=0) as stream:
                stream.write(encoded)
                os.fsync(stream.fileno())

    def _event(self, *, stage: str, edge: dict[str, Any] | None, reason: str | None = None) -> dict[str, Any]:
        if edge is None:
            raise ValueError("journal rejection requires a safe edge identity")
        captured = GraphTraceEdge.from_mapping(edge)
        document_id = self._uuid(captured.document_id, "document")
        if captured.document_version is None or captured.document_version < 1:
            raise ValueError("journal edge requires a document version")
        if not captured.source or not captured.evidence_key:
            raise ValueError("journal edge requires provenance")
        event_material = "\x1f".join((self.run_id, self.operation_id, stage, captured.fingerprint, str(reason or "")))
        return {
            "trace_schema_version": self.schema_version,
            "run_id": self.run_id,
            "operation_id": self.operation_id,
            "document_id": document_id,
            "document_version": captured.document_version,
            "fixture_id": self.fixture_id,
            "stage": stage,
            "event_kind": "rejection" if reason else "edge",
            "event_id": hashlib.sha256(event_material.encode("utf-8")).hexdigest()[:32],
            "edge_fingerprint": captured.fingerprint,
            "subject": captured.subject,
            "canonical_predicate": captured.predicate,
            "raw_predicate": captured.raw_predicate,
            "object": captured.object,
            "direction": captured.direction,
            "source": captured.source,
            "evidence_key": captured.evidence_key,
            "status": captured.status,
            "is_current": captured.is_current,
            "relation_semantics_version": captured.relation_semantics_version,
            "reason": _safe_text(reason, 80) or None,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def record_edges(self, stage: GraphTraceStage | str, edges: Iterable[dict[str, Any]]) -> list[GraphTraceEdge]:
        stage_name = GraphTraceStage(stage).value
        if stage_name not in _JOURNAL_STAGES:
            raise ValueError("journal only accepts ingestion stages")
        captured: list[GraphTraceEdge] = []
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            self._append(self._event(stage=stage_name, edge=edge))
            captured.append(GraphTraceEdge.from_mapping(edge))
        return captured

    def record_stage(
        self, stage: GraphTraceStage | str, *, document_id: str, document_version: int,
        source: str, status: str = "processing", is_current: bool = False,
    ) -> None:
        """Persist an observed empty-or-nonempty stage without document text.

        This marker is essential: a zero-edge extraction snapshot is evidence,
        while an absent snapshot remains unknown and must be fail-closed.
        """
        stage_name = GraphTraceStage(stage).value
        if stage_name not in _JOURNAL_STAGES:
            raise ValueError("journal only accepts ingestion stages")
        document_id = self._uuid(document_id, "document")
        if not isinstance(document_version, int) or isinstance(document_version, bool) or document_version < 1:
            raise ValueError("invalid evaluation trace version")
        source = _safe_text(source, 160).replace("\\", "/").rsplit("/", 1)[-1]
        if not source:
            raise ValueError("journal stage requires a safe source")
        material = "\x1f".join((self.run_id, self.operation_id, stage_name, document_id, str(document_version), "stage"))
        event = {
            "trace_schema_version": self.schema_version,
            "run_id": self.run_id,
            "operation_id": self.operation_id,
            "document_id": document_id,
            "document_version": document_version,
            "fixture_id": self.fixture_id,
            "stage": stage_name,
            "event_kind": "stage_observed",
            "event_id": hashlib.sha256(material.encode("utf-8")).hexdigest()[:32],
            "edge_fingerprint": f"stage:{stage_name}:{document_id}:{document_version}",
            "subject": "", "canonical_predicate": "", "raw_predicate": "", "object": "",
            "direction": "", "source": source, "evidence_key": "", "status": _safe_text(status, 32),
            "is_current": bool(is_current), "relation_semantics_version": "", "reason": None,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        self._append(event)

    def reject(
        self, reason: GraphTraceRejectionReason | str, *, edge: dict[str, Any] | None = None,
        stage: GraphTraceStage | str = GraphTraceStage.SCOPE_REJECTED, detail: str | None = None,
    ) -> None:
        # Dropped relations are represented only by their bounded edge identity
        # and finite reason; parser text, chunks and provider output never enter
        # the journal.  A malformed relation without enough identity is omitted.
        if edge is None:
            return
        stage_name = GraphTraceStage(stage).value
        if stage_name not in _JOURNAL_STAGES | {GraphTraceStage.SCOPE_REJECTED.value}:
            raise ValueError("invalid journal rejection stage")
        self._append(self._event(stage=stage_name, edge=edge, reason=GraphTraceRejectionReason(reason).value))

    @classmethod
    def read_events(cls, *, root: str | Path, run_id: str) -> list[dict[str, Any]]:
        probe = cls(root=root, run_id=run_id, operation_id=str(uuid.uuid4()))
        if not probe.path.exists():
            return []
        events: list[dict[str, Any]] = []
        seen: set[str] = set()
        try:
            with probe.path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    value = cls._safe_event(json.loads(line))
                    if value["run_id"] != run_id or value["event_id"] in seen:
                        raise ValueError("duplicate or foreign evaluation trace event")
                    seen.add(value["event_id"])
                    events.append(value)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError("evaluation trace journal is invalid") from exc
        return events

    @classmethod
    def merge_into_question_trace(
        cls, trace: GraphEvidenceTrace, *, root: str | Path, run_id: str,
        allowed_document_ids: frozenset[str],
    ) -> int:
        """Merge only exact allowlist ingestion evidence into one QA trace."""
        allowed = {cls._uuid(value, "document") for value in allowed_document_ids}
        if not allowed:
            raise ValueError("evaluation trace merge requires a non-empty document allowlist")
        merged = 0
        for event in cls.read_events(root=root, run_id=run_id):
            if event["document_id"] not in allowed or event["stage"] not in _JOURNAL_STAGES:
                continue
            if event["event_kind"] == "stage_observed":
                trace.question.stages.setdefault(event["stage"], [])
                continue
            if event["event_kind"] != "edge":
                continue
            edge = {
                "subject": event["subject"], "predicate": event["canonical_predicate"],
                "raw_predicate": event["raw_predicate"], "object": event["object"],
                "direction": event["direction"], "document_id": event["document_id"],
                "document_version": event["document_version"], "source": event["source"],
                "evidence_key": event["evidence_key"], "status": event.get("status", ""),
                "is_current": event.get("is_current"),
                "relation_semantics_version": event.get("relation_semantics_version", ""),
            }
            trace.record_edges(event["stage"], [edge])
            merged += 1
        return merged
