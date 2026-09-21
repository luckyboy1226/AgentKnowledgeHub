"""Paired, frozen-input RRF versus local-rerank benchmark contracts.

The adapter prepares one immutable RRF Top-20 pool per planned query.  Both
arms consume that exact pool; benchmark failures are explicit and never use
the online ContextBuilder fallback policy.
"""
from __future__ import annotations

import csv
import json
import math
import os
import statistics
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, Sequence

from retrieval.context_builder import ContextBuilderV2
from retrieval.fusion import FusedCandidate
from services.hybrid_v2_ab_benchmark import (
    ABCase, FrozenABInputs, atomic_json, atomic_text, percentile, stable_hash,
)

MODES = ("hybrid_v2_rrf_top20_no_rerank", "hybrid_v2_rrf_top20_local_bge_rerank")
SCHEMA_VERSION = "hybrid-v2-paired-rerank-ab-v1"
INPUT_TOP_K = 20
OUTPUT_TOP_K = 8


def validate_real_rerank_gate(
    inputs: FrozenABInputs, *, expected_fixture_fingerprint: str,
    expected_allowlist_hash: str, expected_query_plan_hash: str,
    expected_snapshot_root_hash: str, model_identity_hash: str,
    expected_model_identity_hash: str, device: str, input_top_k: int,
    output_top_k: int, planned_write_operations: int,
) -> None:
    """Pure validation used before any future read-only real runtime opens."""
    inputs.validate()
    observed = (inputs.fixture_fingerprint, inputs.allowlist_hash,
                inputs.query_plan_hash, inputs.snapshot_root_hash)
    expected = (expected_fixture_fingerprint, expected_allowlist_hash,
                expected_query_plan_hash, expected_snapshot_root_hash)
    if observed != expected:
        raise ValueError("paired_frozen_input_hash_mismatch")
    if model_identity_hash != expected_model_identity_hash:
        raise ValueError("paired_local_model_identity_mismatch")
    if device != "cpu":
        raise ValueError("paired_rerank_device_must_be_cpu")
    if input_top_k != INPUT_TOP_K or output_top_k != OUTPUT_TOP_K:
        raise ValueError("paired_rerank_top_k_mismatch")
    if planned_write_operations != 0:
        raise ValueError("paired_read_only_gate_failed")


@dataclass(frozen=True)
class PairedArmResult:
    candidate_ids: tuple[str, ...]
    document_ranks: tuple[str, ...]
    source_ids: tuple[str, ...]
    pre_rerank_ranks: dict[str, int]
    post_rerank_ranks: dict[str, int]
    rerank_scores: dict[str, float | None]
    rerank_used: bool
    rerank_failed: bool
    fallback_reason: str | None
    rerank_latency_ms: float
    retrieval_latency_ms: float
    call_audit: dict[str, int]


@dataclass(frozen=True)
class PairedQueryResult:
    rrf_candidate_ids: tuple[str, ...]
    no_rerank: PairedArmResult
    local_bge: PairedArmResult
    model_cold_load_latency_ms: float | None = None


class PairedRerankAdapter(Protocol):
    async def retrieve_pair(
        self, *, plan: dict[str, Any], query_ordinal: int,
        allowed_document_ids: frozenset[str], frozen_vector_sha256: str,
    ) -> PairedQueryResult: ...


class SharedPoolContextExecutor:
    """Apply two ContextBuilders to one already-fused immutable Top-20 pool."""
    def __init__(self, no_rerank_builder: ContextBuilderV2, rerank_builder: ContextBuilderV2):
        for builder in (no_rerank_builder, rerank_builder):
            if (builder.rerank_input_top_k != INPUT_TOP_K or
                    builder.rerank_output_top_k != OUTPUT_TOP_K or
                    builder.final_context_top_k != OUTPUT_TOP_K):
                raise ValueError("paired_context_builder_top_k_mismatch")
        if no_rerank_builder.rerank_enabled or not rerank_builder.rerank_enabled:
            raise ValueError("paired_context_builder_mode_mismatch")
        self.no_rerank_builder = no_rerank_builder
        self.rerank_builder = rerank_builder

    async def execute(self, *, query: str, pool: Sequence[FusedCandidate],
                      allowed_document_ids: frozenset[str], retrieval_latency_ms: float,
                      call_audit: dict[str, int]) -> PairedQueryResult:
        frozen_pool = tuple(pool)
        if len(frozen_pool) != INPUT_TOP_K or len({item.candidate_id for item in frozen_pool}) != INPUT_TOP_K:
            raise ValueError("paired_candidate_pool_not_top20")
        no = await self.no_rerank_builder.build(query, frozen_pool,
            allowed_document_ids=allowed_document_ids, strict_rerank=False)
        yes = await self.rerank_builder.build(query, frozen_pool,
            allowed_document_ids=allowed_document_ids, strict_rerank=True)
        expected_ids = tuple(item.candidate_id for item in frozen_pool)
        if tuple(item.candidate_id for item in no.reranked_candidates) != expected_ids:
            raise ValueError("paired_no_rerank_pool_changed")
        if {item.candidate_id for item in yes.reranked_candidates} != set(expected_ids):
            raise ValueError("paired_rerank_pool_changed")

        def arm(result, *, reranked: bool) -> PairedArmResult:
            selected = result.reranked_candidates[:OUTPUT_TOP_K]
            diagnostics = result.diagnostics
            return PairedArmResult(
                candidate_ids=tuple(item.candidate_id for item in selected),
                document_ranks=tuple(item.document_id for item in result.contexts if item.document_id),
                source_ids=tuple(item.source for item in result.contexts if item.source),
                pre_rerank_ranks={item.candidate_id: item.pre_rerank_rank for item in result.reranked_candidates},
                post_rerank_ranks={item.candidate_id: item.post_rerank_rank for item in result.reranked_candidates},
                rerank_scores={item.candidate_id: item.rerank_score for item in result.reranked_candidates},
                rerank_used=diagnostics.rerank_used,
                rerank_failed=False,
                fallback_reason=diagnostics.rerank_fallback_reason,
                rerank_latency_ms=float(diagnostics.rerank_latency_ms or 0.0),
                retrieval_latency_ms=float(retrieval_latency_ms) + float(diagnostics.rerank_latency_ms or 0.0),
                call_audit={**call_audit, "reranker": int(reranked)},
            )
        no_arm = arm(no, reranked=False)
        yes_arm = arm(yes, reranked=True)
        return PairedQueryResult(expected_ids, no_arm, yes_arm,
            yes.diagnostics.model_cold_load_latency_ms)


def _validate_arm(arm: PairedArmResult, pool: tuple[str, ...], *, reranked: bool) -> None:
    if len(pool) != INPUT_TOP_K or len(set(pool)) != INPUT_TOP_K:
        raise ValueError("paired_candidate_pool_not_top20")
    if len(arm.candidate_ids) > OUTPUT_TOP_K or len(set(arm.candidate_ids)) != len(arm.candidate_ids):
        raise ValueError("paired_output_candidate_identity_invalid")
    if not set(arm.candidate_ids).issubset(pool):
        raise ValueError("paired_output_outside_candidate_pool")
    if set(arm.pre_rerank_ranks) != set(pool):
        raise ValueError("paired_pre_rank_identity_mismatch")
    if reranked:
        if arm.rerank_failed or not arm.rerank_used or arm.fallback_reason:
            raise ValueError("rerank_failed")
        if set(arm.rerank_scores) != set(pool):
            raise ValueError("rerank_score_identity_mismatch")
        if any(value is None or not math.isfinite(float(value)) for value in arm.rerank_scores.values()):
            raise ValueError("rerank_score_non_finite")
    elif arm.rerank_used or arm.rerank_failed or arm.fallback_reason not in {None, "disabled"}:
        raise ValueError("no_rerank_arm_invalid")


def _metrics(ranked: Sequence[str], relevant: Sequence[str]) -> dict[str, float | None]:
    unique = list(dict.fromkeys(str(value) for value in ranked))
    relevant_set = set(relevant)
    if not relevant_set:
        return {"recall_at_1": None, "recall_at_3": None, "recall_at_5": None,
                "mrr_at_8": None, "ndcg_at_8": None}
    def recall(k: int) -> float:
        return round(len(set(unique[:k]) & relevant_set) / len(relevant_set), 6)
    first = next((index for index, value in enumerate(unique[:8], 1) if value in relevant_set), None)
    dcg = sum(1 / math.log2(index + 2) for index, value in enumerate(unique[:8]) if value in relevant_set)
    idcg = sum(1 / math.log2(index + 2) for index in range(min(8, len(relevant_set))))
    return {"recall_at_1": recall(1), "recall_at_3": recall(3), "recall_at_5": recall(5),
            "mrr_at_8": round(1 / first, 6) if first else 0.0,
            "ndcg_at_8": round(dcg / idcg, 6) if idcg else 0.0}


class DeterministicOfflinePairedAdapter:
    """Fake-only paired adapter; it never loads a model or service client."""
    def __init__(self, cases: Sequence[ABCase], *, failure: str | None = None):
        self.cases = {case.question_id: case for case in cases}
        self.failure = failure
        self.calls: list[tuple[int, int, str, frozenset[str]]] = []

    async def retrieve_pair(self, *, plan: dict[str, Any], query_ordinal: int,
                            allowed_document_ids: frozenset[str], frozen_vector_sha256: str) -> PairedQueryResult:
        self.calls.append((id(plan), query_ordinal, frozen_vector_sha256, allowed_document_ids))
        case = self.cases[str(plan["question_id"])]
        relevant = sorted(set(case.relevant_documents) & set(allowed_document_ids))
        pool_docs = (relevant + sorted(set(allowed_document_ids) - set(relevant)))[:INPUT_TOP_K]
        pool = tuple(f"candidate:{value}" for value in pool_docs)
        pre = {candidate_id: index for index, candidate_id in enumerate(pool, 1)}
        no_ids = pool[:OUTPUT_TOP_K]
        reranked_pool = tuple(reversed(pool))
        rerank_ids = reranked_pool[:OUTPUT_TOP_K]
        scores = {candidate_id: float(index) for index, candidate_id in enumerate(pool, 1)}
        sources = tuple(case.expected_sources if relevant else ())
        audit = {"vector": 1, "bm25": 1, "neo4j": 1, "reranker": 0, "chat": 0, "embedding": 0}
        no = PairedArmResult(no_ids, tuple(value.split(":", 1)[1] for value in no_ids), sources,
            pre, {value: index for index, value in enumerate(no_ids, 1)}, {value: None for value in pool},
            False, False, "disabled", 0.0, 3.0, audit)
        failed = self.failure is not None
        rerank = PairedArmResult(rerank_ids, tuple(value.split(":", 1)[1] for value in rerank_ids), sources,
            pre, {value: index for index, value in enumerate(rerank_ids, 1)}, scores,
            not failed, failed, self.failure, 4.0, 7.0, {**audit, "reranker": 1})
        return PairedQueryResult(pool, no, rerank, 125.0)


class PairedRerankABRunner:
    def __init__(self, *, inputs: FrozenABInputs, adapter: PairedRerankAdapter,
                 output_root: Path, ab_run_id: str, model_identity_hash: str):
        inputs.validate()
        if not ab_run_id or not model_identity_hash or len(model_identity_hash) != 64:
            raise ValueError("paired_ab_identity_invalid")
        self.inputs = inputs
        self.adapter = adapter
        self.directory = output_root / ab_run_id
        self.ab_run_id = ab_run_id
        self.model_identity_hash = model_identity_hash

    async def run(self) -> dict[str, Any]:
        if self.directory.exists():
            raise ValueError("paired_ab_output_directory_exists")
        rows: list[dict[str, Any]] = []
        cases = {case.question_id: case for case in self.inputs.cases}
        cold_load: float | None = None
        started = datetime.now(UTC).isoformat()
        def append_success(case: ABCase, ordinal: int, pair: PairedQueryResult,
                           mode: str, arm: PairedArmResult) -> None:
            deltas = {candidate_id: arm.pre_rerank_ranks[candidate_id] - arm.post_rerank_ranks[candidate_id]
                      for candidate_id in arm.candidate_ids}
            rows.append({"source_run_id": self.inputs.run_id, "ab_run_id": self.ab_run_id,
                "question_id": case.question_id, "query_ordinal": ordinal, "mode": mode,
                "success": True, "failure_summary": None, "rrf_candidate_ids": list(pair.rrf_candidate_ids),
                "candidate_ids": list(arm.candidate_ids), "document_ranks": list(arm.document_ranks),
                "source_ids": list(arm.source_ids), "pre_rerank_ranks": arm.pre_rerank_ranks,
                "post_rerank_ranks": arm.post_rerank_ranks, "rerank_scores_finite": all(
                    value is None or math.isfinite(float(value)) for value in arm.rerank_scores.values()),
                "rerank_used": arm.rerank_used, "rerank_failed": arm.rerank_failed,
                "fallback_reason": arm.fallback_reason, "rank_deltas": deltas,
                "metrics": _metrics(arm.document_ranks, case.relevant_documents),
                "source_coverage": round(len(set(arm.source_ids) & set(case.expected_sources)) /
                    len(set(case.expected_sources)), 6) if case.expected_sources else None,
                "zero_result": not bool(arm.candidate_ids), "rerank_latency_ms": arm.rerank_latency_ms,
                "retrieval_latency_ms": arm.retrieval_latency_ms, "call_audit": arm.call_audit,
                "timestamp": started})
        def append_failure(case: ABCase, ordinal: int, mode: str, exc: Exception) -> None:
            rows.append({"source_run_id": self.inputs.run_id, "ab_run_id": self.ab_run_id,
                "question_id": case.question_id, "query_ordinal": ordinal, "mode": mode,
                "success": False, "failure_summary": str(exc) if str(exc) else type(exc).__name__,
                "rerank_failed": mode == MODES[1], "timestamp": started})
        for plan in self.inputs.plans:
            case = cases[str(plan["question_id"])]
            for ordinal, query in enumerate(plan["queries"]):
                vector_sha = stable_hash({"snapshot_root": self.inputs.snapshot_root_hash,
                    "question_id": case.question_id, "ordinal": ordinal,
                    "query_sha256": __import__("hashlib").sha256(query.encode()).hexdigest()})
                try:
                    pair = await self.adapter.retrieve_pair(plan=plan, query_ordinal=ordinal,
                        allowed_document_ids=self.inputs.allowed_document_ids,
                        frozen_vector_sha256=vector_sha)
                    _validate_arm(pair.no_rerank, pair.rrf_candidate_ids, reranked=False)
                    if any(document_id not in self.inputs.allowed_document_ids for document_id in pair.no_rerank.document_ranks):
                        raise ValueError("paired_scope_violation")
                    append_success(case, ordinal, pair, MODES[0], pair.no_rerank)
                    if cold_load is None and pair.model_cold_load_latency_ms is not None:
                        cold_load = float(pair.model_cold_load_latency_ms)
                    try:
                        _validate_arm(pair.local_bge, pair.rrf_candidate_ids, reranked=True)
                        if pair.no_rerank.pre_rerank_ranks != pair.local_bge.pre_rerank_ranks:
                            raise ValueError("paired_candidate_pool_mismatch")
                        if any(document_id not in self.inputs.allowed_document_ids for document_id in pair.local_bge.document_ranks):
                            raise ValueError("paired_scope_violation")
                        append_success(case, ordinal, pair, MODES[1], pair.local_bge)
                    except Exception as exc:
                        append_failure(case, ordinal, MODES[1], exc)
                except Exception as exc:
                    append_failure(case, ordinal, "paired", exc)
        summary = summarize(rows, cold_load)
        payload = {"schema_version": SCHEMA_VERSION, "offline": isinstance(self.adapter, DeterministicOfflinePairedAdapter),
                   "results": rows}
        metadata = {"schema_version": SCHEMA_VERSION, "source_run_id": self.inputs.run_id,
            "ab_run_id": self.ab_run_id, "offline": payload["offline"], "input_top_k": INPUT_TOP_K,
            "output_top_k": OUTPUT_TOP_K, "result_count": len(rows), "model_identity_hash": self.model_identity_hash,
            "model_cold_load_latency_ms": cold_load, "external_calls": 0, "document_body_saved": False,
            "query_saved": False, "model_path_saved": False}
        physical_audit = getattr(self.adapter, "safe_call_audit", None)
        if isinstance(physical_audit, dict):
            metadata["physical_call_audit"] = {str(key): int(value) for key, value in physical_audit.items()}
        hashes = {"fixture_fingerprint": self.inputs.fixture_fingerprint,
            "allowlist_hash": self.inputs.allowlist_hash, "query_plan_hash": self.inputs.query_plan_hash,
            "query_embedding_snapshot_root_hash": self.inputs.snapshot_root_hash,
            "model_identity_hash": self.model_identity_hash}
        atomic_json(self.directory / "results.json", payload)
        _write_csv(self.directory / "results.csv", rows)
        atomic_json(self.directory / "safe-run-metadata.json", metadata)
        atomic_json(self.directory / "input-hashes.json", hashes)
        atomic_text(self.directory / "summary.md", _markdown(summary, metadata))
        return {"payload": payload, "summary": summary, "directory": self.directory}


def summarize(rows: Sequence[dict[str, Any]], cold_load_latency_ms: float | None) -> dict[str, Any]:
    result: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "model_cold_load_latency_ms": cold_load_latency_ms,
                              "modes": {}}
    for mode in MODES:
        planned = [row for row in rows if row.get("mode") == mode]
        subset = [row for row in planned if row.get("success")]
        mean = lambda values: round(statistics.mean(values), 6) if values else None
        metric = lambda name: [row["metrics"][name] for row in subset if row["metrics"][name] is not None]
        retrieval = [row["retrieval_latency_ms"] for row in subset]
        rerank = [row["rerank_latency_ms"] for row in subset if row["rerank_used"]]
        result["modes"][mode] = {"queries": len(planned), "successful": len(subset),
            "query_success_rate": round(len(subset) / len(planned), 6) if planned else None,
            **{name: mean(metric(name)) for name in ("recall_at_1", "recall_at_3", "recall_at_5", "mrr_at_8", "ndcg_at_8")},
            "source_coverage": mean([row["source_coverage"] for row in subset if row["source_coverage"] is not None]),
            "zero_result_rate": mean([float(row["zero_result"]) for row in subset]),
            "rerank_success_rate": mean([float(row["rerank_used"] and not row["rerank_failed"]) for row in subset]),
            "rerank_p50_latency_ms": percentile(rerank, .5), "rerank_p95_latency_ms": percentile(rerank, .95),
            "retrieval_p50_latency_ms": percentile(retrieval, .5), "retrieval_p95_latency_ms": percentile(retrieval, .95),
            "component_calls": {key: sum(int(row["call_audit"].get(key, 0)) for row in subset)
                                for key in ("vector", "bm25", "neo4j", "reranker", "chat", "embedding")}}
    successful_rows = [row for row in rows if row.get("success")]
    no = {(row["question_id"], row["query_ordinal"]): row for row in successful_rows if row["mode"] == MODES[0]}
    yes = {(row["question_id"], row["query_ordinal"]): row for row in successful_rows if row["mode"] == MODES[1]}
    keys = sorted(set(no) & set(yes))
    result["paired"] = {"pairs": len(keys),
        "changed_top1_rate": round(sum(no[key]["candidate_ids"][:1] != yes[key]["candidate_ids"][:1] for key in keys) / len(keys), 6) if keys else None,
        "changed_top8_rate": round(sum(no[key]["candidate_ids"] != yes[key]["candidate_ids"] for key in keys) / len(keys), 6) if keys else None,
        "mean_selected_rank_delta": mean_all([value for key in keys for value in yes[key]["rank_deltas"].values()])}
    return result


def mean_all(values: Sequence[float]) -> float | None:
    return round(statistics.mean(values), 6) if values else None


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    fields = ("source_run_id", "ab_run_id", "question_id", "query_ordinal", "mode", "success",
              "failure_summary", "zero_result", "rerank_used", "rerank_failed", "fallback_reason",
              "rerank_latency_ms", "retrieval_latency_ms", "source_coverage")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            writer.writerows({key: row.get(key) for key in fields} for row in rows)
        os.replace(name, path)
    except BaseException:
        try: os.unlink(name)
        except OSError: pass
        raise


def _markdown(summary: dict[str, Any], metadata: dict[str, Any]) -> str:
    offline = bool(metadata.get("offline"))
    title = "offline" if offline else "real read-only"
    note = ("Fake-only contract validation. This is not a retrieval-quality result."
            if offline else "Frozen-input local CPU retrieval result; no Chat or Embedding calls.")
    lines = [f"# Hybrid V2 paired rerank A/B ({title})", "", note, "",
        "| Mode | Queries | Recall@5 | MRR@8 | nDCG@8 | Retrieval P95 ms | Rerank P95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for mode, row in summary["modes"].items():
        lines.append(f"| {mode} | {row['queries']} | {row['recall_at_5']} | {row['mrr_at_8']} | {row['ndcg_at_8']} | {row['retrieval_p95_latency_ms']} | {row['rerank_p95_latency_ms']} |")
    lines.extend(["", f"Paired queries: {summary['paired']['pairs']}.",
                  f"Cold-load latency (separate): {metadata['model_cold_load_latency_ms']} ms.",
                  "External Chat/Embedding calls: 0."])
    return "\n".join(lines) + "\n"
