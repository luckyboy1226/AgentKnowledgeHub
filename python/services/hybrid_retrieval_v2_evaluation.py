"""Phase F: deterministic, offline-only A/B evaluation for Retrieval V2.

This module intentionally has no provider, database, trace-store, or API imports.
It is a contract harness: real runs require a future, separately authorized adapter.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


VARIANTS = ("vector_only", "bm25_vector_rrf", "vector_graph_rrf", "hybrid_v2_full")
GRAPH_VARIANTS = frozenset(("vector_graph_rrf", "hybrid_v2_full"))
_SAFE_ID = __import__("re").compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")


@dataclass(frozen=True)
class FakeQuestion:
    question_id: str
    category: str
    relevant_document_ids: tuple[str, ...]
    expected_edges: tuple[tuple[str, str, str, str], ...] = ()
    answerable: bool = True


# This fixture is deliberately small and is not a claim about production quality.
FAKE_BENCHMARK = (
    FakeQuestion("F01", "exact_identifier", ("D-E1007",)),
    FakeQuestion("F02", "semantic_paraphrase", ("D-SEM",)),
    FakeQuestion("F03", "multi_hop", ("D-PATH",), (("A", "DEPENDS_ON", "B", "forward"), ("B", "RESPONSIBLE_FOR", "C", "forward"))),
    FakeQuestion("F04", "version_conflict", ("D-CURRENT",)),
    FakeQuestion("F05", "unanswerable", (), answerable=False),
    FakeQuestion("F06", "single_hop", ("D-SINGLE",)),
    FakeQuestion("F07", "cross_document", ("D-CROSS-A", "D-CROSS-B")),
    FakeQuestion("F08", "distractor", ("D-DISTRACTOR",)),
)


def atomic_json(path: Path, payload: Any) -> None:
    """Atomically persist JSON without ever overwriting through a partial write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(name, path)
    except BaseException:
        try:
            os.unlink(name)
        except OSError:
            pass
        raise


def _atomic_text(path: Path, value: str) -> None:
    """Use the same replacement discipline for the human-readable summary."""
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
        os.replace(name, path)
    except BaseException:
        try:
            os.unlink(name)
        except OSError:
            pass
        raise


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p
    low, high = math.floor(pos), math.ceil(pos)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (pos - low), 3)


def _edges_match(found: Iterable[tuple[str, str, str, str]], expected: Iterable[tuple[str, str, str, str]]) -> bool:
    # Tuple includes canonical predicate and direction: endpoint-only matches are rejected.
    return set(expected).issubset(set(found))


def _rrf(*lists: list[str]) -> list[str]:
    scores: dict[str, float] = {}
    for candidates in lists:
        for rank, candidate in enumerate(candidates, 1):
            scores[candidate] = scores.get(candidate, 0) + 1 / (60 + rank)
    return sorted(scores, key=lambda candidate: (-scores[candidate], candidate))


def _retrieved(question: FakeQuestion, variant: str) -> tuple[list[str], list[tuple[str, str, str, str]], dict[str, Any]]:
    relevant = list(question.relevant_document_ids)
    fillers = [f"N{i:02d}" for i in range(1, 12)]
    vector = relevant + fillers
    bm25 = relevant + fillers
    graph_edges: list[tuple[str, str, str, str]] = []
    if question.question_id == "F01":  # vector intentionally weak on an identifier
        vector = fillers[:8] + relevant + fillers[8:]
        bm25 = relevant + fillers
    if question.question_id == "F03":
        vector = fillers[:8] + relevant + fillers[8:]
        graph_edges = list(question.expected_edges)
    if question.question_id == "F05":
        vector, bm25 = fillers, fillers
    pre_rerank_rank: int | None = None
    if variant == "vector_only":
        result = vector
    elif variant == "bm25_vector_rrf":
        result = _rrf(bm25, vector)
    elif variant == "vector_graph_rrf":
        graph_docs = relevant if graph_edges else []
        result = _rrf(vector, graph_docs)
    else:
        graph_docs = relevant if graph_edges else []
        result = _rrf(bm25, vector, graph_docs)
        pre_rerank_rank = next((index + 1 for index, item in enumerate(result) if item in relevant), None)
        # deterministic fake reranker demonstrates both an improvement fixture and retained provenance.
        if question.question_id in {"F01", "F03"}:
            result = relevant + [item for item in result if item not in relevant]
    diag = {"bm25_count": len(bm25) if variant in {"bm25_vector_rrf", "hybrid_v2_full"} else None,
            "vector_count": len(vector), "graph_count": len(graph_edges) if variant in GRAPH_VARIANTS else None}
    diag["pre_rerank_relevant_rank"] = pre_rerank_rank
    return result, graph_edges if variant in GRAPH_VARIANTS else [], diag


def _metrics(retrieved: list[str], relevant: tuple[str, ...], final_k: int) -> dict[str, float | None]:
    if not relevant:
        return {"recall_at_5": None, "recall_at_10": None, "mrr": None, "ndcg_at_10": None}
    ranks = [index + 1 for index, item in enumerate(retrieved) if item in relevant]
    def recall(k: int) -> float:
        return round(sum(1 for item in retrieved[:k] if item in relevant) / len(relevant), 6)
    rr = 1 / min(ranks) if ranks else 0.0
    dcg = sum(1 / math.log2(index + 2) for index, item in enumerate(retrieved[:10]) if item in relevant)
    idcg = sum(1 / math.log2(index + 2) for index in range(min(10, len(relevant))))
    return {"recall_at_5": recall(5), "recall_at_10": recall(10), "mrr": round(rr, 6), "ndcg_at_10": round(dcg / idcg if idcg else 0.0, 6)}


def _failure(question: FakeQuestion, retrieved: list[str], final: list[str], edges: list[tuple[str, str, str, str]], variant: str) -> str:
    if not question.answerable:
        return "unsupported_answer" if final else "insufficient_evidence"
    if not any(item in question.relevant_document_ids for item in retrieved):
        return "retrieval_miss"
    if variant in GRAPH_VARIANTS and question.expected_edges and not _edges_match(edges, question.expected_edges):
        return "graph_edge_missing"
    if not any(item in question.relevant_document_ids for item in final):
        return "budget_drop"
    return "insufficient_evidence"


def _row(question: FakeQuestion, variant: str, final_k: int, run_id: str) -> dict[str, Any]:
    started = time.monotonic()
    retrieved, edges, diag = _retrieved(question, variant)
    # Values are deterministic stage accounting, not wall-clock performance claims.
    latency = {"query_plan_ms": 0.1, "bm25_ms": 0.2 if diag["bm25_count"] is not None else None,
               "vector_ms": 0.2, "graph_ms": 0.2 if variant in GRAPH_VARIANTS else None,
               "rrf_ms": 0.1 if variant != "vector_only" else None,
               "rerank_ms": 0.1 if variant == "hybrid_v2_full" else None,
               "parent_expand_ms": 0.1 if variant == "hybrid_v2_full" else None,
               "budget_ms": 0.1 if variant == "hybrid_v2_full" else None}
    final = retrieved[:final_k]
    latency["retrieval_total_ms"] = round(sum(value for value in latency.values() if isinstance(value, float)), 3)
    graph_applicable = variant in GRAPH_VARIANTS
    graph = {"applicable": graph_applicable,
             "graph_participation_rate": (1.0 if edges else 0.0) if graph_applicable else None,
             "expected_edge_coverage": (sum(edge in edges for edge in question.expected_edges) / len(question.expected_edges) if question.expected_edges else None) if graph_applicable else None,
             "complete_path_coverage": (_edges_match(edges, question.expected_edges) if question.expected_edges else None) if graph_applicable else None,
             "graph_context_in_final_top_k": bool(edges and any(item in question.relevant_document_ids for item in final)) if graph_applicable else None}
    pre_rank = diag.get("pre_rerank_relevant_rank")
    post_rank = next((index + 1 for index, item in enumerate(final) if item in question.relevant_document_ids), None)
    diagnostics = {**diag, "unique_candidate_count": len(set(retrieved)), "multi_source_candidate_rate": 0.0 if variant == "vector_only" else 1.0,
                   "rerank_used": variant == "hybrid_v2_full", "rerank_fallback": False if variant == "hybrid_v2_full" else None,
                   "mean_relevant_rank_delta": (post_rank - pre_rank) if variant == "hybrid_v2_full" and pre_rank and post_rank else None,
                   "parent_expand_attempt_rate": 1.0 if variant == "hybrid_v2_full" else None,
                   "parent_expand_success_rate": 1.0 if variant == "hybrid_v2_full" else None,
                   "legacy_child_fallback_rate": 0.0 if variant == "hybrid_v2_full" else None,
                   "parent_missing_rate": 0.0 if variant == "hybrid_v2_full" else None,
                   "parent_to_child_budget_fallback_rate": 0.0 if variant == "hybrid_v2_full" else None}
    return {"run_id": run_id, "request_id": f"{run_id}:{question.question_id}:{variant}", "question_id": question.question_id, "variant": variant, "category": question.category,
            "scope": {"verified": True, "allowed_document_ids": ["fake-corpus-v1"], "fail_closed": True},
            "retrieval_metrics": _metrics(retrieved, question.relevant_document_ids, final_k), "graph_metrics": graph,
            "latency": latency, "final_context_ids": final, "source_ids": final,
            "answer_metrics": {"applicable": False, "abstention_correctness": True if not question.answerable and not final else None},
            "diagnostics": diagnostics, "failure_bucket": _failure(question, retrieved, final, edges, variant),
            "_elapsed_unused": round((time.monotonic() - started) * 1000, 3)}


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def group(items: list[dict[str, Any]]) -> dict[str, Any]:
        variants: dict[str, Any] = {}
        for variant in VARIANTS:
            subset = [row for row in items if row["variant"] == variant]
            if not subset:
                continue
            metric = lambda name: [row["retrieval_metrics"][name] for row in subset if row["retrieval_metrics"][name] is not None]
            latencies = [row["latency"]["retrieval_total_ms"] for row in subset]
            variants[variant] = {"questions": len(subset), "document_recall_at_5": _mean(metric("recall_at_5")),
                "document_recall_at_10": _mean(metric("recall_at_10")), "mrr": _mean(metric("mrr")), "ndcg_at_10": _mean(metric("ndcg_at_10")),
                "p50_retrieval_latency_ms": _percentile(latencies, .5), "p95_retrieval_latency_ms": _percentile(latencies, .95),
                "small_sample": len(subset) < 30}
        return variants
    output = {"overall": group(rows)}
    for category in sorted({row["category"] for row in rows}):
        output[category] = group([row for row in rows if row["category"] == category])
    return output


def _mean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return round(statistics.mean(clean), 6) if clean else None


def run_fake_evaluation(output_root: Path, run_id: str, variants: Iterable[str] = VARIANTS, final_k: int = 8, *, retrieval_trace: bool = False, graph_trace: bool = False) -> dict[str, Any]:
    """Run/resume the deterministic fixture. Existing question/variant rows are immutable."""
    requested = tuple(variants)
    if not _SAFE_ID.fullmatch(run_id) or not requested or any(item not in VARIANTS for item in requested) or final_k < 1:
        raise ValueError("invalid_fake_evaluation_request")
    directory = output_root / run_id
    if directory.exists() and not directory.is_dir():
        raise ValueError("run_id_path_conflict")
    prior: list[dict[str, Any]] = []
    results_path = directory / "results.json"
    if results_path.exists():
        prior_payload = json.loads(results_path.read_text(encoding="utf-8"))
        prior = list(prior_payload.get("results", [])) if isinstance(prior_payload, dict) else []
    seen = {(row.get("question_id"), row.get("variant")) for row in prior if isinstance(row, dict)}
    rows = prior + [_row(question, variant, final_k, run_id) for question in FAKE_BENCHMARK for variant in requested if (question.question_id, variant) not in seen]
    rows.sort(key=lambda row: (row["question_id"], row["variant"]))
    payload = {"schema_version": "phase-f-v1", "run_id": run_id, "fake_only": True, "relevance_level": "document", "final_top_k": final_k, "results": rows}
    summary = _summary(rows)
    atomic_json(results_path, payload)
    atomic_json(directory / "summary.json", summary)
    lines = ["# Hybrid Retrieval V2 Fake Evaluation", "", "This is deterministic fixture validation, not a real benchmark.", "", "| Variant | Document Recall@10 | MRR | P50 ms |", "|---|---:|---:|---:|"]
    for variant, item in summary["overall"].items():
        lines.append(f"| {variant} | {item['document_recall_at_10']} | {item['mrr']} | {item['p50_retrieval_latency_ms']} |")
    _atomic_text(directory / "summary.md", "\n".join(lines) + "\n")
    # Traces are diagnostic observers. They contain IDs/counts only and are
    # generated after the primary result, so toggling them cannot affect ranks.
    if retrieval_trace:
        atomic_json(directory / "retrieval-trace.json", [{"request_id": row["request_id"], "variant": row["variant"], "final_context_count": len(row["final_context_ids"])} for row in rows])
    if graph_trace:
        atomic_json(directory / "graph-trace.json", [{"request_id": row["request_id"], "variant": row["variant"], "applicable": True, "complete_path_coverage": row["graph_metrics"]["complete_path_coverage"]} for row in rows if row["variant"] in GRAPH_VARIANTS])
    return payload
