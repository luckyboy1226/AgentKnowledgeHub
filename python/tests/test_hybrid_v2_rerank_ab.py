import asyncio
import json
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from retrieval.candidates import RetrievalCandidate
from retrieval.context_builder import ContextBuilderV2
from retrieval.fusion import RRFFusion
from retrieval.parent_expander import ParentExpander
from retrieval.reranker import ConfigurableModelReranker, FakeReranker, RerankScore, RerankerUnavailableError
from services.hybrid_v2_ab_benchmark import offline_inputs
from services.hybrid_v2_rerank_ab import (
    DeterministicOfflinePairedAdapter, INPUT_TOP_K, MODES, PairedArmResult,
    PairedQueryResult, PairedRerankABRunner, _markdown, _validate_arm,
    SharedPoolContextExecutor, validate_real_rerank_gate,
)

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmarks" / "enterprise_20docs_retrieval_v2"
SCRIPT = ROOT / "scripts" / "run-hybrid-v2-rerank-ab.py"
MODEL_HASH = "a" * 64


def frozen():
    return offline_inputs(BENCHMARK, run_id="paired-unit")


def run(tmp_path, adapter=None, *, inputs=None, name="paired"):
    current = inputs or frozen()
    return asyncio.run(PairedRerankABRunner(inputs=current,
        adapter=adapter or DeterministicOfflinePairedAdapter(current.cases),
        output_root=tmp_path, ab_run_id=name, model_identity_hash=MODEL_HASH).run())


def test_pair_uses_one_exact_top20_pool_and_emits_360_rows(tmp_path):
    current = frozen(); adapter = DeterministicOfflinePairedAdapter(current.cases)
    result = run(tmp_path, adapter, inputs=current)
    assert len(adapter.calls) == 180 and len(result["payload"]["results"]) == 360
    grouped = {}
    for row in result["payload"]["results"]:
        grouped.setdefault((row["question_id"], row["query_ordinal"]), []).append(row)
        assert len(row["rrf_candidate_ids"]) == 20
    assert all(len(items) == 2 and items[0]["rrf_candidate_ids"] == items[1]["rrf_candidate_ids"] for items in grouped.values())


def test_no_rerank_preserves_rrf_order_and_fake_reranker_changes_it(tmp_path):
    rows = run(tmp_path)["payload"]["results"]
    no = next(row for row in rows if row["mode"] == MODES[0])
    yes = next(row for row in rows if row["mode"] == MODES[1])
    assert no["candidate_ids"] == no["rrf_candidate_ids"][:8]
    assert yes["candidate_ids"] != no["candidate_ids"]


def test_shared_pool_executor_runs_parent_and_budget_pipeline_on_same_pool():
    class Parents:
        def get_parent(self, *_args): return None
    retrieval = [RetrievalCandidate(candidate_id=f"c{i:02d}", content=f"text-{i}", source=f"D{i:02d}.txt",
        document_id=f"D{i:02d}", document_version=1, chunk_id=f"c{i:02d}", parent_chunk_id=None,
        retrieval_type="bm25", raw_score=1.0, rank=i,
        metadata={"status": "ready", "is_current": True, "scope_accepted": True}) for i in range(1, 21)]
    pool = RRFFusion(fusion_top_k=20).fuse({"bm25": retrieval, "vector": [], "graph": []}).candidates
    common = {"parent_expander": ParentExpander(Parents()), "parent_expansion_enabled": True,
        "rerank_input_top_k": 20, "rerank_output_top_k": 8, "final_context_top_k": 8}
    no_builder = ContextBuilderV2(**common)
    scores = {item.candidate_id: float(index) for index, item in enumerate(pool, 1)}
    yes_builder = ContextBuilderV2(**common, reranker=FakeReranker(scores), rerank_enabled=True)
    pair = asyncio.run(SharedPoolContextExecutor(no_builder, yes_builder).execute(query="q", pool=pool,
        allowed_document_ids=frozenset(f"D{i:02d}" for i in range(1, 21)), retrieval_latency_ms=2.0,
        call_audit={"vector": 1, "bm25": 1, "neo4j": 1, "chat": 0, "embedding": 0}))
    assert pair.rrf_candidate_ids == tuple(item.candidate_id for item in pool)
    assert pair.no_rerank.candidate_ids == pair.rrf_candidate_ids[:8]
    assert pair.local_bge.candidate_ids != pair.no_rerank.candidate_ids
    assert len(pair.no_rerank.document_ranks) == len(pair.local_bge.document_ranks) == 8


def test_equal_rerank_scores_use_candidate_id_as_stable_final_tie_break():
    candidates = [RetrievalCandidate(candidate_id=f"c-{value}", content=str(value), source="s",
        document_id="doc", document_version=1, chunk_id=str(value), parent_chunk_id=None,
        retrieval_type="bm25", raw_score=1.0, rank=1,
        metadata={"status": "ready", "is_current": True, "scope_accepted": True}) for value in ("z", "a")]
    fused = RRFFusion(fusion_top_k=20).fuse({"bm25": candidates, "vector": [], "graph": []}).candidates
    reranker = FakeReranker({item.candidate_id: 1.0 for item in fused})
    result = asyncio.run(reranker.rerank("q", list(fused), 8))
    assert [item.candidate_id for item in result.candidates] == sorted(item.candidate_id for item in fused)


@pytest.mark.parametrize("kind", ["missing", "nonfinite"])
def test_benchmark_rejects_missing_or_nonfinite_rerank_scores(kind):
    pool = tuple(f"c{i:02d}" for i in range(20)); pre = {value: index for index, value in enumerate(pool, 1)}
    scores = {value: float(index) for index, value in enumerate(pool, 1)}
    if kind == "missing": scores.pop(pool[-1])
    else: scores[pool[-1]] = math.nan
    arm = PairedArmResult(pool[:8], tuple("doc" for _ in range(8)), (), pre,
        {value: index for index, value in enumerate(pool[:8], 1)}, scores, True, False, None, 1, 2, {})
    with pytest.raises(ValueError): _validate_arm(arm, pool, reranked=True)


def test_rerank_timeout_is_explicit_failure_while_no_rerank_remains_success(tmp_path):
    current = frozen(); result = run(tmp_path, DeterministicOfflinePairedAdapter(current.cases, failure="rerank_timeout"), inputs=current)
    no = [row for row in result["payload"]["results"] if row["mode"] == MODES[0]]
    yes = [row for row in result["payload"]["results"] if row["mode"] == MODES[1]]
    assert len(no) == 180 and all(row["success"] for row in no)
    assert len(yes) == 180 and all(not row["success"] and row["rerank_failed"] for row in yes)


def test_online_context_builder_fallback_remains_available():
    class Failing:
        async def score(self, request): raise RerankerUnavailableError("offline")
    class Parents:
        def get_parent(self, *_args): return None
    candidate = RetrievalCandidate(candidate_id="c", content="text", source="s", document_id="doc",
        document_version=1, chunk_id="c", parent_chunk_id=None, retrieval_type="bm25",
        raw_score=1.0, rank=1, metadata={"status": "ready", "is_current": True, "scope_accepted": True})
    fused = RRFFusion(fusion_top_k=20).fuse({"bm25": [candidate], "vector": [], "graph": []}).candidates
    builder = ContextBuilderV2(parent_expander=ParentExpander(Parents()),
        reranker=ConfigurableModelReranker(Failing(), max_attempts=1), rerank_enabled=True)
    result = asyncio.run(builder.build("q", fused, allowed_document_ids=frozenset({"doc"})))
    assert not result.diagnostics.rerank_used and result.diagnostics.rerank_fallback_reason
    with pytest.raises(RerankerUnavailableError):
        asyncio.run(builder.build("q", fused, allowed_document_ids=frozenset({"doc"}), strict_rerank=True))


@pytest.mark.parametrize("field", ["snapshot_root_hash", "allowlist_hash", "query_plan_hash"])
def test_frozen_hash_mismatch_fails_closed(field, tmp_path):
    with pytest.raises(ValueError):
        run(tmp_path, inputs=replace(frozen(), **{field: "0" * 64}))


@pytest.mark.parametrize("change,match", [
    ({"expected_snapshot_root_hash": "0" * 64}, "hash_mismatch"),
    ({"expected_model_identity_hash": "b" * 64}, "model_identity_mismatch"),
    ({"device": "cuda"}, "device_must_be_cpu"),
    ({"input_top_k": 30}, "top_k_mismatch"),
    ({"planned_write_operations": 1}, "read_only_gate_failed"),
])
def test_real_preflight_gate_fails_closed(change, match):
    current = frozen()
    values = {"expected_fixture_fingerprint": current.fixture_fingerprint,
        "expected_allowlist_hash": current.allowlist_hash,
        "expected_query_plan_hash": current.query_plan_hash,
        "expected_snapshot_root_hash": current.snapshot_root_hash,
        "model_identity_hash": MODEL_HASH, "expected_model_identity_hash": MODEL_HASH,
        "device": "cpu", "input_top_k": 20, "output_top_k": 8, "planned_write_operations": 0}
    values.update(change)
    with pytest.raises(ValueError, match=match): validate_real_rerank_gate(current, **values)


def test_scope_outside_candidate_is_rejected(tmp_path):
    class Foreign(DeterministicOfflinePairedAdapter):
        async def retrieve_pair(self, **kwargs):
            pair = await super().retrieve_pair(**kwargs)
            bad = replace(pair.local_bge, document_ranks=("foreign",) + pair.local_bge.document_ranks[1:])
            return replace(pair, local_bge=bad)
    current = frozen(); rows = run(tmp_path, Foreign(current.cases), inputs=current)["payload"]["results"]
    reranked = [row for row in rows if row["mode"] == MODES[1]]
    assert reranked and all(not row["success"] and row["failure_summary"] == "paired_scope_violation" for row in reranked)


def test_cold_load_latency_is_separate_from_query_latency(tmp_path):
    result = run(tmp_path)
    assert result["summary"]["model_cold_load_latency_ms"] == 125.0
    assert result["summary"]["modes"][MODES[1]]["rerank_p50_latency_ms"] == 4.0


def test_outputs_are_safe_parseable_and_do_not_load_model_or_clients(tmp_path):
    before = set(sys.modules); result = run(tmp_path); added = set(sys.modules) - before
    directory = result["directory"]
    assert {item.name for item in directory.iterdir()} == {"results.json", "results.csv", "summary.md", "safe-run-metadata.json", "input-hashes.json"}
    assert len(json.loads((directory / "results.json").read_text(encoding="utf-8"))["results"]) == 360
    combined = "\n".join(item.read_text(encoding="utf-8") for item in directory.iterdir())
    assert "D:\\" not in combined and "API_KEY" not in combined and "Authorization" not in combined
    assert not any(name.startswith(("pymongo", "neo4j", "chromadb", "transformers", "torch")) for name in added)


def test_real_cli_refuses_without_authorization_or_service_access():
    completed = subprocess.run([sys.executable, str(SCRIPT), "--runtime", "real", "--run-id", "x"], capture_output=True, text=True)
    assert completed.returncode == 2 and "explicit authorization flags" in completed.stderr


def test_existing_output_and_baseline_bytes_are_never_overwritten(tmp_path):
    baseline = tmp_path / "baseline.bin"; baseline.write_bytes(b"immutable")
    run(tmp_path, name="first")
    with pytest.raises(ValueError, match="output_directory_exists"): run(tmp_path, name="first")
    assert baseline.read_bytes() == b"immutable"


def test_markdown_distinguishes_offline_from_real_read_only_runs():
    summary = {"modes": {}, "paired": {"pairs": 0}}
    assert "fake-only" in _markdown(summary, {"offline": True, "model_cold_load_latency_ms": 0}).lower()
    real = _markdown(summary, {"offline": False, "model_cold_load_latency_ms": 0})
    assert "real read-only" in real and "Fake-only" not in real
