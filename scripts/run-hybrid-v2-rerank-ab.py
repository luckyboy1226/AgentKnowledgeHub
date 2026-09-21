"""Offline command boundary for the paired local-rerank A/B contract."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
os.chdir(ROOT / "python")

from services.hybrid_v2_ab_benchmark import FrozenABInputs, offline_inputs, stable_hash
from services.hybrid_v2_rerank_ab import (DeterministicOfflinePairedAdapter,
    PairedRerankABRunner, SharedPoolContextExecutor, validate_real_rerank_gate)

BASELINE_RELATIVE = Path(".runtime/evaluation/hybrid-v2-ab-20260921T010806Z/results.json")
BASELINE_SHA256 = "73b98274d92109e533c0b151a9325f2fdccfd27d91d22c9c3033243c9ae62f85"
SOURCE_RUN = "g2-real-ingestion-20260920-02"
SNAPSHOT_ROOT = "6d7099a3deda177146933c7c35f709e240bf061f5a2b62c138a08d276e343c4b"
PLAN_HASH = "99eb2478bb57b59e1847ad7220af03925a5562369032757861eabef64f382b8a"
ALLOWLIST_HASH = "4778224a999f6f3ec91d0555c313a8c58c97fa0e34567a532c23621b92376fa0"


def _real_inputs(run_id: str):
    if run_id != SOURCE_RUN:
        raise ValueError("paired_source_run_not_authorized")
    from services.hybrid_retrieval_v2_real_evaluation import ControlledRealEvaluationRunner
    from config import settings
    source = ControlledRealEvaluationRunner(ROOT / ".runtime" / "evaluation", run_id,
        ROOT / "benchmarks" / "enterprise_20docs_retrieval_v2", {})
    manifest = source._manifest()
    plans = tuple(source._read("query-plans.json", {"plans": []})["plans"])
    allow = source._allowlist()
    allow_payload = source._read("document-allowlist.json", {})
    snapshot = source.frozen_query_embeddings()
    snapshot.validate_partial(plans=plans, run_id=run_id, query_plans_hash=PLAN_HASH,
        embedding=manifest["embedding"])
    if (manifest.get("query_plans_hash") != PLAN_HASH or stable_hash(list(plans)) != PLAN_HASH
            or allow_payload.get("sha256") != ALLOWLIST_HASH
            or stable_hash(sorted(allow)) != ALLOWLIST_HASH
            or snapshot.payload.get("snapshot_hash") != SNAPSHOT_ROOT):
        raise ValueError("paired_frozen_artifact_mismatch")
    template = offline_inputs(ROOT / "benchmarks" / "enterprise_20docs_retrieval_v2", run_id=run_id)
    mapping = source._read("document-map.json", {}).get("mapping", {})
    cases = tuple(replace(case, relevant_documents=tuple(mapping[value] for value in case.relevant_documents))
                  for case in template.cases)
    inputs = FrozenABInputs(run_id, template.fixture_fingerprint, ALLOWLIST_HASH, PLAN_HASH,
        SNAPSHOT_ROOT, plans, cases, frozenset(allow), trusted_snapshot_root_hash=SNAPSHOT_ROOT)
    inputs.validate()
    return inputs, snapshot, manifest


class _RealPairedAdapter:
    def __init__(self, host, snapshot, model_path: Path):
        from retrieval.context_builder import ContextBuilderV2
        from retrieval.local_bge_reranker import LocalBGERerankProvider
        from retrieval.reranker import ConfigurableModelReranker
        components = host.components
        parent = components["ParentExpander"]
        common = {"parent_expander": parent, "parent_expansion_enabled": True,
            "rerank_input_top_k": 20, "rerank_output_top_k": 8,
            "final_context_top_k": 8, "final_context_token_budget": host.settings.final_context_token_budget}
        provider = LocalBGERerankProvider(model_path, device="cpu", batch_size=2, max_length=512)
        started = time.perf_counter(); provider._runtime_once()
        self._cold_load_latency_ms = round((time.perf_counter() - started) * 1000, 3)
        self.executor = SharedPoolContextExecutor(ContextBuilderV2(**common),
            ContextBuilderV2(**common, reranker=ConfigurableModelReranker(provider,
                timeout_seconds=30, max_attempts=1), rerank_enabled=True))
        self.hybrid = components["HybridRetrieverV2"]
        self.fusion = components["RRFFusion"]
        self.snapshot = snapshot
        self.host = host
        self._reported_cold = False
        self.safe_call_audit = {"vector": 0, "bm25": 0, "neo4j": 0, "reranker": 0,
                                "chat": 0, "embedding": 0, "writes": 0}

    async def retrieve_pair(self, *, plan, query_ordinal, allowed_document_ids, frozen_vector_sha256):
        vector = self.snapshot.vector_for(run_id=SOURCE_RUN, plan=plan, query_plans_hash=PLAN_HASH,
            embedding={"provider": self.host.settings.embedding_config.provider,
                "model": self.host.settings.embedding_config.model,
                "dimension": self.host.settings.embedding_dimensions,
                "embedding_space_id": self.host.settings.resolved_embedding_space_id},
            query_ordinal=query_ordinal)
        expected_vector_identity = stable_hash({"snapshot_root": SNAPSHOT_ROOT,
            "question_id": str(plan["question_id"]), "ordinal": query_ordinal,
            "query_sha256": hashlib.sha256(list(plan["queries"])[query_ordinal].encode()).hexdigest()})
        if expected_vector_identity != frozen_vector_sha256:
            raise ValueError("paired_frozen_vector_hash_mismatch")
        query = list(plan["queries"])[query_ordinal]
        started = time.perf_counter()
        raw = await self.hybrid.retrieve(query, entities=list(plan.get("entities") or []),
            allowed_document_ids=allowed_document_ids, bm25_enabled=True,
            selection={"vector": True, "bm25": True, "graph": True},
            query_embedding=vector)
        fused = self.fusion.fuse({"bm25": raw["bm25"], "vector": raw["vector"],
                                  "graph": raw["graph"]}).candidates[:20]
        retrieval_ms = (time.perf_counter() - started) * 1000
        self.safe_call_audit["vector"] += 1; self.safe_call_audit["bm25"] += 1
        self.safe_call_audit["neo4j"] += int(bool(plan.get("entities")))
        self.safe_call_audit["reranker"] += 1
        pair = await self.executor.execute(query=query, pool=fused,
            allowed_document_ids=allowed_document_ids, retrieval_latency_ms=retrieval_ms,
            call_audit={"vector": 1, "bm25": 1, "neo4j": int(bool(plan.get("entities"))),
                        "reranker": 0, "chat": 0, "embedding": 0})
        if not self._reported_cold:
            pair = replace(pair, model_cold_load_latency_ms=self._cold_load_latency_ms)
            self._reported_cold = True
        return pair


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--runtime", choices=("offline", "real"), default="offline")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ab-run-id")
    parser.add_argument("--model-path")
    parser.add_argument("--allow-real-read-only", action="store_true")
    parser.add_argument("--allow-frozen-snapshot", action="store_true")
    parser.add_argument("--allow-local-model", action="store_true")
    args = parser.parse_args(argv)
    runtime = "offline" if args.offline else args.runtime
    baseline = ROOT / BASELINE_RELATIVE
    if not baseline.is_file() or hashlib.sha256(baseline.read_bytes()).hexdigest() != BASELINE_SHA256:
        parser.error("immutable no-rerank baseline hash mismatch")
    ab_run_id = args.ab_run_id or f"{args.run_id}-rerank-ab-offline"
    if runtime == "offline":
        inputs = offline_inputs(ROOT / "benchmarks" / "enterprise_20docs_retrieval_v2", run_id=args.run_id)
        model_identity_hash = hashlib.sha256(b"local-bge-reranker-v2-m3-offline-contract-v1").hexdigest()
        result = asyncio.run(PairedRerankABRunner(inputs=inputs,
            adapter=DeterministicOfflinePairedAdapter(inputs.cases),
            output_root=ROOT / ".runtime" / "evaluation", ab_run_id=ab_run_id,
            model_identity_hash=model_identity_hash).run())
    else:
        if not (args.allow_real_read_only and args.allow_frozen_snapshot and args.allow_local_model):
            parser.error("real paired rerank A/B requires all explicit authorization flags")
        if args.run_id != SOURCE_RUN or not args.model_path:
            parser.error("real paired rerank A/B requires frozen source run and local model path")
        from config import settings
        from retrieval.local_bge_reranker import local_bge_model_identity_hash
        model_identity_hash = local_bge_model_identity_hash(args.model_path)
        inputs, snapshot, _manifest = _real_inputs(args.run_id)
        validate_real_rerank_gate(inputs,
            expected_fixture_fingerprint=inputs.fixture_fingerprint,
            expected_allowlist_hash=ALLOWLIST_HASH, expected_query_plan_hash=PLAN_HASH,
            expected_snapshot_root_hash=SNAPSHOT_ROOT, model_identity_hash=model_identity_hash,
            expected_model_identity_hash=model_identity_hash, device="cpu", input_top_k=20,
            output_top_k=8, planned_write_operations=0)
        from services.hybrid_retrieval_v2_host import build_real_runtime_host
        async def real_run():
            with build_real_runtime_host(settings) as host:
                readiness = host.validate_readiness()
                if not all(readiness[key] for key in ("mongo_ready", "chroma_ready", "neo4j_ready")):
                    raise ValueError("paired_runtime_not_ready")
                await host.open_ingestion_dependencies()
                try:
                    from services.hybrid_retrieval_v2_runtime import RealRuntimeStorageVerifier
                    verified = await RealRuntimeStorageVerifier(host.components).verify_scope(inputs.allowed_document_ids)
                    if not all(verified.values()): raise ValueError("paired_scope_not_ready")
                    adapter = _RealPairedAdapter(host, snapshot, Path(args.model_path))
                    return await PairedRerankABRunner(inputs=inputs, adapter=adapter,
                        output_root=ROOT / ".runtime" / "evaluation", ab_run_id=ab_run_id,
                        model_identity_hash=model_identity_hash).run()
                finally:
                    await host.close_ingestion_dependencies()
        result = asyncio.run(real_run())
    print(json.dumps({"offline": bool(result["payload"]["offline"]), "ab_run_id": ab_run_id,
        "result_count": len(result["payload"]["results"]),
        "directory": str(result["directory"].relative_to(ROOT)).replace("\\", "/")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
