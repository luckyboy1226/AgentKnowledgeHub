"""Command boundary for the frozen-input Hybrid V2 Retrieval A/B benchmark."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"python")); os.chdir(ROOT/"python")
from services.hybrid_v2_ab_benchmark import (ABCase, DeterministicOfflineABAdapter, FrozenABInputs,
    HybridV2ABRunner, MODES, offline_inputs, stable_hash)

SOURCE_RUN="g2-real-ingestion-20260920-02"
SNAPSHOT_ROOT="6d7099a3deda177146933c7c35f709e240bf061f5a2b62c138a08d276e343c4b"
PLAN_HASH="99eb2478bb57b59e1847ad7220af03925a5562369032757861eabef64f382b8a"
ALLOWLIST_HASH="4778224a999f6f3ec91d0555c313a8c58c97fa0e34567a532c23621b92376fa0"

def _output_root(value:str|None)->Path:
    base=(ROOT/".runtime"/"evaluation").resolve(); target=(Path(value).resolve() if value else base)
    if target!=base: raise ValueError("ab_report_root_must_be_runtime_evaluation")
    return target

def _real_inputs(run_id:str)->tuple[FrozenABInputs,Any]:
    if run_id!=SOURCE_RUN: raise ValueError("ab_source_run_not_authorized")
    from services.hybrid_retrieval_v2_real_evaluation import ControlledRealEvaluationRunner
    from config import settings
    source=ControlledRealEvaluationRunner(ROOT/".runtime"/"evaluation",run_id,ROOT/"benchmarks"/"enterprise_20docs_retrieval_v2",{})
    manifest=source._manifest(); plans=tuple(source._read("query-plans.json",{"plans":[]})["plans"])
    allow=source._allowlist(); allow_payload=source._read("document-allowlist.json",{})
    snapshot=source.frozen_query_embeddings(); snapshot.validate_partial(plans=plans,run_id=run_id,
        query_plans_hash=PLAN_HASH,embedding=manifest["embedding"])
    if (manifest.get("query_plans_hash")!=PLAN_HASH or stable_hash(list(plans))!=PLAN_HASH
            or allow_payload.get("sha256")!=ALLOWLIST_HASH or stable_hash(sorted(allow))!=ALLOWLIST_HASH
            or snapshot.payload.get("snapshot_hash")!=SNAPSHOT_ROOT): raise ValueError("ab_frozen_artifact_mismatch")
    template=offline_inputs(ROOT/"benchmarks"/"enterprise_20docs_retrieval_v2",run_id=run_id)
    mapping=source._read("document-map.json",{}).get("mapping",{})
    cases=tuple(replace(case,relevant_documents=tuple(mapping[value] for value in case.relevant_documents)) for case in template.cases)
    inputs=FrozenABInputs(run_id,template.fixture_fingerprint,ALLOWLIST_HASH,PLAN_HASH,SNAPSHOT_ROOT,
                          plans,cases,frozenset(allow),trusted_snapshot_root_hash=SNAPSHOT_ROOT); inputs.validate()
    return inputs,snapshot

class _RealAdapter:
    def __init__(self,runtime:Any): self.runtime=runtime
    async def retrieve(self,*,plan,query_ordinal,mode,allowed_document_ids,top_k,candidate_budget,frozen_vector_sha256):
        result=await self.runtime.evaluate(plan,mode,allowed_document_ids,trace=False,graph_trace=False,query_ordinal=query_ordinal)
        return {"document_ranks":list(result.get("document_ranks") or [])[:candidate_budget],
                "candidate_ids":list(result.get("final_context_ids") or [])[:top_k],
                "source_ids":list(result.get("source_ids") or []),"graph_candidate_ids":list(result.get("graph_candidate_ids") or []),
                "graph_final_candidate_ids":list(result.get("graph_final_candidate_ids") or []),"graph_edges":list(result.get("graph_edges") or []),
                "graph_provenance":list(result.get("graph_provenance") or []),"latency_ms":float((result.get("latency") or {}).get("retrieval_total_ms") or 0.0),
                "call_audit":{"vector":int((result.get("call_audit") or {}).get("vector_searches",0)),"bm25":int((result.get("call_audit") or {}).get("bm25_searches",0)),"neo4j":int((result.get("call_audit") or {}).get("graph_queries",0)),"reranker":int((result.get("call_audit") or {}).get("reranker_calls",0)),"chat":int((result.get("call_audit") or {}).get("query_plan_llm_calls",0)),"embedding":int((result.get("call_audit") or {}).get("embedding_queries",0))}}

def main(argv=None)->int:
    parser=argparse.ArgumentParser(); parser.add_argument("--offline",action="store_true"); parser.add_argument("--runtime",choices=("offline","real"),default="offline")
    parser.add_argument("--run-id",default="offline-frozen-fixture"); parser.add_argument("--ab-run-id",required=True)
    parser.add_argument("--modes",default=",".join(MODES)); parser.add_argument("--top-k",type=int,default=8); parser.add_argument("--candidate-budget",type=int,default=30)
    parser.add_argument("--allow-real-retrieval",action="store_true"); parser.add_argument("--allow-frozen-snapshot",action="store_true"); parser.add_argument("--report-dir")
    args=parser.parse_args(argv); runtime="offline" if args.offline else args.runtime; modes=tuple(value.strip() for value in args.modes.split(",") if value.strip())
    output=_output_root(args.report_dir)
    async def execute():
        if runtime=="offline":
            inputs=offline_inputs(ROOT/"benchmarks"/"enterprise_20docs_retrieval_v2",run_id=args.run_id)
            return await HybridV2ABRunner(inputs=inputs,adapter=DeterministicOfflineABAdapter(inputs.cases),output_root=output,
                ab_run_id=args.ab_run_id,modes=modes,top_k=args.top_k,candidate_budget=args.candidate_budget).run()
        if not args.allow_real_retrieval: parser.error("real runtime requires --allow-real-retrieval")
        if not args.allow_frozen_snapshot: parser.error("real runtime requires --allow-frozen-snapshot")
        if args.run_id!=SOURCE_RUN: parser.error("real runtime requires the authorized frozen run-id")
        from config import settings
        if args.top_k!=settings.final_context_top_k or args.candidate_budget!=settings.rrf_fusion_top_k:
            parser.error("real runtime top-k and candidate budget must match frozen configuration")
        inputs,snapshot=_real_inputs(args.run_id)
        from services.hybrid_retrieval_v2_host import build_real_runtime_host
        from services.hybrid_retrieval_v2_runtime import build_hosted_real_runtime
        with build_real_runtime_host(settings) as host:
            await host.open_ingestion_dependencies()
            try:
                real=build_hosted_real_runtime(host,trace_run_id=args.ab_run_id,embedding_snapshot=snapshot,query_plans_hash=PLAN_HASH)
                return await HybridV2ABRunner(inputs=inputs,adapter=_RealAdapter(real),output_root=output,
                    ab_run_id=args.ab_run_id,modes=modes,top_k=args.top_k,candidate_budget=args.candidate_budget).run()
            finally: await host.close_ingestion_dependencies()
    result=asyncio.run(execute()); failures=sum(not row["success"] for row in result["payload"]["results"])
    print(json.dumps({"ab_run_id":args.ab_run_id,"result_count":len(result["payload"]["results"]),"failure_count":failures,"offline":result["payload"]["offline"],"directory":str(result["directory"].relative_to(ROOT)).replace("\\","/")},ensure_ascii=False))
    return 1 if failures else 0

if __name__=="__main__": raise SystemExit(main())
