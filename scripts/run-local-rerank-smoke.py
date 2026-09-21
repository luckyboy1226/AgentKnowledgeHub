"""One-shot synthetic smoke for a pre-provisioned local BGE reranker."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
from retrieval.local_bge_reranker import LocalBGERerankProvider
from retrieval.reranker import RerankDocument, RerankRequest


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    descriptor,name=tempfile.mkstemp(prefix=f".{path.name}.",suffix=".tmp",dir=path.parent)
    try:
        with os.fdopen(descriptor,"w",encoding="utf-8",newline="\n") as handle:
            json.dump(payload,handle,ensure_ascii=False,indent=2,sort_keys=True); handle.write("\n")
        os.replace(name,path)
    except BaseException:
        try: os.unlink(name)
        except OSError: pass
        raise


async def execute(args) -> tuple[Path,dict]:
    os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
    os.environ.setdefault("HF_HOME",str((ROOT/".runtime"/"hf-cache").resolve()))
    provider=LocalBGERerankProvider(args.model_path,device=args.device,batch_size=args.batch_size,max_length=args.max_length)
    request=RerankRequest(
        query="哪个候选描述了向量数据库？",
        documents=(
            RerankDocument("candidate-a","Chroma 用于保存文本向量。"),
            RerankDocument("candidate-b","Neo4j 用于保存实体关系。"),
            RerankDocument("candidate-c","Atlas 提供事件分发能力。"),
        ),
    )
    started=time.perf_counter(); scores=list(await provider.score(request)); elapsed=round((time.perf_counter()-started)*1000,3)
    if len(scores)!=3 or {row.candidate_id for row in scores}!={"candidate-a","candidate-b","candidate-c"}:
        raise RuntimeError("smoke_candidate_identity_mismatch")
    if any(not math.isfinite(row.score) for row in scores): raise RuntimeError("smoke_non_finite_score")
    ordered=sorted(scores,key=lambda row:(-row.score,row.candidate_id))
    run_id=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"); directory=ROOT/".runtime"/"rerank-smoke"/run_id
    payload={
        "schema_version":"local-bge-rerank-smoke-v1","local_smoke_only":True,
        "provider_kind":"local_bge","device_type":provider.last_diagnostics["device_type"],
        "candidate_count":len(scores),"candidate_ids":[row.candidate_id for row in scores],
        "ranked_candidate_ids":[row.candidate_id for row in ordered],
        "scores_finite":True,"elapsed_ms":elapsed,"batch_count":provider.last_diagnostics["batch_count"],
        "model_path_saved":False,"query_or_content_saved":False,
    }
    atomic_json(directory/"summary.json",payload); return directory,payload


def main(argv=None)->int:
    parser=argparse.ArgumentParser(); parser.add_argument("--model-path",required=True)
    parser.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    parser.add_argument("--batch-size",type=int,default=2); parser.add_argument("--max-length",type=int,default=256)
    args=parser.parse_args(argv); directory,payload=asyncio.run(execute(args))
    print(json.dumps({"summary":str((directory/"summary.json").relative_to(ROOT)).replace("\\","/"),
                      "device_type":payload["device_type"],"candidate_count":payload["candidate_count"],
                      "elapsed_ms":payload["elapsed_ms"]},ensure_ascii=False)); return 0


if __name__=="__main__": raise SystemExit(main())
