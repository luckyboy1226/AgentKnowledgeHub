"""Frozen-input Retrieval A/B benchmark with no answer generation.

The runner is I/O-agnostic.  Its offline adapter is deterministic and never
constructs providers or storage clients; a real adapter must be injected by an
explicitly authorized command boundary.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence


SCHEMA_VERSION = "hybrid-v2-ab-v1"
TIE_BREAK_POLICY = "score-desc-candidate-id-asc-v1"
MODES = ("vector_only", "bm25_only", "graph_only", "full_hybrid_v2")
GRAPH_MODES = frozenset(("graph_only", "full_hybrid_v2"))
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",", ":")).encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    descriptor,name=tempfile.mkstemp(prefix=f".{path.name}.",suffix=".tmp",dir=path.parent)
    try:
        with os.fdopen(descriptor,"w",encoding="utf-8",newline="\n") as handle:
            json.dump(value,handle,ensure_ascii=False,indent=2,sort_keys=True); handle.write("\n")
        os.replace(name,path)
    except BaseException:
        try: os.unlink(name)
        except OSError: pass
        raise


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    descriptor,name=tempfile.mkstemp(prefix=f".{path.name}.",suffix=".tmp",dir=path.parent)
    try:
        with os.fdopen(descriptor,"w",encoding="utf-8",newline="\n") as handle: handle.write(value)
        os.replace(name,path)
    except BaseException:
        try: os.unlink(name)
        except OSError: pass
        raise


@dataclass(frozen=True)
class ABCase:
    question_id: str
    category: str
    answerable: bool
    relevant_documents: tuple[str,...]
    expected_sources: tuple[str,...]
    expected_edges: tuple[tuple[str,str,str,str],...]
    expected_path: tuple[tuple[str,str,str,str],...]


@dataclass(frozen=True)
class FrozenABInputs:
    run_id: str
    fixture_fingerprint: str
    allowlist_hash: str
    query_plan_hash: str
    snapshot_root_hash: str
    plans: tuple[dict[str,Any],...]
    cases: tuple[ABCase,...]
    allowed_document_ids: frozenset[str]
    trusted_snapshot_root_hash: str|None = None

    def validate(self) -> None:
        if (not SAFE_ID.fullmatch(self.run_id) or not self.allowed_document_ids
                or stable_hash(sorted(self.allowed_document_ids)) != self.allowlist_hash
                or stable_hash(list(self.plans)) != self.query_plan_hash):
            raise ValueError("ab_frozen_input_identity_mismatch")
        if not re.fullmatch(r"[0-9a-f]{64}",self.fixture_fingerprint): raise ValueError("ab_fixture_fingerprint_invalid")
        if not re.fullmatch(r"[0-9a-f]{64}",self.snapshot_root_hash): raise ValueError("ab_snapshot_root_invalid")
        if self.trusted_snapshot_root_hash is not None and self.snapshot_root_hash!=self.trusted_snapshot_root_hash:
            raise ValueError("ab_snapshot_root_mismatch")
        plan_ids=[str(row.get("question_id") or "") for row in self.plans]
        if len(plan_ids)!=len(set(plan_ids)) or set(plan_ids)!={case.question_id for case in self.cases}:
            raise ValueError("ab_query_plan_case_mismatch")
        for plan in self.plans:
            queries=plan.get("queries")
            if not isinstance(queries,list) or not queries or not all(isinstance(value,str) and value for value in queries):
                raise ValueError("ab_query_plan_invalid")


class ABAdapter(Protocol):
    async def retrieve(self, *, plan: dict[str,Any], query_ordinal: int, mode: str,
                       allowed_document_ids: frozenset[str], top_k: int,
                       candidate_budget: int, frozen_vector_sha256: str) -> dict[str,Any]: ...


def percentile(values: Sequence[float], fraction: float) -> float|None:
    if not values: return None
    ordered=sorted(float(value) for value in values); position=(len(ordered)-1)*fraction
    low,high=math.floor(position),math.ceil(position)
    return round(ordered[low]+(ordered[high]-ordered[low])*(position-low),3)


def retrieval_metrics(ranked: Sequence[str], relevant: Sequence[str], *, available_k: int|None=None) -> dict[str,float|None]:
    relevant_set=set(relevant)
    if not relevant_set:
        return {"recall_at_1":None,"recall_at_3":None,"recall_at_5":None,"recall_at_10":None,"mrr_at_10":None,"ndcg_at_10":None}
    # Multiple chunks/graph evidence can point at one document.  Document-level
    # metrics count only the first occurrence so DCG cannot exceed ideal DCG.
    unique_ranked=list(dict.fromkeys(str(value) for value in ranked))
    def recall(k:int)->float|None:
        if available_k is not None and available_k<k: return None
        return round(len(set(unique_ranked[:k])&relevant_set)/len(relevant_set),6)
    first=next((index for index,value in enumerate(unique_ranked[:10],1) if value in relevant_set),None)
    dcg=sum(1/math.log2(index+2) for index,value in enumerate(unique_ranked[:10]) if value in relevant_set)
    idcg=sum(1/math.log2(index+2) for index in range(min(10,len(relevant_set))))
    return {"recall_at_1":recall(1),"recall_at_3":recall(3),"recall_at_5":recall(5),"recall_at_10":recall(10),
            "mrr_at_10":(None if available_k is not None and available_k<10 else (round(1/first,6) if first else 0.0)),
            "ndcg_at_10":(None if available_k is not None and available_k<10 else (round(dcg/idcg,6) if idcg else 0.0))}


def graph_metrics(*, mode: str, edges: Sequence[Sequence[str]], expected_edges: Sequence[Sequence[str]],
                  expected_path: Sequence[Sequence[str]], graph_ids: Sequence[str], final_ids: Sequence[str],
                  provenance: Sequence[dict[str,Any]]) -> dict[str,Any]:
    if mode not in GRAPH_MODES:
        return {key:None for key in ("graph_evidence_participation_rate","graph_evidence_final_top_k_rate","expected_relation_edge_recall","expected_relation_path_recall","direction_fidelity","predicate_fidelity","provenance_coverage")}
    found={tuple(str(value) for value in edge) for edge in edges}; expected={tuple(str(value) for value in edge) for edge in expected_edges}; path={tuple(str(value) for value in edge) for edge in expected_path}
    endpoint_matches=[edge for edge in found if any(edge[0]==target[0] and edge[2]==target[2] for target in expected)]
    predicate_matches=sum(any(edge[0]==target[0] and edge[1]==target[1] and edge[2]==target[2] for target in expected) for edge in found)
    direction_matches=sum(edge in expected for edge in found)
    return {
        "graph_evidence_participation_rate":1.0 if graph_ids else 0.0,
        "graph_evidence_final_top_k_rate":round(len(set(graph_ids)&set(final_ids))/len(set(graph_ids)),6) if graph_ids else 0.0,
        "expected_relation_edge_recall":round(len(found&expected)/len(expected),6) if expected else None,
        "expected_relation_path_recall":round(len(found&path)/len(path),6) if path else None,
        "direction_fidelity":round(direction_matches/len(endpoint_matches),6) if endpoint_matches else (None if not expected else 0.0),
        "predicate_fidelity":round(predicate_matches/len(endpoint_matches),6) if endpoint_matches else (None if not expected else 0.0),
        "provenance_coverage":round(sum(bool(row.get("document_id") and row.get("source")) for row in provenance)/len(provenance),6) if provenance else 0.0,
    }


class DeterministicOfflineABAdapter:
    """Gold-shaped fake data validates plumbing only; it is not a quality result."""
    def __init__(self, cases: Sequence[ABCase], *, fail_modes: Iterable[str]=()):
        self.cases={case.question_id:case for case in cases}; self.fail_modes=set(fail_modes); self.calls=[]

    async def retrieve(self, *, plan: dict[str,Any], query_ordinal: int, mode: str,
                       allowed_document_ids: frozenset[str], top_k: int,
                       candidate_budget: int, frozen_vector_sha256: str) -> dict[str,Any]:
        self.calls.append((id(plan),query_ordinal,mode,frozen_vector_sha256,allowed_document_ids))
        if mode in self.fail_modes: raise RuntimeError("offline_mode_failure")
        case=self.cases[str(plan["question_id"])]
        relevant=sorted(value for value in case.relevant_documents if value in allowed_document_ids)
        noise=sorted(allowed_document_ids.difference(relevant))
        ranked=(relevant+noise)[:candidate_budget]
        graph=mode in GRAPH_MODES; graph_ids=[f"graph:{stable_hash(edge)}" for edge in case.expected_edges] if graph else []
        edges=[list(edge) for edge in case.expected_edges] if graph else []
        return {"document_ranks":ranked,"candidate_ids":[f"{mode}:{value}" for value in ranked],
                "source_ids":list(case.expected_sources) if relevant else [],"graph_candidate_ids":graph_ids,
                "graph_edges":edges,"graph_provenance":[{"document_id":relevant[0],"source":case.expected_sources[0]}] if graph and relevant and case.expected_sources else [],
                "latency_ms":float(1+query_ordinal+MODES.index(mode)),
                "call_audit":{"vector":int(mode in {"vector_only","full_hybrid_v2"}),"bm25":int(mode in {"bm25_only","full_hybrid_v2"}),"neo4j":int(graph),"reranker":0,"chat":0,"embedding":0}}


class HybridV2ABRunner:
    def __init__(self, *, inputs: FrozenABInputs, adapter: ABAdapter, output_root: Path,
                 ab_run_id: str, modes: Sequence[str]=MODES, top_k: int=10, candidate_budget: int=30):
        inputs.validate()
        if (not SAFE_ID.fullmatch(ab_run_id) or not modes or any(mode not in MODES for mode in modes)
                or top_k<1 or candidate_budget<top_k): raise ValueError("ab_request_invalid")
        self.inputs=inputs; self.adapter=adapter; self.directory=output_root/ab_run_id
        self.ab_run_id=ab_run_id; self.modes=tuple(modes); self.top_k=top_k; self.candidate_budget=candidate_budget

    async def run(self) -> dict[str,Any]:
        if self.directory.exists(): raise ValueError("ab_output_directory_exists")
        started=datetime.now(UTC).isoformat(); rows=[]; by_case={case.question_id:case for case in self.inputs.cases}
        for plan in self.inputs.plans:
            case=by_case[str(plan["question_id"])]
            for ordinal,query in enumerate(plan["queries"]):
                vector_sha=stable_hash({"snapshot_root":self.inputs.snapshot_root_hash,"question_id":case.question_id,"ordinal":ordinal,"query_sha256":hashlib.sha256(query.encode()).hexdigest()})
                for mode in self.modes:
                    base={"run_id":self.inputs.run_id,"ab_run_id":self.ab_run_id,"benchmark_schema_version":SCHEMA_VERSION,
                          "fixture_fingerprint":self.inputs.fixture_fingerprint,"allowlist_hash":self.inputs.allowlist_hash,
                          "query_plan_hash":self.inputs.query_plan_hash,"query_embedding_snapshot_root_hash":self.inputs.snapshot_root_hash,
                          "question_id":case.question_id,"query_ordinal":ordinal,"retrieval_mode":mode,"top_k":self.top_k,
                          "candidate_budget":self.candidate_budget,"tie_break_policy_version":TIE_BREAK_POLICY,"timestamp":started}
                    try:
                        observed=await self.adapter.retrieve(plan=plan,query_ordinal=ordinal,mode=mode,
                            allowed_document_ids=self.inputs.allowed_document_ids,top_k=self.top_k,
                            candidate_budget=self.candidate_budget,frozen_vector_sha256=vector_sha)
                        ranked=list(observed.get("document_ranks") or [])[:self.top_k]; final_ids=list(observed.get("candidate_ids") or [])[:self.top_k]
                        sources=list(observed.get("source_ids") or [])[:self.top_k]; graph_ids=list(observed.get("graph_candidate_ids") or [])
                        graph_final_ids=list(observed.get("graph_final_candidate_ids") or [value for value in graph_ids if value in final_ids])
                        row={**base,"success":True,"failure_summary":None,"zero_result":not bool(ranked),
                             "document_ranks":ranked[:self.top_k],"candidate_ids":final_ids,"source_ids":sources,
                             "retrieval_metrics":retrieval_metrics(ranked,case.relevant_documents,available_k=self.top_k),
                             "source_coverage":round(len(set(sources)&set(case.expected_sources))/len(set(case.expected_sources)),6) if case.expected_sources else None,
                             "graph_metrics":graph_metrics(mode=mode,edges=observed.get("graph_edges") or [],expected_edges=case.expected_edges,
                                 expected_path=case.expected_path,graph_ids=graph_ids,final_ids=graph_final_ids,provenance=observed.get("graph_provenance") or []),
                             "latency_ms":float(observed.get("latency_ms") or 0.0),"call_audit":dict(observed.get("call_audit") or {})}
                    except Exception as exc:
                        row={**base,"success":False,"failure_summary":type(exc).__name__,"zero_result":False,
                             "document_ranks":[],"candidate_ids":[],"source_ids":[],"retrieval_metrics":retrieval_metrics([],case.relevant_documents,available_k=self.top_k),
                             "source_coverage":None,"graph_metrics":graph_metrics(mode=mode,edges=[],expected_edges=case.expected_edges,
                                 expected_path=case.expected_path,graph_ids=[],final_ids=[],provenance=[]),"latency_ms":None,
                             "call_audit":{"vector":0,"bm25":0,"neo4j":0,"reranker":0,"chat":0,"embedding":0}}
                    row["result_hash"]=stable_hash(row); rows.append(row)
        payload={"schema_version":SCHEMA_VERSION,"ab_run_id":self.ab_run_id,"offline":isinstance(self.adapter,DeterministicOfflineABAdapter),"results":rows}
        summary=summarize(rows); definitions=metric_definitions()
        offline=isinstance(self.adapter,DeterministicOfflineABAdapter)
        metadata={"schema_version":SCHEMA_VERSION,"ab_run_id":self.ab_run_id,"source_run_id":self.inputs.run_id,
                  "modes":list(self.modes),"top_k":self.top_k,"candidate_budget":self.candidate_budget,
                  "result_count":len(rows),"offline":offline,"document_body_saved":False,"embedding_saved":False,"external_calls":0}
        input_hashes={"fixture_fingerprint":self.inputs.fixture_fingerprint,"allowlist_hash":self.inputs.allowlist_hash,
                      "query_plan_hash":self.inputs.query_plan_hash,"query_embedding_snapshot_root_hash":self.inputs.snapshot_root_hash}
        atomic_json(self.directory/"results.json",payload); write_csv(self.directory/"results.csv",rows)
        atomic_json(self.directory/"metric-definitions.json",definitions); atomic_json(self.directory/"safe-run-metadata.json",metadata)
        atomic_json(self.directory/"input-hashes.json",input_hashes); atomic_text(self.directory/"summary.md",summary_markdown(summary,metadata))
        return {"payload":payload,"summary":summary,"directory":self.directory}


def summarize(rows: Sequence[dict[str,Any]]) -> dict[str,Any]:
    output={}
    for mode in MODES:
        subset=[row for row in rows if row["retrieval_mode"]==mode]; successful=[row for row in subset if row["success"]]
        metric=lambda name:[row["retrieval_metrics"][name] for row in successful if row["retrieval_metrics"][name] is not None]
        mean=lambda values:round(statistics.mean(values),6) if values else None
        latencies=[row["latency_ms"] for row in successful if row["latency_ms"] is not None]
        graph_keys=("graph_evidence_participation_rate","graph_evidence_final_top_k_rate","expected_relation_edge_recall","expected_relation_path_recall","direction_fidelity","predicate_fidelity","provenance_coverage")
        graph_summary={key:(mean([row["graph_metrics"][key] for row in successful if row["graph_metrics"][key] is not None]) if mode in GRAPH_MODES else None) for key in graph_keys}
        output[mode]={"queries":len(subset),"successful":len(successful),"query_success_rate":round(len(successful)/len(subset),6) if subset else None,
            "zero_result_rate":round(sum(bool(row["zero_result"]) for row in successful)/len(successful),6) if successful else None,
            **{name:mean(metric(name)) for name in ("recall_at_1","recall_at_3","recall_at_5","recall_at_10","mrr_at_10","ndcg_at_10")},
            "source_coverage":mean([row["source_coverage"] for row in successful if row["source_coverage"] is not None]),
            "average_latency_ms":mean(latencies),"p50_latency_ms":percentile(latencies,.5),"p95_latency_ms":percentile(latencies,.95),
            "component_calls":{key:sum(int(row["call_audit"].get(key,0)) for row in successful) for key in ("vector","bm25","neo4j","reranker","chat","embedding")},
            **graph_summary,"answer_accuracy":None,"answer_quality":None,"llm_judge":None}
    return {"schema_version":SCHEMA_VERSION,"modes":output}


def metric_definitions() -> dict[str,Any]:
    return {"schema_version":SCHEMA_VERSION,"relevance_level":"document",
            "metrics":{"recall_at_k":"unique relevant documents in top K / gold relevant documents","mrr_at_10":"reciprocal rank of first relevant document within 10","ndcg_at_10":"binary document relevance DCG / ideal DCG","source_coverage":"expected source filenames present / expected source filenames","zero_result_rate":"successful queries with no result / successful queries","query_success_rate":"successful rows / planned rows","graph_metrics":"N/A for non-graph modes","answer_metrics":"N/A; no answer generation or judge"}}


def write_csv(path: Path, rows: Sequence[dict[str,Any]]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); descriptor,name=tempfile.mkstemp(prefix=f".{path.name}.",suffix=".tmp",dir=path.parent)
    fields=("run_id","ab_run_id","benchmark_schema_version","question_id","query_ordinal","retrieval_mode","success","zero_result","latency_ms","source_coverage","result_hash")
    try:
        with os.fdopen(descriptor,"w",encoding="utf-8",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows({key:row.get(key) for key in fields} for row in rows)
        os.replace(name,path)
    except BaseException:
        try: os.unlink(name)
        except OSError: pass
        raise


def summary_markdown(summary: dict[str,Any], metadata: dict[str,Any]) -> str:
    offline=bool(metadata.get("offline"))
    title="offline" if offline else "real read-only"
    note=("Fake-only contract validation; not a real retrieval quality result."
          if offline else "Frozen-input read-only retrieval result; no answer generation or LLM judge.")
    lines=[f"# Hybrid V2 Retrieval A/B ({title})","",note,"", "| Mode | Success | Recall@5 | Recall@10 | MRR@10 | P95 ms |", "|---|---:|---:|---:|---:|---:|"]
    for mode,item in summary["modes"].items(): lines.append(f"| {mode} | {item['successful']}/{item['queries']} | {item['recall_at_5']} | {item['recall_at_10']} | {item['mrr_at_10']} | {item['p95_latency_ms']} |")
    lines.extend(["",f"Rows: {metadata['result_count']}; external calls: 0.","Answer accuracy and LLM quality: N/A."])
    return "\n".join(lines)+"\n"


def offline_inputs(benchmark_root: Path, *, run_id: str="offline-frozen-fixture") -> FrozenABInputs:
    ground_path=benchmark_root/"ground_truth.json"; manifest_path=benchmark_root/"benchmark_manifest.json"
    ground=json.loads(ground_path.read_text(encoding="utf-8")); manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    cases=tuple(ABCase(str(row["question_id"]),str(row["category"]),bool(row["answerable"]),tuple(row.get("relevant_documents") or ()),tuple(row.get("expected_sources") or ()),tuple(tuple(value) for value in row.get("expected_edges") or ()),tuple(tuple(value) for value in row.get("expected_path") or ())) for row in ground["questions"])
    plans=[]
    for case in cases:
        queries=[f"{case.question_id}:canonical",f"{case.question_id}:entity",f"{case.question_id}:relation"]
        plans.append({"question_id":case.question_id,"queries":queries,"entities":[],"keywords":[],"intent":"retrieval","plan_hash":stable_hash({"question_id":case.question_id,"queries":queries})})
    allowed=frozenset(f"D{i:02d}" for i in range(1,int(manifest["document_count"])+1))
    fixture=stable_hash({"ground_truth_sha256":file_hash(ground_path),"manifest_sha256":file_hash(manifest_path),"source_benchmark_sha256":manifest["source_benchmark_sha256"]})
    snapshot_root=stable_hash({"fake_snapshot":plans})
    return FrozenABInputs(run_id,fixture,stable_hash(sorted(allowed)),stable_hash(plans),snapshot_root,tuple(plans),cases,allowed,
                          trusted_snapshot_root_hash=snapshot_root)
