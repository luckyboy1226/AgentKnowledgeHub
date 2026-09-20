"""Controlled Phase G runner: durable orchestration, never an API switch.

All I/O is injected.  Importing this module cannot construct a provider,
connect to storage, upload a document, or schedule background work.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, Sequence


VARIANTS = ("vector_only", "bm25_vector_rrf", "vector_graph_rrf", "hybrid_v2_no_rerank")
GRAPH_VARIANTS = frozenset(("vector_graph_rrf", "hybrid_v2_no_rerank"))


class RunState(str, Enum):
    PREPARED = "PREPARED"
    INGESTED = "INGESTED"
    SCOPE_VERIFIED = "SCOPE_VERIFIED"
    QUERY_PLANS_FROZEN = "QUERY_PLANS_FROZEN"
    TRACE_EQUIVALENCE_VERIFIED = "TRACE_EQUIVALENCE_VERIFIED"
    EVALUATING = "EVALUATING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class RealEvaluationAdapter(Protocol):
    """Explicit I/O boundary; production construction belongs in the CLI only."""
    async def ingest(self, document: dict[str, Any], run_id: str) -> dict[str, Any]: ...
    async def verify_document(self, document_id: str) -> dict[str, Any]: ...
    async def verify_scope(self, allowed_document_ids: frozenset[str]) -> dict[str, Any]: ...
    async def build_query_plan(self, question: dict[str, Any], run_id: str) -> dict[str, Any]: ...
    async def evaluate(self, plan: dict[str, Any], variant: str, allowed_document_ids: frozenset[str], *, trace: bool, graph_trace: bool) -> dict[str, Any]: ...


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(value, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try: os.unlink(temporary)
        except OSError: pass
        raise
def atomic_text(path: Path, value:str)->None:
    path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:f.write(value)
        os.replace(tmp,path)
    except BaseException:
        try:os.unlink(tmp)
        except OSError:pass
        raise


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass
class ControlledRealEvaluationRunner:
    root: Path
    run_id: str
    benchmark_root: Path
    configuration: dict[str, Any]

    @property
    def directory(self) -> Path: return self.root / self.run_id
    def path(self, name: str) -> Path: return self.directory / name
    def _read(self, name: str, default: Any) -> Any:
        try: return json.loads(self.path(name).read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError): return default
    def _state(self) -> dict[str, Any]: return self._read("state.json", {"run_id": self.run_id, "status": None, "completed": {}})
    def _write_state(self, status: RunState, **extra: Any) -> None:
        state = self._state(); state.update({"run_id": self.run_id, "status": status.value, "updated_at": datetime.now(UTC).isoformat(), **extra}); atomic_json(self.path("state.json"), state)
    def _fail(self, reason: str) -> None: self._write_state(RunState.FAILED, error=reason)

    def prepare(self, *, git_commit: str, embedding: dict[str, Any]) -> dict[str, Any]:
        ground = self._read_benchmark("ground_truth.json"); base_manifest = self._read_benchmark("benchmark_manifest.json")
        identity = {"run_id": self.run_id, "git_commit": git_commit, "benchmark_id": base_manifest["benchmark_id"], "benchmark_version": base_manifest["benchmark_version"], "benchmark_hash": _hash(ground), "document_count": base_manifest["document_count"], "question_count": base_manifest["question_count"], "embedding": embedding, "reranker_enabled": False, "variants": list(VARIANTS), "configuration": self.configuration}
        existing = self._read("manifest.json", None)
        if existing is not None and {key: existing.get(key) for key in identity} != identity:
            raise ValueError("manifest_conflict")
        manifest = {**identity, "created_at": existing.get("created_at") if existing else datetime.now(UTC).isoformat(), "status": RunState.PREPARED.value}
        atomic_json(self.path("manifest.json"), manifest); self._write_state(RunState.PREPARED); return manifest

    def _read_benchmark(self, name: str) -> dict[str, Any]:
        value = json.loads((self.benchmark_root / name).read_text(encoding="utf-8"))
        if not isinstance(value, dict): raise ValueError("invalid_benchmark")
        return value

    def _manifest(self) -> dict[str, Any]:
        value = self._read("manifest.json", None)
        if not isinstance(value, dict): raise ValueError("run_not_prepared")
        return value

    async def ingest(self, adapter: RealEvaluationAdapter, source_documents: Sequence[dict[str, Any]]) -> dict[str, Any]:
        self._manifest(); records = self._read("ingestion-results.json", {"documents": []}); done = {row.get("logical_key"): row for row in records["documents"] if row.get("status") == "ready"}
        for document in source_documents:
            key = str(document.get("id") or document.get("logical_key") or "")
            if not key: self._fail("invalid_logical_key"); raise ValueError("invalid_logical_key")
            if key in done: continue
            try:
                row = await adapter.ingest(document, self.run_id)
                document_id = str(row.get("document_id") or "")
                if not _uuid(document_id): raise ValueError("invalid_document_uuid")
                verified = await adapter.verify_document(document_id)
                if not all(verified.get(key) is True for key in ("ready", "current", "parents_ready", "children_ready", "vectors_current", "graph_provenance_current")):
                    raise ValueError("ingestion_incomplete")
                item = {"logical_key": key, "source": Path(str(document.get("filename") or "")).name, "document_id": document_id, "document_version": int(row.get("document_version") or row.get("version") or 0), "status": "ready", "content_hash": str(row.get("content_hash") or "")}
                records["documents"].append(item); atomic_json(self.path("ingestion-results.json"), records)
            except Exception as exc:
                self._fail(type(exc).__name__); raise
        expected = self._manifest()["document_count"]
        if len(records["documents"]) != expected: self._fail("ingestion_count_mismatch"); raise ValueError("ingestion_count_mismatch")
        mapping = {row["logical_key"]: row["document_id"] for row in records["documents"]}; allowlist = sorted(mapping.values())
        if len(set(allowlist)) != expected or any(not _uuid(value) for value in allowlist): self._fail("invalid_allowlist"); raise ValueError("invalid_allowlist")
        atomic_json(self.path("document-map.json"), {"run_id": self.run_id, "mapping": mapping}); atomic_json(self.path("document-allowlist.json"), {"run_id": self.run_id, "document_ids": allowlist, "sha256": _hash(allowlist)}); self._write_state(RunState.INGESTED); return mapping

    async def verify_scope(self, adapter: RealEvaluationAdapter) -> None:
        allow = self._allowlist(); verdict = await adapter.verify_scope(frozenset(allow))
        if not all(verdict.get(key) is True for key in ("mongo", "chroma", "neo4j", "bm25")): self._fail("scope_verification_failed"); raise ValueError("scope_verification_failed")
        self._write_state(RunState.SCOPE_VERIFIED, scope=verdict)

    async def build_query_plans(self, adapter: RealEvaluationAdapter) -> list[dict[str, Any]]:
        if self._state().get("status") not in {RunState.SCOPE_VERIFIED.value, RunState.QUERY_PLANS_FROZEN.value}:
            raise ValueError("invalid_state_transition")
        questions = self._read_benchmark("ground_truth.json")["questions"]; payload = self._read("query-plans.json", {"plans": []}); done = {plan.get("question_id") for plan in payload["plans"]}
        for question in questions:
            if question["question_id"] in done: continue
            plan = await adapter.build_query_plan(question, self.run_id)
            plan = {"question_id": question["question_id"], "queries": list(plan.get("queries", [])), "entities": list(plan.get("entities", [])), "keywords": list(plan.get("keywords", [])), "intent": plan.get("intent"), **{"plan_hash": _hash({key: plan.get(key) for key in ("question_id", "queries", "entities", "keywords", "intent")})}}
            payload["plans"].append(plan); atomic_json(self.path("query-plans.json"), payload)
        if len(payload["plans"]) != len(questions): self._fail("query_plans_incomplete"); raise ValueError("query_plans_incomplete")
        manifest = self._manifest(); manifest["query_plans_hash"] = _hash(payload["plans"]); manifest["status"] = RunState.QUERY_PLANS_FROZEN.value; atomic_json(self.path("manifest.json"), manifest); self._write_state(RunState.QUERY_PLANS_FROZEN); return payload["plans"]

    async def trace_equivalence(self, adapter: RealEvaluationAdapter) -> None:
        self._require(RunState.QUERY_PLANS_FROZEN); plans = self._read("query-plans.json", {"plans": []})["plans"][:5]; allow = frozenset(self._allowlist()); rows=[]
        for plan in plans:
            for variant in VARIANTS:
                off=await adapter.evaluate(plan, variant, allow, trace=False, graph_trace=False); on=await adapter.evaluate(plan, variant, allow, trace=True, graph_trace=variant in GRAPH_VARIANTS)
                equal={key: off.get(key) for key in ("final_context_ids","candidate_ranks")} == {key: on.get(key) for key in ("final_context_ids","candidate_ranks")}; rows.append({"question_id":plan["question_id"],"variant":variant,"equivalent":equal})
        atomic_json(self.path("trace-equivalence.json"), {"rows":rows,"passed":all(row["equivalent"] for row in rows)})
        if not all(row["equivalent"] for row in rows): self._fail("trace_equivalence_failed"); raise ValueError("trace_equivalence_failed")
        self._write_state(RunState.TRACE_EQUIVALENCE_VERIFIED)

    async def evaluate(self, adapter: RealEvaluationAdapter) -> list[dict[str, Any]]:
        if self._state().get("status") not in {RunState.TRACE_EQUIVALENCE_VERIFIED.value,RunState.EVALUATING.value}: raise ValueError("invalid_state_transition")
        self._write_state(RunState.EVALUATING)
        plans=self._read("query-plans.json",{"plans":[]})["plans"]; allow=frozenset(self._allowlist()); payload=self._read("results.json",{"results":[]}); done={(r.get("question_id"),r.get("variant"),r.get("order")) for r in payload["results"]}
        for order, variants in ((1,VARIANTS),(2,tuple(reversed(VARIANTS)))):
            for plan in plans:
                for variant in variants:
                    key=(plan["question_id"],variant,order)
                    if key in done: continue
                    row=await adapter.evaluate(plan,variant,allow,trace=True,graph_trace=variant in GRAPH_VARIANTS)
                    audit=dict(row.get("call_audit",{})); self._audit(variant,audit)
                    payload["results"].append({"question_id":plan["question_id"],"variant":variant,"order":order,"final_context_ids":list(row.get("final_context_ids",[])),"document_ranks":list(row.get("document_ranks",[])),"candidate_ranks":list(row.get("candidate_ranks",[])),"latency":dict(row.get("latency",{})),"call_audit":audit,"graph_metrics":dict(row.get("graph_metrics",{}))}); atomic_json(self.path("results.json"),payload)
        for plan in plans:
            for variant in VARIANTS:
                rows=[r for r in payload["results"] if r["question_id"]==plan["question_id"] and r["variant"]==variant]
                if len(rows)!=2 or any(rows[0][k]!=rows[1][k] for k in ("final_context_ids","document_ranks")):
                    self._fail("variant_order_nondeterminism"); raise ValueError("variant_order_nondeterminism")
        return payload["results"]

    def report(self) -> dict[str, Any]:
        if self._state().get("status")!=RunState.EVALUATING.value: raise ValueError("invalid_state_transition")
        ground={row["question_id"]:row for row in self._read_benchmark("ground_truth.json")["questions"]}; mapping=self._read("document-map.json",{}).get("mapping",{}); rows=self._read("results.json",{"results":[]})["results"]; primary=[row for row in rows if row["order"]==1]
        for row in primary:
            expected=[mapping[key] for key in ground[row["question_id"]].get("relevant_documents",[]) if key in mapping]
            if len(expected)!=len(ground[row["question_id"]].get("relevant_documents",[])): self._fail("ground_truth_mapping_missing"); raise ValueError("ground_truth_mapping_missing")
            row["category"]=ground[row["question_id"]].get("category"); row["document_metrics"]=_document_metrics(row["document_ranks"],expected)
        summary={variant:_mean_metrics([r for r in primary if r["variant"]==variant]) for variant in VARIANTS}; atomic_json(self.path("summary.json"),{"controlled_local_service_benchmark":True,"variants":summary})
        lines=["# FAKE / STUB CONTRACT VALIDATION","","Not real retrieval quality, real latency, or production performance.","","| Variant | Recall@10 | MRR |","|---|---:|---:|"]
        lines += [f"| {variant} | {values['recall_at_10']} | {values['mrr']} |" for variant,values in summary.items()]
        atomic_text(self.path("summary.md"),"\n".join(lines)+"\n")
        by={(r["question_id"],r["variant"]):r for r in primary}; improvements=[]; regressions=[]; failures=[]
        for question_id in ground:
            a,d=by.get((question_id,"vector_only")),by.get((question_id,"hybrid_v2_no_rerank"))
            if not a or not d: continue
            ah=bool(a["document_metrics"].get("hit_at_10")); dh=bool(d["document_metrics"].get("hit_at_10")); item={"question_id":question_id,"category":ground[question_id].get("category")}
            if not ah and dh: improvements.append(item)
            if ah and not dh: regressions.append(item)
            if not dh: failures.append({**item,"variant":"hybrid_v2_no_rerank","bucket":"retrieval_miss"})
        atomic_json(self.path("improvements.json"),improvements); atomic_json(self.path("regressions.json"),regressions); atomic_json(self.path("failures.json"),failures); self.cleanup_plan(); self._write_state(RunState.COMPLETED); return summary

    @staticmethod
    def _audit(variant: str,audit: dict[str,Any])->None:
        forbidden={"vector_only":("bm25_searches","graph_queries","reranker_calls"),"bm25_vector_rrf":("graph_queries","reranker_calls"),"vector_graph_rrf":("bm25_searches","reranker_calls"),"hybrid_v2_no_rerank":("reranker_calls",)}[variant]
        if any(audit.get(key,0)!=0 for key in forbidden): raise ValueError("variant_call_audit_failed")

    def cleanup_plan(self) -> dict[str, Any]:
        value={"run_id":self.run_id,"document_ids":self._allowlist(),"document_count":len(self._allowlist()),"requires_explicit_authorization":True,"cleanup_executed":False}; atomic_json(self.path("cleanup-plan.json"),value); return value
    def _allowlist(self) -> list[str]:
        values=self._read("document-allowlist.json",{}).get("document_ids",[])
        if not values or len(values)!=len(set(values)) or any(not _uuid(str(value)) for value in values): raise ValueError("verified_nonempty_allowlist_required")
        return [str(value) for value in values]
    def _require(self,status: RunState)->None:
        if self._state().get("status")!=status.value: raise ValueError("invalid_state_transition")


def _uuid(value: str) -> bool:
    try: return str(uuid.UUID(value)) == value.lower()
    except (ValueError, AttributeError): return False

def _document_metrics(ranks:list[Any], expected:list[str])->dict[str,Any]:
    if not expected:return {"applicable":False,"hit_at_10":None,"recall_at_5":None,"recall_at_10":None,"mrr":None,"ndcg_at_10":None}
    docs=[str(x) for x in ranks]; hits=[i+1 for i,x in enumerate(docs) if x in expected]; recall=lambda k:sum(x in expected for x in docs[:k])/len(expected); dcg=sum(1/__import__('math').log2(i+2) for i,x in enumerate(docs[:10]) if x in expected); idcg=sum(1/__import__('math').log2(i+2) for i in range(min(10,len(expected))))
    return {"applicable":True,"hit_at_10":bool(hits and min(hits)<=10),"recall_at_5":recall(5),"recall_at_10":recall(10),"mrr":1/min(hits) if hits else 0.0,"ndcg_at_10":dcg/idcg if idcg else 0.0}
def _mean_metrics(rows:list[dict[str,Any]])->dict[str,Any]:
    keys=("recall_at_5","recall_at_10","mrr","ndcg_at_10"); return {key:(sum(r["document_metrics"][key] for r in rows if r["document_metrics"][key] is not None)/max(1,sum(r["document_metrics"][key] is not None for r in rows))) for key in keys}
