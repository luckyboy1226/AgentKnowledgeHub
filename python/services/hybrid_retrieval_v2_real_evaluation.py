"""Controlled Phase G runner: durable orchestration, never an API switch.

All I/O is injected.  Importing this module cannot construct a provider,
connect to storage, upload a document, or schedule background work.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, Sequence


VARIANTS = ("vector_only", "bm25_vector_rrf", "vector_graph_rrf", "hybrid_v2_no_rerank")
GRAPH_VARIANTS = frozenset(("vector_graph_rrf", "hybrid_v2_no_rerank"))
TRACE_SAMPLE_QUESTION_IDS = ("Q01", "Q02", "Q03", "Q04", "Q05")
TRACE_COMPARISON_FIELDS = ("final_context_ids", "candidate_ranks", "document_ranks")


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
    async def verify_recovery_identity(self, *, logical_key: str, source: str, document_id: str, operation_id: str) -> dict[str, Any]: ...
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


def _hash_file_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path for path in root.rglob("*") if path.is_file()):
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        digest.update(item.read_bytes())
    return digest.hexdigest()


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

    def _recovery_directory(self, recovery_id: str) -> Path:
        value = str(recovery_id or "")
        if value == self.run_id or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
            raise ValueError("g4_recovery_id_invalid")
        directory = (self.root / value).resolve()
        if directory.parent != self.root.resolve():
            raise ValueError("g4_recovery_id_invalid")
        return directory

    @staticmethod
    def _file_sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def prepare(self, *, git_commit: str, embedding: dict[str, Any]) -> dict[str, Any]:
        ground = self._read_benchmark("ground_truth.json"); base_manifest = self._read_benchmark("benchmark_manifest.json")
        ground_hash = hashlib.sha256((self.benchmark_root / "ground_truth.json").read_bytes()).hexdigest()
        identity = {"run_id": self.run_id, "git_commit": git_commit, "benchmark_id": base_manifest["benchmark_id"], "benchmark_version": base_manifest["benchmark_version"], "benchmark_hash": ground_hash, "document_count": base_manifest["document_count"], "question_count": base_manifest["question_count"], "embedding": embedding, "reranker_enabled": False, "variants": list(VARIANTS), "configuration": self.configuration}
        existing = self._read("manifest.json", None)
        if existing is not None and {key: existing.get(key) for key in identity} != identity:
            raise ValueError("manifest_conflict")
        manifest = {**identity, "created_at": existing.get("created_at") if existing else datetime.now(UTC).isoformat(), "status": RunState.PREPARED.value}
        atomic_json(self.path("manifest.json"), manifest); self._write_state(RunState.PREPARED); return manifest

    def _read_benchmark(self, name: str) -> dict[str, Any]:
        value = json.loads((self.benchmark_root / name).read_text(encoding="utf-8"))
        if not isinstance(value, dict): raise ValueError("invalid_benchmark")
        return value

    def joined_questions(self) -> list[dict[str, Any]]:
        """Join frozen relevance labels to their single canonical text source.

        v2 intentionally omits question wording from ground truth.  The source
        bundle recorded in its immutable metadata is authoritative for wording;
        no answer-derived or generated text is accepted here.
        """
        ground = self._read_benchmark("ground_truth.json")
        source = ground.get("source_benchmark")
        if not isinstance(source, dict):
            raise ValueError("canonical_question_source_missing")
        source_name, source_hash = str(source.get("path") or ""), str(source.get("sha256") or "")
        if not source_name or Path(source_name).name != source_name or not source_hash:
            raise ValueError("canonical_question_source_invalid")
        source_root = self.benchmark_root.parent / source_name
        source_manifest_path = source_root / "benchmark_manifest.json"
        if not source_manifest_path.is_file() or _hash_file_tree(source_root) != source_hash:
            raise ValueError("canonical_question_source_hash_mismatch")
        manifest = self._read_benchmark("benchmark_manifest.json")
        if manifest.get("source_benchmark_sha256") != source_hash:
            raise ValueError("canonical_question_source_manifest_mismatch")
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        filename = str((source_manifest.get("files") or {}).get("questions_json") or "")
        if not filename or Path(filename).name != filename:
            raise ValueError("canonical_question_filename_invalid")
        text_payload = json.loads((source_root / filename).read_text(encoding="utf-8"))
        texts = text_payload.get("questions") if isinstance(text_payload, dict) else None
        labels = ground.get("questions")
        if not isinstance(texts, list) or not isinstance(labels, list):
            raise ValueError("canonical_question_payload_invalid")
        def index(rows: list[Any], *, require_text: bool) -> dict[str, dict[str, Any]]:
            indexed: dict[str, dict[str, Any]] = {}
            for row in rows:
                if not isinstance(row, dict): raise ValueError("canonical_question_row_invalid")
                question_id = str(row.get("question_id") or "")
                if not question_id or question_id in indexed: raise ValueError("canonical_question_id_invalid")
                if require_text and not str(row.get("question") or "").strip(): raise ValueError("canonical_question_text_empty")
                indexed[question_id] = row
            return indexed
        text_by_id, label_by_id = index(texts, require_text=True), index(labels, require_text=False)
        if set(text_by_id) != set(label_by_id): raise ValueError("canonical_question_id_set_mismatch")
        return [{**label_by_id[question_id], "question": str(text_by_id[question_id]["question"]).strip()}
                for question_id in (str(row["question_id"]) for row in labels)]

    def resume_g3_pre_llm_failure(self, *, allowlist_hash: str) -> None:
        """Restore only the known pre-LLM Q01 failure to the G3 entry state."""
        state = self._state()
        if (state.get("status") != RunState.FAILED.value or state.get("failed_question_id") != "Q01"
                or state.get("error") != "KeyError" or self.path("query-plans.json").exists()):
            raise ValueError("g3_resume_guard_state_failed")
        manifest = self._manifest()
        ground_hash = hashlib.sha256((self.benchmark_root / "ground_truth.json").read_bytes()).hexdigest()
        if manifest.get("benchmark_hash") != ground_hash or manifest.get("status") not in {RunState.PREPARED.value, RunState.INGESTED.value}:
            raise ValueError("g3_resume_guard_manifest_failed")
        allow_payload = self._read("document-allowlist.json", {})
        mapping = self._read("document-map.json", {}).get("mapping", {})
        ids = allow_payload.get("document_ids", [])
        if (allow_payload.get("sha256") != allowlist_hash or len(ids) != 20 or len(ids) != len(set(ids))
                or set(mapping.values()) != set(ids) or not all(state.get("scope", {}).get(key) is True for key in ("mongo", "chroma", "neo4j", "bm25"))):
            raise ValueError("g3_resume_guard_g2_artifacts_failed")
        self.joined_questions()
        self._write_state(RunState.SCOPE_VERIFIED, error=None, failed_question_id=None,
                          g3_resumed_from_pre_llm_failure=True)

    def _manifest(self) -> dict[str, Any]:
        value = self._read("manifest.json", None)
        if not isinstance(value, dict): raise ValueError("run_not_prepared")
        return value

    async def freeze_query_embeddings(self, provider: Any, embedding: dict[str, Any]) -> dict[str, Any]:
        from services.query_embedding_snapshot import FrozenQueryEmbeddingSnapshot
        state=self._state()
        if not (state.get("status") == RunState.QUERY_PLANS_FROZEN.value
                or (state.get("status") == RunState.FAILED.value and state.get("error") == "trace_equivalence_failed")):
            raise ValueError("query_embedding_freeze_invalid_state")
        plans=self._read("query-plans.json", {"plans": []}).get("plans",[]); manifest=self._manifest()
        if manifest.get("query_plans_hash") != _hash(plans): raise ValueError("query_plans_hash_mismatch")
        target=self.path("query-embeddings.json")
        payload=self._read("query-embeddings.json", None)
        snapshot=(FrozenQueryEmbeddingSnapshot(payload) if isinstance(payload,dict)
                  else FrozenQueryEmbeddingSnapshot.empty(run_id=self.run_id,query_plans_hash=manifest["query_plans_hash"],embedding=embedding))
        stats=await snapshot.freeze_missing(
            plans=plans,run_id=self.run_id,query_plans_hash=manifest["query_plans_hash"],embedding=embedding,
            provider=provider,persist_partial=lambda value: atomic_json(target,value),
        )
        atomic_json(target,snapshot.payload)
        return {**stats,"snapshot_hash":snapshot.payload["snapshot_hash"],"record_count":len(snapshot.payload["records"])}

    def frozen_query_embeddings(self) -> Any:
        from services.query_embedding_snapshot import FrozenQueryEmbeddingSnapshot
        payload=self._read("query-embeddings.json", None)
        if not isinstance(payload,dict): raise ValueError("query_embedding_snapshot_missing")
        return FrozenQueryEmbeddingSnapshot(payload)

    async def ingest(self, adapter: RealEvaluationAdapter, source_documents: Sequence[dict[str, Any]]) -> dict[str, Any]:
        self._manifest(); progress = self._read("ingestion-progress.json", {"documents": []}); done = {row.get("logical_key"): row for row in progress["documents"] if row.get("status") == "ready"}
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
                item = {"logical_key": key, "source": Path(str(document.get("filename") or "")).name, "document_id": document_id, "document_version": int(row.get("document_version") or row.get("version") or 0), "status": "ready", "content_hash": str(row.get("content_hash") or ""), "operation_id": row.get("operation_id")}
                progress["documents"].append(item); atomic_json(self.path("ingestion-progress.json"), progress)
            except Exception as exc:
                self._write_state(RunState.FAILED, error=type(exc).__name__, failed_document={"logical_key": key, "operation_id": getattr(adapter, "last_ingestion_operation", {}).get("operation_id")})
                raise
        expected = self._manifest()["document_count"]
        if len(progress["documents"]) != expected: self._fail("ingestion_count_mismatch"); raise ValueError("ingestion_count_mismatch")
        mapping = {row["logical_key"]: row["document_id"] for row in progress["documents"]}; allowlist = sorted(mapping.values())
        if len(set(allowlist)) != expected or any(not _uuid(value) for value in allowlist): self._fail("invalid_allowlist"); raise ValueError("invalid_allowlist")
        atomic_json(self.path("ingestion-results.json"), progress); atomic_json(self.path("document-map.json"), {"run_id": self.run_id, "mapping": mapping}); atomic_json(self.path("document-allowlist.json"), {"run_id": self.run_id, "document_ids": allowlist, "sha256": _hash(allowlist)}); self._write_state(RunState.INGESTED); return mapping

    async def recover_ready_document(self, adapter: RealEvaluationAdapter, record: dict[str, Any]) -> dict[str, Any]:
        """Record one previously completed Coordinator operation without re-ingestion.

        This method accepts only a caller-supplied exact identity.  It never
        searches by source, never calls ``adapter.ingest``, and does not emit a
        final map or allowlist; normal ingestion will skip the recovered logical
        key and resume with the next frozen document.
        """
        self._manifest()
        if self.path("ingestion-results.json").exists() or self.path("document-allowlist.json").exists():
            raise ValueError("recovery_after_finalization_forbidden")
        required = ("logical_key", "source", "document_id", "document_version", "operation_id")
        if any(not record.get(key) for key in required):
            raise ValueError("recovery_record_incomplete")
        if not _uuid(str(record["document_id"])) or not _uuid(str(record["operation_id"])):
            raise ValueError("recovery_identity_not_uuid")
        verified = await adapter.verify_recovery_identity(
            logical_key=str(record["logical_key"]), source=Path(str(record["source"])).name, document_id=str(record["document_id"]),
            operation_id=str(record["operation_id"]),
        )
        expected = {**record, "source": Path(str(record["source"])).name,
                    "content_hash": str(verified.get("content_hash") or ""), "status": "ready"}
        exact = ("logical_key", "source", "document_id", "operation_id", "document_version", "content_hash", "status")
        if any(str(verified.get(key)) != str(expected.get(key)) for key in exact):
            raise ValueError("recovery_identity_mismatch")
        progress = self._read("ingestion-progress.json", {"documents": []})
        known = {row.get("logical_key"): row for row in progress["documents"]}
        prior = known.get(record["logical_key"])
        if prior is not None and prior != expected:
            raise ValueError("recovery_logical_key_conflict")
        if prior is None:
            progress["documents"].append(expected)
            atomic_json(self.path("ingestion-progress.json"), progress)
        return expected

    async def verify_scope(self, adapter: RealEvaluationAdapter) -> None:
        allow = self._allowlist(); verdict = await adapter.verify_scope(frozenset(allow))
        if not all(verdict.get(key) is True for key in ("mongo", "chroma", "neo4j", "bm25")): self._fail("scope_verification_failed"); raise ValueError("scope_verification_failed")
        self._write_state(RunState.SCOPE_VERIFIED, scope=verdict)

    async def build_query_plans(self, adapter: RealEvaluationAdapter) -> list[dict[str, Any]]:
        if self._state().get("status") not in {RunState.SCOPE_VERIFIED.value, RunState.QUERY_PLANS_FROZEN.value}:
            raise ValueError("invalid_state_transition")
        questions = self.joined_questions(); expected_ids=[str(question.get("question_id") or "") for question in questions]
        if not expected_ids or len(set(expected_ids)) != len(expected_ids) or any(not value for value in expected_ids): self._fail("invalid_ground_truth_question_ids"); raise ValueError("invalid_ground_truth_question_ids")
        payload = self._read("query-plans.json", {"plans": []}); done = {str(plan.get("question_id") or "") for plan in payload["plans"]}
        if len(done) != len(payload["plans"]) or not done.issubset(set(expected_ids)): self._fail("invalid_existing_query_plans"); raise ValueError("invalid_existing_query_plans")
        for question in questions:
            question_id=str(question["question_id"])
            if question_id in done: continue
            try:
                plan = await adapter.build_query_plan(question, self.run_id)
                if str(plan.get("question_id") or "") != question_id: raise ValueError("query_plan_question_id_mismatch")
                plan = {"question_id": question_id, "queries": list(plan.get("queries", [])), "entities": list(plan.get("entities", [])), "keywords": list(plan.get("keywords", [])), "intent": plan.get("intent"), **{"plan_hash": _hash({key: plan.get(key) for key in ("question_id", "queries", "entities", "keywords", "intent")})}}
                if not plan["queries"] or not isinstance(plan["intent"], str) or not plan["intent"]: raise ValueError("invalid_query_plan")
                payload["plans"].append(plan); atomic_json(self.path("query-plans.json"), payload)
            except Exception as exc:
                self._write_state(RunState.FAILED, error=type(exc).__name__, failed_question_id=question_id)
                raise
        if len(payload["plans"]) != len(questions) or {plan["question_id"] for plan in payload["plans"]} != set(expected_ids): self._fail("query_plans_incomplete"); raise ValueError("query_plans_incomplete")
        manifest = self._manifest(); manifest["query_plans_hash"] = _hash(payload["plans"]); manifest["status"] = RunState.QUERY_PLANS_FROZEN.value; atomic_json(self.path("manifest.json"), manifest); self._write_state(RunState.QUERY_PLANS_FROZEN); return payload["plans"]

    async def trace_equivalence(self, adapter: RealEvaluationAdapter) -> None:
        self._require(RunState.QUERY_PLANS_FROZEN)
        state=self._state(); base_artifact=self.path("trace-equivalence.json")
        if base_artifact.exists() and state.get("g4_resume_authorized") is not True:
            raise ValueError("trace_equivalence_artifact_already_exists")
        artifact_path=base_artifact
        if base_artifact.exists():
            retry=1
            while self.path(f"trace-equivalence-retry-{retry}.json").exists(): retry += 1
            artifact_path=self.path(f"trace-equivalence-retry-{retry}.json")
        payload=self._read("query-plans.json", {"plans": []}); plans=payload.get("plans", [])
        manifest=self._manifest()
        if manifest.get("query_plans_hash") != _hash(plans):
            self._fail("query_plans_hash_mismatch"); raise ValueError("query_plans_hash_mismatch")
        plans_by_id={str(plan.get("question_id") or ""):plan for plan in plans}
        if len(plans_by_id) != len(plans) or set(TRACE_SAMPLE_QUESTION_IDS).difference(plans_by_id):
            self._fail("trace_sample_not_frozen"); raise ValueError("trace_sample_not_frozen")
        allow_payload=self._read("document-allowlist.json", {})
        allow=self._allowlist(); allow_hash=str(allow_payload.get("sha256") or "")
        if not allow_hash or allow_hash != _hash(allow):
            self._fail("allowlist_hash_mismatch"); raise ValueError("allowlist_hash_mismatch")
        rows=[]
        prior_failure_hash=state.get("g4_failure_artifact_sha256") if base_artifact.exists() else None
        for question_id in TRACE_SAMPLE_QUESTION_IDS:
            plan=plans_by_id[question_id]
            for variant in VARIANTS:
                off=await adapter.evaluate(plan, variant, frozenset(allow), trace=False, graph_trace=False)
                on=await adapter.evaluate(plan, variant, frozenset(allow), trace=True, graph_trace=variant in GRAPH_VARIANTS)
                off_output={key:list(off.get(key, [])) for key in TRACE_COMPARISON_FIELDS}
                on_output={key:list(on.get(key, [])) for key in TRACE_COMPARISON_FIELDS}
                expected_audit={"vector_searches":1,"bm25_searches":int(variant in {"bm25_vector_rrf","hybrid_v2_no_rerank"}),
                                "graph_queries":int(variant in GRAPH_VARIANTS),"reranker_calls":0}
                off_audit=dict(off.get("call_audit") or {}); on_audit=dict(on.get("call_audit") or {})
                audit_valid=all(off_audit.get(key)==value and on_audit.get(key)==value for key,value in expected_audit.items())
                equal=off_output == on_output and audit_valid
                row={"question_id":question_id,"variant":variant,"equivalent":equal,
                     "trace_enabled":True,"graph_trace_enabled":variant in GRAPH_VARIANTS,
                     "compared_outputs":{"off":off_output,"on":on_output},
                     "call_audit":{"expected":expected_audit,"off":off_audit,"on":on_audit},
                     "trace_diagnostics":dict(on.get("trace_diagnostics") or {})}
                if not equal:
                    row["differences"]={key:{"off":off_output[key],"on":on_output[key]}
                                        for key in TRACE_COMPARISON_FIELDS if off_output[key] != on_output[key]}
                    if not audit_valid: row["failure_reason"]="trace_call_audit_mismatch"
                rows.append(row)
                if not equal:
                    artifact={"run_id":self.run_id,"sample_question_ids":list(TRACE_SAMPLE_QUESTION_IDS),
                              "query_plans_hash":manifest["query_plans_hash"],"allowlist_hash":allow_hash,
                              "rows":rows,"passed":False}
                    if prior_failure_hash: artifact["prior_failure_artifact_sha256"]=prior_failure_hash
                    atomic_json(artifact_path, artifact)
                    self._fail("trace_equivalence_failed")
                    raise ValueError("trace_equivalence_failed")
        artifact={"run_id":self.run_id,"sample_question_ids":list(TRACE_SAMPLE_QUESTION_IDS),
                  "query_plans_hash":manifest["query_plans_hash"],"allowlist_hash":allow_hash,"rows":rows,"passed":True}
        if prior_failure_hash: artifact["prior_failure_artifact_sha256"]=prior_failure_hash
        atomic_json(artifact_path, artifact)
        self._write_state(RunState.TRACE_EQUIVALENCE_VERIFIED)

    def resume_g4_trace_equivalence_failure(self, *, query_plans_hash: str, allowlist_hash: str,
                                             failure_artifact_hash: str) -> None:
        """Reject legacy in-place recovery, which would mutate failure evidence."""
        raise ValueError("g4_in_place_resume_forbidden")

    def _validate_g4_recovery_inputs(self, *, query_plans_hash: str, allowlist_hash: str,
                                      failure_artifact_hash: str, retry_artifact_hash: str,
                                      snapshot_hash: str) -> dict[str, Any]:
        """Validate every frozen G4 input without changing the source run."""
        state=self._state(); artifact_path=self.path("trace-equivalence.json")
        if (state.get("status") != RunState.FAILED.value or state.get("error") != "trace_equivalence_failed"
                or not artifact_path.is_file()):
            raise ValueError("g4_resume_guard_state_failed")
        if any(self.path(name).exists() for name in ("results.json", "summary.json", "summary.md", "failures.json", "improvements.json", "regressions.json")):
            raise ValueError("g4_resume_guard_g5_artifacts_present")
        plans=self._read("query-plans.json", {"plans": []}).get("plans", [])
        manifest=self._manifest(); allow_payload=self._read("document-allowlist.json", {})
        if (manifest.get("query_plans_hash") != query_plans_hash or _hash(plans) != query_plans_hash
                or allow_payload.get("sha256") != allowlist_hash or allow_payload.get("sha256") != _hash(self._allowlist())):
            raise ValueError("g4_resume_guard_frozen_hash_mismatch")
        raw=artifact_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != failure_artifact_hash:
            raise ValueError("g4_resume_guard_artifact_hash_mismatch")
        artifact=json.loads(raw.decode("utf-8")); failed=[row for row in artifact.get("rows", []) if row.get("equivalent") is False]
        expected_failure=(len(failed)==1 and failed[0].get("question_id")=="Q01"
                          and failed[0].get("variant")=="bm25_vector_rrf")
        if (artifact.get("run_id") != self.run_id or artifact.get("passed") is not False
                or artifact.get("sample_question_ids") != list(TRACE_SAMPLE_QUESTION_IDS)
                or artifact.get("query_plans_hash") != query_plans_hash
                or artifact.get("allowlist_hash") != allowlist_hash or not expected_failure):
            raise ValueError("g4_resume_guard_failure_evidence_mismatch")
        retry_path=self.path("trace-equivalence-retry-1.json")
        if not retry_path.is_file() or self._file_sha256(retry_path) != retry_artifact_hash:
            raise ValueError("g4_resume_guard_retry_artifact_hash_mismatch")
        from services.query_embedding_snapshot import FrozenQueryEmbeddingSnapshot
        snapshot_path=self.path("query-embeddings.json")
        if not snapshot_path.is_file() or self._file_sha256(snapshot_path) != snapshot_hash:
            raise ValueError("g4_resume_guard_snapshot_file_hash_mismatch")
        snapshot=FrozenQueryEmbeddingSnapshot(json.loads(snapshot_path.read_text(encoding="utf-8")))
        records=snapshot.validate_partial(plans=plans,run_id=self.run_id,
                                          query_plans_hash=query_plans_hash,
                                          embedding=dict(manifest.get("embedding") or {}))
        if snapshot.payload.get("snapshot_hash") is None or len(records) != len(snapshot.planned_queries(plans)):
            raise ValueError("g4_resume_guard_snapshot_incomplete")
        return {
            "source_run_id":self.run_id,
            "query_plans_hash":query_plans_hash,
            "allowlist_hash":allowlist_hash,
            "failure_artifact_sha256":failure_artifact_hash,
            "retry_artifact_sha256":retry_artifact_hash,
            "snapshot_file_sha256":snapshot_hash,
            "snapshot_root_sha256":snapshot.payload["snapshot_hash"],
            "source_state_sha256":self._file_sha256(self.path("state.json")),
            "source_manifest_sha256":self._file_sha256(self.path("manifest.json")),
        }

    def prepare_g4_recovery(self, *, recovery_id: str, **guards: str) -> dict[str, Any]:
        identity=self._validate_g4_recovery_inputs(**guards)
        directory=self._recovery_directory(recovery_id)
        manifest_path=directory/"recovery-manifest.json"; state_path=directory/"state.json"
        expected={"schema_version":"g4-recovery-v1","recovery_id":recovery_id,**identity}
        if directory.exists():
            existing=json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
            if not isinstance(existing,dict) or any(existing.get(key)!=value for key,value in expected.items()):
                raise ValueError("g4_recovery_identity_conflict")
            return existing
        manifest={**expected,"created_at":datetime.now(UTC).isoformat()}
        atomic_json(manifest_path,manifest)
        atomic_json(state_path,{"recovery_id":recovery_id,"source_run_id":self.run_id,
                                "status":"RECOVERY_PREPARED","updated_at":datetime.now(UTC).isoformat()})
        return manifest

    async def trace_equivalence_recovery(self, adapter: RealEvaluationAdapter, *, recovery_id: str,
                                         **guards: str) -> dict[str, Any]:
        """Run G4 into an isolated directory while source evidence stays byte-identical."""
        identity=self._validate_g4_recovery_inputs(**guards)
        directory=self._recovery_directory(recovery_id)
        manifest_path=directory/"recovery-manifest.json"; state_path=directory/"state.json"
        if not manifest_path.is_file() or not state_path.is_file():
            raise ValueError("g4_recovery_not_prepared")
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        recovery_state=json.loads(state_path.read_text(encoding="utf-8"))
        if recovery_state.get("status") != "RECOVERY_PREPARED":
            raise ValueError("g4_recovery_not_runnable")
        if any(manifest.get(key)!=value for key,value in {"recovery_id":recovery_id,**identity}.items()):
            raise ValueError("g4_recovery_identity_conflict")
        artifact_path=directory/"trace-equivalence.json"
        if artifact_path.exists(): raise ValueError("g4_recovery_artifact_already_exists")
        protected={name:self._file_sha256(self.path(name)) for name in (
            "state.json","manifest.json","trace-equivalence.json","trace-equivalence-retry-1.json",
            "query-plans.json","document-allowlist.json","query-embeddings.json")}
        def write_state(status: str, **extra: Any) -> None:
            atomic_json(state_path,{"recovery_id":recovery_id,"source_run_id":self.run_id,"status":status,
                                    "updated_at":datetime.now(UTC).isoformat(),**extra})
        write_state("RUNNING")
        plans=self._read("query-plans.json",{"plans":[]})["plans"]
        plans_by_id={str(plan.get("question_id") or ""):plan for plan in plans}
        allow=frozenset(self._allowlist()); rows=[]
        try:
            for question_id in TRACE_SAMPLE_QUESTION_IDS:
                for variant in VARIANTS:
                    plan=plans_by_id[question_id]
                    off=await adapter.evaluate(plan,variant,allow,trace=False,graph_trace=False)
                    on=await adapter.evaluate(plan,variant,allow,trace=True,graph_trace=variant in GRAPH_VARIANTS)
                    off_output={key:list(off.get(key,[])) for key in TRACE_COMPARISON_FIELDS}
                    on_output={key:list(on.get(key,[])) for key in TRACE_COMPARISON_FIELDS}
                    expected={"vector_searches":1,"bm25_searches":int(variant in {"bm25_vector_rrf","hybrid_v2_no_rerank"}),
                              "graph_queries":int(variant in GRAPH_VARIANTS),"reranker_calls":0}
                    off_audit=dict(off.get("call_audit") or {}); on_audit=dict(on.get("call_audit") or {})
                    audit_valid=all(off_audit.get(key)==value and on_audit.get(key)==value for key,value in expected.items())
                    equivalent=off_output==on_output and audit_valid
                    row={"question_id":question_id,"variant":variant,"equivalent":equivalent,
                         "trace_enabled":True,"graph_trace_enabled":variant in GRAPH_VARIANTS,
                         "compared_outputs":{"off":off_output,"on":on_output},
                         "call_audit":{"expected":expected,"off":off_audit,"on":on_audit},
                         "trace_diagnostics":dict(on.get("trace_diagnostics") or {})}
                    if not equivalent:
                        row["differences"]={key:{"off":off_output[key],"on":on_output[key]}
                                            for key in TRACE_COMPARISON_FIELDS if off_output[key]!=on_output[key]}
                    rows.append(row)
                    if not equivalent:
                        payload={"source_run_id":self.run_id,"recovery_id":recovery_id,
                                 "sample_question_ids":list(TRACE_SAMPLE_QUESTION_IDS),"rows":rows,"passed":False,
                                 "query_plans_hash":identity["query_plans_hash"],"allowlist_hash":identity["allowlist_hash"],
                                 "prior_failure_artifact_sha256":identity["failure_artifact_sha256"]}
                        atomic_json(artifact_path,payload); write_state("FAILED",error="trace_equivalence_failed")
                        raise ValueError("trace_equivalence_failed")
            payload={"source_run_id":self.run_id,"recovery_id":recovery_id,
                     "sample_question_ids":list(TRACE_SAMPLE_QUESTION_IDS),"rows":rows,"passed":True,
                     "query_plans_hash":identity["query_plans_hash"],"allowlist_hash":identity["allowlist_hash"],
                     "prior_failure_artifact_sha256":identity["failure_artifact_sha256"]}
            atomic_json(artifact_path,payload); write_state("TRACE_EQUIVALENCE_VERIFIED")
            return payload
        except Exception as exc:
            current=json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
            if current.get("status") != "FAILED":
                write_state("FAILED",error=type(exc).__name__,failure_phase="trace_equivalence")
            raise
        finally:
            if any(self._file_sha256(self.path(name))!=digest for name,digest in protected.items()):
                raise RuntimeError("g4_source_run_mutated")

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
