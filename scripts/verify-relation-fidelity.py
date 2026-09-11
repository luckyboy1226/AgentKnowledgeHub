"""Run the one authorized, document-scoped S4.4 relation fidelity check.

This tool contains only the two reviewed synthetic documents and three reviewed
questions. It never retries a write request, records only safe public results,
and deletes only the exact document IDs returned by its own uploads.
"""

from __future__ import annotations

import asyncio
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = PROJECT_ROOT / "python"
os.chdir(PYTHON_ROOT)
sys.path.insert(0, str(PYTHON_ROOT))
for _name in ("NO_PROXY", "no_proxy"):
    _entries = [item for item in os.environ.get(_name, "").split(",") if item]
    os.environ[_name] = ",".join(dict.fromkeys([*_entries, "localhost", "127.0.0.1", "::1"]))

from agents.qa_agent import QAAgent, RetrievalMode  # noqa: E402
from config import settings  # noqa: E402
from providers.factory import create_chat_provider, create_embedding_provider  # noqa: E402
from services.evaluation_scope import EvaluationScope  # noqa: E402
from services.knowledge_graph import KnowledgeGraphService  # noqa: E402
from services.rag_evaluation import EvaluationCase, RAGEvaluationRunner, atomic_json  # noqa: E402
from services.vector_store import VectorStoreService  # noqa: E402


API_BASE_URL = "http://127.0.0.1:8080"
DOCUMENTS = (
    ("a", "s4-relation-fidelity-a.txt", "北极星为天枢提供检索索引。天枢使用 Neo4j 和 Chroma。"),
    ("b", "s4-relation-fidelity-b.txt", "天枢依赖 Atlas 事件服务。赵启负责 Atlas 事件服务。"),
)
CASES = (
    EvaluationCase(
        "RF01", "北极星与天枢是什么关系？", "relation_fidelity", ("提供检索索引",),
        ("s4-relation-fidelity-a.txt",), forbidden_keywords=("北极星依赖天枢",),
        expected_relation_path=(("北极星", "PROVIDES_INDEX", "天枢", "forward"),),
    ),
    EvaluationCase(
        "RF02", "天枢依赖哪个服务？", "relation_fidelity", ("Atlas",),
        ("s4-relation-fidelity-b.txt",), forbidden_keywords=("北极星",),
        expected_relation_path=(("天枢", "DEPENDS_ON", "Atlas 事件服务", "forward"),),
    ),
    EvaluationCase(
        "RF03", "北极星是否依赖天枢？", "relation_fidelity", ("否", "提供检索索引"),
        ("s4-relation-fidelity-a.txt",), forbidden_keywords=("北极星依赖天枢",),
        requires_abstention=False,
        expected_relation_path=(("北极星", "PROVIDES_INDEX", "天枢", "forward"),),
        negation_target_terms=("北极星", "天枢"),
        positive_claim_patterns=(r"北极星\s*(?:确实|正是|就是|依赖)\s*天枢",),
    ),
)


class CountingChat:
    def __init__(self, provider: Any) -> None:
        self._provider = provider
        self.provider_name, self.model_name, self.call_count = provider.provider_name, provider.model_name, 0

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        self.call_count += 1
        return await self._provider.ainvoke(messages, **kwargs)


class CountingEmbeddings:
    def __init__(self, provider: Any) -> None:
        self._provider = provider
        self.provider_name, self.model_name, self.dimensions = provider.provider_name, provider.model_name, provider.dimensions
        self.document_calls = self.query_calls = 0

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return await self._provider.aembed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return await self._provider.aembed_query(text)


class CountingVectors:
    def __init__(self, service: VectorStoreService) -> None:
        self._service, self.embeddings, self.search_calls = service, service.embeddings, 0

    async def search(self, *args: Any, **kwargs: Any) -> Any:
        self.search_calls += 1
        return await self._service.search(*args, **kwargs)


class CountingGraph:
    def __init__(self, service: KnowledgeGraphService) -> None:
        self._service, self.calls = service, 0

    def safe_source(self, value: Any) -> str:
        return self._service.safe_source(value)

    async def get_neighbors(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return await self._service.get_neighbors(*args, **kwargs)


def safe_error(value: object) -> str:
    return type(value).__name__[:80]


def safe_response(body: dict[str, Any]) -> dict[str, Any]:
    fields = ("operation_id", "document_id", "version", "status", "changed", "content_hash", "namespace", "logical_key")
    return {field: body.get(field) for field in fields if field in body}


async def api_json(client: httpx.AsyncClient, method: str, path: str, **kwargs: Any) -> tuple[int, dict[str, Any], float]:
    started = time.monotonic()
    try:
        response = await client.request(method, f"{API_BASE_URL}{path}", **kwargs)
        body = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        return 0, {"error_summary": safe_error(exc)}, round((time.monotonic() - started) * 1000, 3)
    return response.status_code, body if isinstance(body, dict) else {"items": body}, round((time.monotonic() - started) * 1000, 3)


async def wait_operation(client: httpx.AsyncClient, operation_id: str, output_dir: Path) -> dict[str, Any]:
    deadline = time.monotonic() + 240
    latest: dict[str, Any] = {}
    while time.monotonic() < deadline:
        status, body, elapsed = await api_json(client, "GET", f"/api/document-operations/{operation_id}")
        latest = {"http_status": status, "elapsed_ms": elapsed, **safe_response(body)}
        atomic_json(output_dir / f"operation-{operation_id}.json", latest)
        if status != 200 or body.get("status") in {"succeeded", "failed", "cleanup_pending", "needs_reconciliation"}:
            return {**latest, "completed_steps": body.get("completed_steps", []), "compensation_steps": body.get("compensation_steps", [])}
        await asyncio.sleep(1)
    return {**latest, "status": "timed_out", "error_summary": "operation_poll_timeout"}


async def graph_counts(graph: KnowledgeGraphService, document_id: str | None = None) -> dict[str, int]:
    where, params = ("WHERE r.document_id = $document_id", {"document_id": document_id}) if document_id else ("WHERE r.document_id IS NOT NULL", {})
    rows = await graph.execute_cypher(f"MATCH ()-[r]->() {where} RETURN count(r) AS count", params)
    evidence = int(rows[0]["count"]) if rows else 0
    dv_where = "WHERE dv.document_id = $document_id" if document_id else ""
    rows = await graph.execute_cypher(f"MATCH (dv:DocumentVersion) {dv_where} RETURN count(dv) AS count", params)
    document_versions = int(rows[0]["count"]) if rows else 0
    mentions_query = (
        "MATCH (:DocumentVersion {document_id: $document_id})-[m:MENTIONS]->() RETURN count(m) AS count"
        if document_id else "MATCH (:DocumentVersion)-[m:MENTIONS]->() RETURN count(m) AS count"
    )
    rows = await graph.execute_cypher(mentions_query, params)
    return {
        "document_versions": document_versions,
        "mentions": int(rows[0]["count"]) if rows else 0,
        "evidence": evidence,
    }


async def baseline(exclude_document_ids: tuple[str, ...] = ()) -> dict[str, Any]:
    from pymongo import MongoClient
    import chromadb

    client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=10_000)
    try:
        database = client[settings.mongodb_database]
        query = {"document_id": {"$nin": list(exclude_document_ids)}} if exclude_document_ids else {}
        ids = sorted(str(row["document_id"]) for row in database.documents.find(query, {"document_id": 1, "_id": 0}))
        mongo = {
            "documents": database.documents.count_documents(query),
            "document_versions": database.document_versions.count_documents(query),
            "operations": database.document_operations.count_documents({}),
            "document_id_digest": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        }
    finally:
        client.close()
    chroma = chromadb.HttpClient(host=VectorStoreService.chroma_http_host(settings.chroma_host), port=settings.chroma_port)
    graph = KnowledgeGraphService()
    await graph.init()
    try:
        return {"mongo": mongo, "chroma_vectors": chroma.get_collection("knowledge_chunks").count(), "neo4j": await graph_counts(graph)}
    finally:
        await graph.close()


async def document_graph_evidence(graph: KnowledgeGraphService, document_ids: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for document_id in document_ids:
        for evidence in await graph.list_document_evidence(document_id, 1):
            rows.append({
                "document_id": document_id, "document_version": 1,
                "subject": evidence.get("head"), "predicate": evidence.get("predicate"),
                "object": evidence.get("tail"), "raw_predicate": evidence.get("raw_predicate"),
                "source": KnowledgeGraphService.safe_source(evidence.get("source", "")),
                "evidence_key": evidence.get("evidence_key"),
                "status": evidence.get("status"), "is_current": evidence.get("is_current"),
            })
    return sorted(rows, key=lambda row: (str(row["document_id"]), str(row["evidence_key"])))


async def verify(run_id: str) -> int:
    output_dir = PROJECT_ROOT / ".runtime" / "evaluation" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {"run_id": run_id, "authorized_scope": "two reviewed synthetic documents and three fixed questions", "uploads": [], "cleanup": [], "overall_pass": False}
    atomic_json(output_dir / "safe-run-metadata.json", metadata)
    created: list[dict[str, Any]] = []
    success = False
    try:
        metadata["baseline"] = await baseline()
        atomic_json(output_dir / "baseline.json", metadata["baseline"])
        async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=15), trust_env=False) as client:
            for name, filename, text in DOCUMENTS:
                logical_key = f"s4-relation-fidelity-{run_id}-{name}"
                status, body, elapsed = await api_json(client, "POST", "/api/documents", data={"namespace": "default", "logical_key": logical_key}, files={"file": (filename, text.encode("utf-8"), "text/plain")})
                record = {"name": name, "logical_key": logical_key, "http_status": status, "elapsed_ms": elapsed, **safe_response(body)}
                metadata["uploads"].append(record)
                atomic_json(output_dir / f"upload-{name}.json", record)
                if status < 200 or status >= 300 or not record.get("document_id") or not record.get("operation_id"):
                    metadata["error_summary"] = "upload_failed"
                    break
                operation = await wait_operation(client, str(record["operation_id"]), output_dir)
                record["operation_status"] = operation.get("status")
                if operation.get("status") != "succeeded":
                    metadata["error_summary"] = "upload_operation_failed"
                    break
                document_status, document, _ = await api_json(client, "GET", f"/api/documents/{record['document_id']}")
                if document_status != 200 or document.get("status") != "ready" or document.get("logical_key") != logical_key:
                    metadata["error_summary"] = "uploaded_document_verification_failed"
                    break
                created.append(record)
                atomic_json(output_dir / "safe-run-metadata.json", metadata)

            if len(created) == len(DOCUMENTS) and not metadata.get("error_summary"):
                ids = [str(row["document_id"]) for row in created]
                scope = EvaluationScope.from_uploaded_document_ids(run_id, ids)
                metadata["scope"] = {"scope_verified": scope.is_verified(), "allowed_document_ids": sorted(scope.allowed_document_ids)}
                if not scope.is_verified() or set(ids) != set(scope.allowed_document_ids):
                    metadata["error_summary"] = "scope_verification_failed"
                else:
                    graph = KnowledgeGraphService()
                    await graph.init()
                    try:
                        metadata["stored_graph_evidence"] = await document_graph_evidence(graph, ids)
                        atomic_json(output_dir / "stored-graph-evidence.json", {"edges": metadata["stored_graph_evidence"]})
                    finally:
                        await graph.close()
                    chat, embeddings = CountingChat(create_chat_provider(settings)), CountingEmbeddings(create_embedding_provider(settings))
                    vectors, graph = VectorStoreService(embeddings), KnowledgeGraphService()
                    await vectors.init(); await graph.init()
                    try:
                        runner = RAGEvaluationRunner(QAAgent(chat, CountingVectors(vectors), CountingGraph(graph), memory_service=None), CASES, scope=scope, offline=False)
                        payload = await runner.run(run_id=run_id, modes=(RetrievalMode.VECTOR_ONLY, RetrievalMode.GRAPH_RAG), output_root=output_dir.parent)
                        metadata["evaluation"] = {
                            "result_count": len(payload["results"]),
                            "requests": [
                                {"request_id": f"{run_id}-{row['question_id']}-{row['mode']}", "question_id": row["question_id"], "mode": row["mode"]}
                                for row in payload["results"]
                            ],
                            "chat_logical_calls": chat.call_count,
                            "embedding_document_calls": embeddings.document_calls,
                            "embedding_query_calls": embeddings.query_calls,
                            "vector_only_graph_calls": sum(int(row.get("model_call_counts", {}).get("graph_calls") or 0) for row in payload["results"] if row["mode"] == "vector_only"),
                            "all_sources_scoped": all(source.get("document_id") in scope.allowed_document_ids for row in payload["results"] for source in row["sources"]),
                        }
                        metadata["relation_checks"] = {
                            "provides_index_edge": any(edge.get("subject") == "北极星" and edge.get("predicate") == "PROVIDES_INDEX" and edge.get("object") == "天枢" for edge in metadata["stored_graph_evidence"]),
                            "depends_on_edge": any(edge.get("subject") == "天枢" and edge.get("predicate") == "DEPENDS_ON" and edge.get("object") == "Atlas 事件服务" for edge in metadata["stored_graph_evidence"]),
                            "forbidden_edge_absent": not any(edge.get("subject") == "北极星" and edge.get("predicate") == "DEPENDS_ON" and edge.get("object") == "天枢" for edge in metadata["stored_graph_evidence"]),
                        }
                        success = len(payload["results"]) == 6 and metadata["evaluation"]["vector_only_graph_calls"] == 0 and metadata["evaluation"]["all_sources_scoped"] and all(metadata["relation_checks"].values())
                    finally:
                        await graph.close()
    except Exception as exc:
        metadata["error_summary"] = safe_error(exc)
    finally:
        if created:
            async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=15), trust_env=False) as client:
                for record in created:
                    document_id = str(record["document_id"])
                    status, body, elapsed = await api_json(client, "DELETE", f"/api/documents/{document_id}")
                    cleanup = {"document_id": document_id, "http_status": status, "elapsed_ms": elapsed, **safe_response(body)}
                    if status < 200 or status >= 300 or not cleanup.get("operation_id"):
                        cleanup["error_summary"] = "delete_failed"
                    else:
                        operation = await wait_operation(client, str(cleanup["operation_id"]), output_dir)
                        cleanup["operation_status"] = operation.get("status")
                        if operation.get("status") != "succeeded":
                            cleanup["error_summary"] = "cleanup_not_succeeded"
                    metadata["cleanup"].append(cleanup)
                    atomic_json(output_dir / "safe-run-metadata.json", metadata)
            graph = KnowledgeGraphService(); await graph.init()
            try:
                vectors = VectorStoreService(create_embedding_provider(settings)); await vectors.init()
                for cleanup in metadata["cleanup"]:
                    document_id = cleanup["document_id"]
                    cleanup["chroma_vectors"] = len(vectors.list_document_vector_ids(document_id, None))
                    cleanup["neo4j"] = await graph_counts(graph, document_id)
                    async with httpx.AsyncClient(timeout=httpx.Timeout(30, connect=15), trust_env=False) as client:
                        status, document, _ = await api_json(client, "GET", f"/api/documents/{document_id}")
                    cleanup["mongo_tombstone_retained"] = status == 200 and document.get("status") == "deleted"
            finally:
                await graph.close()
        try:
            metadata["final"] = await baseline(tuple(str(item["document_id"]) for item in created))
        except Exception:
            metadata["final"] = {"error_summary": "baseline_unavailable"}
        cleanup_ok = len(metadata["cleanup"]) == len(created) and all(item.get("operation_status") == "succeeded" and item.get("chroma_vectors") == 0 and not any(item.get("neo4j", {}).values()) and item.get("mongo_tombstone_retained") for item in metadata["cleanup"])
        if metadata.get("baseline") and metadata.get("final"):
            metadata["other_document_digest_unchanged"] = metadata["baseline"]["mongo"]["document_id_digest"] == metadata["final"]["mongo"]["document_id_digest"]
            cleanup_ok = cleanup_ok and metadata["other_document_digest_unchanged"]
        metadata["overall_pass"] = bool(success and cleanup_ok and not metadata.get("error_summary"))
        metadata["completed_at_utc"] = datetime.now(UTC).isoformat()
        atomic_json(output_dir / "safe-run-metadata.json", metadata)
    return 0 if metadata["overall_pass"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the authorized two-document S4.4 relation check")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    run_id = args.run_id or f"s4-rf-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    return asyncio.run(verify(run_id))


if __name__ == "__main__":
    raise SystemExit(main())
