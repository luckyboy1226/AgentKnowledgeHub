"""Run the S4 retrieval comparison with explicit offline or authorized real modes."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = PROJECT_ROOT / "python"
# Settings intentionally use a relative ``.env`` path, the same working
# directory as the backend process.  Keep the evaluation runner aligned with
# that deployment convention without reading or printing the file itself.
os.chdir(PYTHON_ROOT)
_LOCAL_NO_PROXY = ("localhost", "127.0.0.1", "::1")
_no_proxy_values = [value.strip() for value in os.environ.get("NO_PROXY", "").split(",") if value.strip()]
for _host in _LOCAL_NO_PROXY:
    if _host not in _no_proxy_values:
        _no_proxy_values.append(_host)
os.environ["NO_PROXY"] = ",".join(_no_proxy_values)
os.environ["no_proxy"] = os.environ["NO_PROXY"]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from agents.qa_agent import QAAgent, RetrievalMode  # noqa: E402
from config import settings  # noqa: E402
from providers.factory import create_chat_provider, create_embedding_provider  # noqa: E402
from services.evaluation_scope import EvaluationScope  # noqa: E402
from services.knowledge_graph import KnowledgeGraphService  # noqa: E402
from services.rag_evaluation import (  # noqa: E402
    RAGEvaluationRunner, SYNTHETIC_DOCUMENTS, atomic_json, run_offline_sync,
)
from services.vector_store import VectorStoreService  # noqa: E402


API_BASE_URL = "http://127.0.0.1:8080"


def _parse_modes(value: str) -> tuple[RetrievalMode, ...]:
    return (RetrievalMode.VECTOR_ONLY, RetrievalMode.GRAPH_RAG) if value == "both" else (RetrievalMode(value),)


def _safe_error(value: object) -> str:
    return type(value).__name__[:80]


class CountingChatProvider:
    def __init__(self, provider: Any) -> None:
        self._provider, self.provider_name, self.model_name, self.call_count = provider, provider.provider_name, provider.model_name, 0

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        self.call_count += 1
        return await self._provider.ainvoke(messages, **kwargs)


class CountingEmbeddingProvider:
    def __init__(self, provider: Any) -> None:
        self._provider, self.provider_name, self.model_name, self.dimensions = provider, provider.provider_name, provider.model_name, provider.dimensions
        self.document_calls = 0
        self.query_calls = 0

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return await self._provider.aembed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return await self._provider.aembed_query(text)


class CountingVectorStore:
    def __init__(self, service: VectorStoreService) -> None:
        self._service, self.embeddings, self.search_calls = service, service.embeddings, 0

    async def search(self, *args: Any, **kwargs: Any) -> Any:
        self.search_calls += 1
        return await self._service.search(*args, **kwargs)


class CountingKnowledgeGraph:
    def __init__(self, service: KnowledgeGraphService) -> None:
        self._service, self.calls = service, 0

    def safe_source(self, value: Any) -> str:
        return self._service.safe_source(value)

    async def get_neighbors(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return await self._service.get_neighbors(*args, **kwargs)


async def _api_json(client: Any, method: str, path: str, **kwargs: Any) -> tuple[int, dict[str, Any], float]:
    started = time.monotonic()
    try:
        response = await client.request(method, f"{API_BASE_URL}{path}", **kwargs)
    except httpx.HTTPError as exc:
        elapsed_ms = round((time.monotonic() - started) * 1000, 3)
        return 0, {"error_summary": _safe_error(exc)}, elapsed_ms
    elapsed_ms = round((time.monotonic() - started) * 1000, 3)
    try:
        body = response.json()
    except ValueError:
        body = {"error_summary": "non_json_response"}
    return response.status_code, body if isinstance(body, dict) else {"items": body}, elapsed_ms


async def _wait_operation(client: Any, operation_id: str, output_dir: Path) -> dict[str, Any]:
    deadline, latest = time.monotonic() + 180, {}
    while time.monotonic() < deadline:
        status, body, elapsed = await _api_json(client, "GET", f"/api/document-operations/{operation_id}")
        latest = {"http_status": status, "elapsed_ms": elapsed, **body}
        atomic_json(output_dir / f"operation-{operation_id}.json", latest)
        if status != 200 or body.get("status") in {"succeeded", "failed", "cleanup_pending", "needs_reconciliation"}:
            return latest
        await asyncio.sleep(1)
    return {**latest, "status": "timed_out", "error_summary": "operation_poll_timeout"}


async def _graph_counts(graph: KnowledgeGraphService, document_id: str) -> dict[str, int]:
    async def count(cypher: str) -> int:
        rows = await graph.execute_cypher(cypher, {"document_id": document_id})
        return int(rows[0].get("count", 0)) if rows else 0
    return {
        "document_versions": await count("MATCH (dv:DocumentVersion {document_id: $document_id}) RETURN count(dv) AS count"),
        "mentions": await count("MATCH (:DocumentVersion {document_id: $document_id})-[m:MENTIONS]->() RETURN count(m) AS count"),
        "evidence": await count("MATCH ()-[r]->() WHERE r.document_id = $document_id RETURN count(r) AS count"),
    }


async def _build_real_runner(scope: EvaluationScope):
    if not settings.has_usable_llm_key:
        raise RuntimeError("configured_chat_provider_unavailable")
    chat = CountingChatProvider(create_chat_provider(settings))
    embeddings = CountingEmbeddingProvider(create_embedding_provider(settings))
    vectors, graph = VectorStoreService(embeddings), KnowledgeGraphService()
    await vectors.init()
    await graph.init()
    runner = RAGEvaluationRunner(
        QAAgent(chat, CountingVectorStore(vectors), CountingKnowledgeGraph(graph), memory_service=None),
        scope=scope, offline=False,
    )
    return runner, vectors, graph, chat, embeddings


async def _run_real(run_id: str) -> int:
    output_dir = PROJECT_ROOT / ".runtime" / "evaluation" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {"run_id": run_id, "mode": "authorized_real_s4", "started_at_utc": datetime.now(UTC).isoformat(), "uploaded_documents": [], "operations": [], "scope_verified": False, "cleanup": []}
    atomic_json(output_dir / "safe-run-metadata.json", metadata)
    created: list[dict[str, Any]] = []
    success = False

    async with httpx.AsyncClient(timeout=httpx.Timeout(180, connect=10)) as client:
        stats_status, stats, _ = await _api_json(client, "GET", "/api/admin/stats")
        docs_status, docs, _ = await _api_json(client, "GET", "/api/ingest/documents")
        if stats_status != 200 or docs_status != 200:
            metadata["error_summary"] = "baseline_api_unavailable"
            atomic_json(output_dir / "safe-run-metadata.json", metadata)
            return 1
        metadata["baseline"] = {"vector_total": stats.get("vector_store", {}).get("total_vectors"), "graph_entities": stats.get("knowledge_graph", {}).get("total_entities"), "graph_relations": stats.get("knowledge_graph", {}).get("total_relations"), "existing_document_count": len(docs.get("items", docs if isinstance(docs, list) else []))}
        atomic_json(output_dir / "safe-run-metadata.json", metadata)

        for name, filename, text in SYNTHETIC_DOCUMENTS:
            logical_key = f"s4-eval-{run_id}-{name}"
            status, body, elapsed = await _api_json(client, "POST", "/api/documents", data={"namespace": "default", "logical_key": logical_key}, files={"file": (filename, text.encode("utf-8"), "text/plain")})
            record = {"name": name, "logical_key": logical_key, "http_status": status, "elapsed_ms": elapsed, "operation_id": body.get("operation_id"), "document_id": body.get("document_id"), "version": body.get("version"), "content_hash": body.get("content_hash"), "status": body.get("status"), "error_summary": body.get("detail") if status >= 400 else None}
            metadata["uploaded_documents"].append(record)
            atomic_json(output_dir / f"upload-{name}.json", record)
            atomic_json(output_dir / "safe-run-metadata.json", metadata)
            if status < 200 or status >= 300 or not record["document_id"] or not record["operation_id"]:
                metadata["error_summary"] = "upload_failed"
                break
            operation = await _wait_operation(client, record["operation_id"], output_dir)
            metadata["operations"].append(operation)
            if operation.get("status") != "succeeded":
                metadata["error_summary"] = "upload_operation_failed"
                break
            doc_status, document, _ = await _api_json(client, "GET", f"/api/documents/{record['document_id']}")
            if doc_status != 200 or document.get("status") != "ready" or document.get("namespace") != "default" or document.get("logical_key") != logical_key or document.get("current_version") != 1:
                metadata["error_summary"] = "uploaded_document_verification_failed"
                break
            created.append(record)
            atomic_json(output_dir / "safe-run-metadata.json", metadata)

        if len(created) == len(SYNTHETIC_DOCUMENTS) and not metadata.get("error_summary"):
            scope = EvaluationScope.from_uploaded_document_ids(run_id, [item["document_id"] for item in created])
            if not scope.is_verified() or len(scope.allowed_document_ids) != 4:
                metadata["error_summary"] = "scope_verification_failed"
            else:
                metadata["scope_verified"], metadata["allowed_document_ids_count"] = True, len(scope.allowed_document_ids)
                atomic_json(output_dir / "safe-run-metadata.json", metadata)
                graph: KnowledgeGraphService | None = None
                try:
                    runner, _, graph, chat, embeddings = await _build_real_runner(scope)
                    payload = await runner.run(run_id=run_id, modes=("vector_only", "graph_rag"), output_root=output_dir.parent)
                    all_sources_scoped = all(source.get("document_id") in scope.allowed_document_ids for row in payload["results"] for source in row["sources"])
                    vector_only_graph_calls = sum(int(row.get("model_call_counts", {}).get("graph_calls") or 0) for row in payload["results"] if row["mode"] == "vector_only")
                    metadata["evaluation"] = {"results_count": len(payload["results"]), "all_sources_scoped": all_sources_scoped, "vector_only_graph_calls": vector_only_graph_calls, "chat_logical_calls": chat.call_count, "embedding_query_logical_calls": embeddings.query_calls, "chat_retry_observability": "SDK-internal retries are not externally observable"}
                    success = len(payload["results"]) == 24 and all_sources_scoped and vector_only_graph_calls == 0
                    if not success:
                        metadata["error_summary"] = "evaluation_scope_or_coverage_failed"
                except Exception as exc:
                    metadata["error_summary"] = _safe_error(exc)
                finally:
                    if graph is not None:
                        await graph.close()
                atomic_json(output_dir / "safe-run-metadata.json", metadata)

        for record in created:
            document_id = record["document_id"]
            status, body, elapsed = await _api_json(client, "DELETE", f"/api/documents/{document_id}")
            cleanup = {"document_id": document_id, "http_status": status, "elapsed_ms": elapsed, "operation_id": body.get("operation_id"), "status": body.get("status")}
            if status < 200 or status >= 300 or not cleanup["operation_id"]:
                cleanup["error_summary"] = body.get("detail") or "delete_failed"
            else:
                operation = await _wait_operation(client, cleanup["operation_id"], output_dir)
                cleanup["operation_status"] = operation.get("status")
                if operation.get("status") != "succeeded":
                    cleanup["error_summary"] = "cleanup_not_succeeded"
            metadata["cleanup"].append(cleanup)
            atomic_json(output_dir / "safe-run-metadata.json", metadata)

        if created:
            check_vectors, check_graph = VectorStoreService(create_embedding_provider(settings)), KnowledgeGraphService()
            await check_vectors.init()
            await check_graph.init()
            try:
                for cleanup in metadata["cleanup"]:
                    document_id = cleanup["document_id"]
                    cleanup["chroma_vector_count"] = len(check_vectors.list_document_vector_ids(document_id, None))
                    cleanup["neo4j"] = await _graph_counts(check_graph, document_id)
                    doc_status, document, _ = await _api_json(client, "GET", f"/api/documents/{document_id}")
                    cleanup["mongo_status"] = document.get("status") if doc_status == 200 else None
                    cleanup["mongo_tombstone_retained"] = doc_status == 200 and document.get("status") == "deleted"
                    if cleanup.get("chroma_vector_count") or any(cleanup["neo4j"].values()) or not cleanup["mongo_tombstone_retained"]:
                        cleanup["error_summary"] = "exact_cleanup_verification_failed"
            finally:
                await check_graph.close()
        final_status, final_stats, _ = await _api_json(client, "GET", "/api/admin/stats")
        metadata["final"] = {"vector_total": final_stats.get("vector_store", {}).get("total_vectors") if final_status == 200 else None, "graph_entities": final_stats.get("knowledge_graph", {}).get("total_entities") if final_status == 200 else None, "graph_relations": final_stats.get("knowledge_graph", {}).get("total_relations") if final_status == 200 else None}

    cleanup_ok = len(metadata["cleanup"]) == len(created) and all(item.get("operation_status") == "succeeded" and item.get("chroma_vector_count") == 0 and not any(item.get("neo4j", {}).values()) and item.get("mongo_tombstone_retained") for item in metadata["cleanup"])
    metadata["completed_at_utc"] = datetime.now(UTC).isoformat()
    metadata["overall_pass"] = bool(success and cleanup_ok and not metadata.get("error_summary"))
    atomic_json(output_dir / "safe-run-metadata.json", metadata)
    return 0 if metadata["overall_pass"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S4 Vector RAG vs GraphRAG evaluation runner")
    parser.add_argument("--mode", choices=("vector_only", "graph_rag", "both"), default="both")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--authorized-s4", action="store_true")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    if not args.offline and not args.real:
        parser.error("real evaluation is refused without --real and --authorized-s4")
    if args.offline and args.real:
        parser.error("choose exactly one of --offline or --real")
    if args.real and not args.authorized_s4:
        parser.error("real evaluation is refused without --authorized-s4")
    run_id = args.run_id or f"s4-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    if args.offline:
        try:
            payload = run_offline_sync(run_id, _parse_modes(args.mode), PROJECT_ROOT / ".runtime" / "evaluation")
        except (ValueError, OSError) as exc:
            print(f"offline evaluation failed safely: {_safe_error(exc)}", file=sys.stderr)
            return 1
        print(f"offline fake-only evaluation completed: {PROJECT_ROOT / '.runtime' / 'evaluation' / run_id}")
        for mode, summary in payload["summary"].items():
            print(f"{mode}: questions={summary['questions']} accuracy={summary['question_accuracy']:.3f}")
        return 0
    return asyncio.run(_run_real(run_id))


if __name__ == "__main__":
    raise SystemExit(main())
