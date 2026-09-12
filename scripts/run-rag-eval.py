"""Run the S4 retrieval comparison with explicit offline or authorized real modes."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
import uuid
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
    BenchmarkDocument, EvaluationFixture, RAGEvaluationRunner, SYNTHETIC_DOCUMENTS,
    atomic_json, load_evaluation_fixture, rescore_payload_v2, run_offline_sync,
    write_rescore_reports,
)
from services.vector_store import VectorStoreService  # noqa: E402


API_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_ENTERPRISE_BENCHMARK_DIR = PROJECT_ROOT / "benchmarks" / "enterprise_20docs_60q"
UPLOAD_RESPONSE_TIMEOUT_SECONDS = 180
OPERATION_POLL_TIMEOUT_SECONDS = 300
INGESTION_STATE_FILE = "ingestion-state.json"
_OPERATION_TERMINAL_STATUSES = {"succeeded", "failed", "cleanup_pending", "needs_reconciliation"}


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
    """Observe a known operation without ever replaying its write request."""
    deadline, latest = time.monotonic() + OPERATION_POLL_TIMEOUT_SECONDS, {}
    while time.monotonic() < deadline:
        status, body, elapsed = await _api_json(client, "GET", f"/api/document-operations/{operation_id}")
        latest = {"http_status": status, "elapsed_ms": elapsed, **body}
        atomic_json(output_dir / f"operation-{operation_id}.json", latest)
        # The API may still be reading a multipart request before the
        # coordinator creates its durable journal record.  A short-lived 404
        # is therefore ambiguous, not evidence that a retry is safe.
        if status == 404:
            await asyncio.sleep(1)
            continue
        if status != 200 or body.get("status") in _OPERATION_TERMINAL_STATUSES:
            return latest
        await asyncio.sleep(1)
    return {
        **latest,
        "operation_id": operation_id,
        "status": "ambiguous",
        "error_summary": "operation_poll_timeout",
    }


def _ingestion_state_path(output_dir: Path) -> Path:
    return output_dir / INGESTION_STATE_FILE


def _write_ingestion_state(output_dir: Path, state: dict[str, Any]) -> None:
    """Persist only durable, non-sensitive request identities atomically."""
    atomic_json(_ingestion_state_path(output_dir), state)


def _refresh_cleanup_manifest(state: dict[str, Any]) -> None:
    """Keep an exact, UUID-only cleanup set for every observed late success."""
    document_ids = [
        str(item["document_id"])
        for item in state.get("documents", [])
        if item.get("document_id") and item.get("request_state") in {"ready", "cleanup_pending", "deleted", "cleanup_failed"}
    ]
    state["cleanup_document_ids"] = sorted(set(document_ids))
    state["updated_at"] = datetime.now(UTC).isoformat()


def _new_upload_state(run_id: str, document: BenchmarkDocument, logical_key: str) -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    return {
        "run_id": run_id,
        "fixture_document_id": document.document_id,
        "logical_key": logical_key,
        "safe_filename": Path(document.filename).name,
        "content_hash": hashlib.sha256(document.content.encode("utf-8")).hexdigest(),
        "operation_id": str(uuid.uuid4()),
        "request_state": "planned",
        "request_started_at": None,
        "last_observed_status": None,
        "document_id": None,
        "version": None,
        "terminal_status": None,
        "cleanup_operation_id": None,
        "created_at": now,
        "updated_at": now,
        "error_summary": None,
    }


def _update_upload_state(state: dict[str, Any], **fields: Any) -> dict[str, Any]:
    """Advance one persisted document state without retaining response bodies."""
    state.update(fields)
    state["updated_at"] = datetime.now(UTC).isoformat()
    return state


def _record_upload_observation(state: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    """Translate an operation response into the runner's durable state machine."""
    status = str(observation.get("status") or "")
    document_id = observation.get("document_id")
    if document_id:
        state["document_id"] = str(document_id)
    if observation.get("version") is not None:
        state["version"] = observation["version"]
    if status == "succeeded" and state.get("document_id"):
        request_state, terminal = "ready", "succeeded"
    elif status == "succeeded":
        # A successful operation without its durable document UUID cannot be
        # safely cleaned up.  Preserve the handle and fail closed.
        request_state, terminal = "ambiguous", None
    elif status in {"failed", "needs_reconciliation"}:
        request_state, terminal = "failed", status
    elif status == "cleanup_pending":
        request_state, terminal = "cleanup_pending", status
    else:
        request_state, terminal = "ambiguous" if status == "ambiguous" else "processing", None
    return _update_upload_state(
        state,
        request_state=request_state,
        terminal_status=terminal,
        last_observed_status=status or None,
        error_summary=("operation_missing_document_id" if status == "succeeded" and not state.get("document_id") else observation.get("error_summary")),
    )


def _load_ingestion_state(output_dir: Path) -> dict[str, Any]:
    path = _ingestion_state_path(output_dir)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("ingestion recovery state is unavailable") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("documents"), list):
        raise ValueError("ingestion recovery state is invalid")
    return payload


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


def exact_cleanup_document_ids(created: list[dict[str, Any]]) -> tuple[str, ...]:
    """Return only exact upload IDs; this stays generic for any fixture size."""
    document_ids = tuple(str(item.get("document_id") or "") for item in created)
    if not document_ids or any(not document_id for document_id in document_ids):
        raise ValueError("exact cleanup requires non-empty uploaded document IDs")
    if len(set(document_ids)) != len(document_ids):
        raise ValueError("exact cleanup refuses duplicate document IDs")
    return document_ids


async def _run_real(run_id: str, fixture: EvaluationFixture | None = None) -> int:
    output_dir = PROJECT_ROOT / ".runtime" / "evaluation" / run_id
    if output_dir.exists() and any(output_dir.iterdir()):
        # A run directory is evidence.  A caller must use explicit recovery
        # rather than overwrite it with a second upload attempt.
        return 1
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

        documents = fixture.documents if fixture else tuple(
            BenchmarkDocument(name, filename, text) for name, filename, text in SYNTHETIC_DOCUMENTS
        )
        cases = fixture.cases if fixture else None
        ingestion_state = {
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "documents": [
                _new_upload_state(run_id, document, f"s4-eval-{run_id}-{document.document_id}")
                for document in documents
            ],
            "cleanup_document_ids": [],
        }
        _write_ingestion_state(output_dir, ingestion_state)
        for document in documents:
            name, filename, text = document.document_id, document.filename, document.content
            logical_key = f"s4-eval-{run_id}-{name}"
            state = next(item for item in ingestion_state["documents"] if item["fixture_document_id"] == name)
            _update_upload_state(state, request_state="request_sent", request_started_at=datetime.now(UTC).isoformat())
            _write_ingestion_state(output_dir, ingestion_state)
            status, body, elapsed = await _api_json(
                client,
                "POST",
                "/api/documents",
                headers={"X-Operation-Id": state["operation_id"]},
                data={"namespace": "default", "logical_key": logical_key},
                files={"file": (filename, text.encode("utf-8"), "text/plain")},
            )
            response_status = body.get("status") if status else None
            _update_upload_state(
                state,
                request_state="response_received" if status else "ambiguous",
                last_observed_status=response_status,
                document_id=body.get("document_id") or state.get("document_id"),
                version=body.get("version") if body.get("version") is not None else state.get("version"),
                error_summary=body.get("detail") if status >= 400 else body.get("error_summary"),
            )
            _write_ingestion_state(output_dir, ingestion_state)
            record = {
                "name": name,
                "logical_key": logical_key,
                "http_status": status,
                "elapsed_ms": elapsed,
                "operation_id": state["operation_id"],
                "document_id": state.get("document_id"),
                "version": state.get("version"),
                "content_hash": state["content_hash"],
                "status": response_status,
                "error_summary": state.get("error_summary"),
            }
            metadata["uploaded_documents"].append(record)
            atomic_json(output_dir / f"upload-{name}.json", record)
            atomic_json(output_dir / "safe-run-metadata.json", metadata)
            # A client timeout is deliberately not retried.  The durable,
            # caller-generated operation ID is now the sole recovery handle.
            operation = await _wait_operation(client, state["operation_id"], output_dir)
            _record_upload_observation(state, operation)
            _refresh_cleanup_manifest(ingestion_state)
            _write_ingestion_state(output_dir, ingestion_state)
            metadata["operations"].append(operation)
            record.update(
                document_id=state.get("document_id"), version=state.get("version"),
                status=state.get("last_observed_status"), error_summary=state.get("error_summary"),
            )
            atomic_json(output_dir / f"upload-{name}.json", record)
            atomic_json(output_dir / "safe-run-metadata.json", metadata)
            if operation.get("status") != "succeeded" or not state.get("document_id"):
                metadata["error_summary"] = "upload_operation_failed"
                atomic_json(output_dir / "safe-run-metadata.json", metadata)
                break
            doc_status, document, _ = await _api_json(client, "GET", f"/api/documents/{state['document_id']}")
            if doc_status != 200 or document.get("status") != "ready" or document.get("namespace") != "default" or document.get("logical_key") != logical_key or document.get("current_version") != 1:
                metadata["error_summary"] = "uploaded_document_verification_failed"
                break
            created.append({"document_id": state["document_id"], "fixture_document_id": name})
            atomic_json(output_dir / "safe-run-metadata.json", metadata)

        if len(created) == len(documents) and not metadata.get("error_summary"):
            scope = EvaluationScope.from_uploaded_document_ids(run_id, [item["document_id"] for item in created])
            if not scope.is_verified() or len(scope.allowed_document_ids) != len(documents):
                metadata["error_summary"] = "scope_verification_failed"
            else:
                metadata["scope_verified"], metadata["allowed_document_ids_count"] = True, len(scope.allowed_document_ids)
                atomic_json(output_dir / "safe-run-metadata.json", metadata)
                graph: KnowledgeGraphService | None = None
                try:
                    runner, _, graph, chat, embeddings = await _build_real_runner(scope)
                    if cases is not None:
                        runner = RAGEvaluationRunner(runner.agent, cases, scope=scope, offline=False)
                    payload = await runner.run(run_id=run_id, modes=("vector_only", "graph_rag"), output_root=output_dir.parent)
                    all_sources_scoped = all(source.get("document_id") in scope.allowed_document_ids for row in payload["results"] for source in row["sources"])
                    vector_only_graph_calls = sum(int(row.get("model_call_counts", {}).get("graph_calls") or 0) for row in payload["results"] if row["mode"] == "vector_only")
                    metadata["evaluation"] = {"results_count": len(payload["results"]), "all_sources_scoped": all_sources_scoped, "vector_only_graph_calls": vector_only_graph_calls, "chat_logical_calls": chat.call_count, "embedding_query_logical_calls": embeddings.query_calls, "chat_retry_observability": "SDK-internal retries are not externally observable"}
                    success = len(payload["results"]) == len(cases or runner.cases) * 2 and all_sources_scoped and vector_only_graph_calls == 0
                    if not success:
                        metadata["error_summary"] = "evaluation_scope_or_coverage_failed"
                except Exception as exc:
                    metadata["error_summary"] = _safe_error(exc)
                finally:
                    if graph is not None:
                        await graph.close()
                atomic_json(output_dir / "safe-run-metadata.json", metadata)

        for document_id in exact_cleanup_document_ids(created) if created else ():
            status, body, elapsed = await _api_json(client, "DELETE", f"/api/documents/{document_id}")
            cleanup = {"document_id": document_id, "http_status": status, "elapsed_ms": elapsed, "operation_id": body.get("operation_id"), "status": body.get("status")}
            state = next((item for item in ingestion_state["documents"] if item.get("document_id") == document_id), None)
            if state:
                _update_upload_state(
                    state,
                    request_state="cleanup_pending",
                    cleanup_operation_id=cleanup["operation_id"],
                    last_observed_status=cleanup["status"],
                )
                _write_ingestion_state(output_dir, ingestion_state)
            if status < 200 or status >= 300 or not cleanup["operation_id"]:
                cleanup["error_summary"] = body.get("detail") or "delete_failed"
            else:
                operation = await _wait_operation(client, cleanup["operation_id"], output_dir)
                cleanup["operation_status"] = operation.get("status")
                if operation.get("status") != "succeeded":
                    cleanup["error_summary"] = "cleanup_not_succeeded"
            if state:
                _update_upload_state(
                    state,
                    request_state="deleted" if cleanup.get("operation_status") == "succeeded" else "cleanup_failed",
                    terminal_status=cleanup.get("operation_status"),
                    error_summary=cleanup.get("error_summary"),
                )
                _write_ingestion_state(output_dir, ingestion_state)
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


async def _recover_real_ingestion(run_id: str) -> int:
    """Resume observation of a prior run without uploading any document again.

    This mode intentionally performs no POST.  It turns late server success
    into a durable cleanup candidate; a future authorized cleanup step can use
    only the exact document IDs recorded here.
    """
    output_dir = PROJECT_ROOT / ".runtime" / "evaluation" / run_id
    try:
        state = _load_ingestion_state(output_dir)
    except ValueError:
        return 1
    if state.get("run_id") != run_id:
        return 1
    changed = False
    async with httpx.AsyncClient(timeout=httpx.Timeout(UPLOAD_RESPONSE_TIMEOUT_SECONDS, connect=10)) as client:
        for document in state["documents"]:
            if document.get("request_state") in {"ready", "failed", "deleted", "cleanup_failed"}:
                continue
            operation_id = document.get("operation_id")
            if not isinstance(operation_id, str):
                _update_upload_state(document, request_state="ambiguous", error_summary="missing_operation_id")
                changed = True
                continue
            observation = await _wait_operation(client, operation_id, output_dir)
            _record_upload_observation(document, observation)
            changed = True
        if changed:
            _refresh_cleanup_manifest(state)
            _write_ingestion_state(output_dir, state)
    # Recovery is deliberately observation-only.  It never guesses a delete
    # target and it never replays the original multipart POST.
    return 0 if all(item.get("request_state") in {"ready", "failed", "deleted", "cleanup_failed"} for item in state["documents"]) else 1


def _rescore_v2(results_file: str) -> int:
    """Create a score-only v2 derivative without changing a historical run."""
    supplied = Path(results_file)
    # The script aligns its current working directory to ``python/`` for the
    # runtime settings.  CLI paths remain project-root relative for users.
    source = (supplied if supplied.is_absolute() else PROJECT_ROOT / supplied).resolve()
    evaluation_root = (PROJECT_ROOT / ".runtime" / "evaluation").resolve()
    if source.name != "results.json" or evaluation_root not in source.parents:
        print("rescoring is refused outside .runtime/evaluation/<run_id>/results.json", file=sys.stderr)
        return 1
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        rescored = rescore_payload_v2(payload)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"rescoring failed safely: {_safe_error(exc)}", file=sys.stderr)
        return 1
    output = source.parent.with_name(f"{source.parent.name}-rescored-v2")
    if output.exists() and not output.is_dir():
        print("rescoring is refused because the v2 output path is not a directory", file=sys.stderr)
        return 1
    try:
        write_rescore_reports(output, rescored)
    except OSError as exc:
        print(f"rescoring output failed safely: {_safe_error(exc)}", file=sys.stderr)
        return 1
    print(f"deterministic v2 rescoring completed: {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S4 Vector RAG vs GraphRAG evaluation runner")
    parser.add_argument("--mode", choices=("vector_only", "graph_rag", "both"), default="both")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--authorized-s4", action="store_true")
    parser.add_argument("--recover-run", metavar="RUN_ID", help="observe a prior ingestion state without replaying POST")
    parser.add_argument("--rescore", metavar="RESULTS_JSON")
    parser.add_argument("--benchmark-dir", metavar="DIR", help="read a reviewed, immutable benchmark fixture from DIR")
    parser.add_argument("--s4-baseline", action="store_true", help="use the historic built-in 4-document S4 fixture")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    if args.rescore:
        if args.offline or args.real or args.authorized_s4 or args.run_id or args.benchmark_dir or args.s4_baseline or args.recover_run:
            parser.error("--rescore cannot be combined with evaluation execution options")
        return _rescore_v2(args.rescore)
    if args.recover_run:
        if not args.real or not args.authorized_s4 or args.offline or args.run_id or args.benchmark_dir or args.s4_baseline:
            parser.error("--recover-run requires --real --authorized-s4 and cannot be combined with evaluation options")
        return asyncio.run(_recover_real_ingestion(args.recover_run))
    if not args.offline and not args.real:
        parser.error("real evaluation is refused without --real and --authorized-s4")
    if args.offline and args.real:
        parser.error("choose exactly one of --offline or --real")
    if args.real and not args.authorized_s4:
        parser.error("real evaluation is refused without --authorized-s4")
    if args.s4_baseline and args.benchmark_dir:
        parser.error("--s4-baseline cannot be combined with --benchmark-dir")
    run_id = args.run_id or f"s4-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    fixture: EvaluationFixture | None = None
    if not args.s4_baseline:
        supplied = Path(args.benchmark_dir) if args.benchmark_dir else DEFAULT_ENTERPRISE_BENCHMARK_DIR
        fixture_root = supplied if supplied.is_absolute() else PROJECT_ROOT / supplied
        try:
            fixture = load_evaluation_fixture(fixture_root)
        except (OSError, ValueError) as exc:
            print(f"benchmark fixture validation failed safely: {_safe_error(exc)}", file=sys.stderr)
            return 1
    if args.offline:
        try:
            payload = run_offline_sync(
                run_id, _parse_modes(args.mode), PROJECT_ROOT / ".runtime" / "evaluation", fixture=fixture,
            )
        except (ValueError, OSError) as exc:
            print(f"offline evaluation failed safely: {_safe_error(exc)}", file=sys.stderr)
            return 1
        print(f"offline fake-only evaluation completed: {PROJECT_ROOT / '.runtime' / 'evaluation' / run_id}")
        for mode, summary in payload["summary"].items():
            print(f"{mode}: questions={summary['questions']} accuracy={summary['question_accuracy']:.3f}")
        return 0
    return asyncio.run(_run_real(run_id, fixture))


if __name__ == "__main__":
    raise SystemExit(main())
