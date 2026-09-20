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
    BenchmarkDocument, EvaluationFixture, EVALUATION_CASES, RAGEvaluationRunner, SYNTHETIC_DOCUMENTS,
    atomic_json, load_evaluation_fixture, rescore_payload_v2, run_offline_sync,
    select_evaluation_subset,
    write_rescore_reports,
)
from services.vector_store import VectorStoreService  # noqa: E402


API_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_ENTERPRISE_BENCHMARK_DIR = PROJECT_ROOT / "benchmarks" / "enterprise_20docs_60q"
UPLOAD_RESPONSE_TIMEOUT_SECONDS = 180
OPERATION_POLL_TIMEOUT_SECONDS = 300
INGESTION_STATE_FILE = "ingestion-state.json"
RUNNER_SCHEMA_VERSION = "s4.6b-subset-v1"
_OPERATION_TERMINAL_STATUSES = {"succeeded", "failed", "cleanup_pending", "needs_reconciliation"}


def _parse_modes(value: str) -> tuple[RetrievalMode, ...]:
    return (RetrievalMode.VECTOR_ONLY, RetrievalMode.GRAPH_RAG) if value == "both" else (RetrievalMode(value),)


def _parse_selection(value: str | None, *, kind: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    # Preserve the caller's values for duplicate detection; selection itself
    # is later reordered by the immutable fixture, never by CLI order.
    values = tuple(part.strip() for part in value.split(","))
    if not values or any(not part for part in values):
        raise ValueError(f"{kind}_selection_empty")
    return values


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


def _fixture_fingerprint(documents: tuple[BenchmarkDocument, ...], cases: tuple[Any, ...] | None) -> str:
    """Bind recovery to one reviewed fixture without persisting its text."""
    safe_manifest = {
        "documents": [
            {
                "fixture_document_id": document.document_id,
                "filename": Path(document.filename).name,
                "content_hash": hashlib.sha256(document.content.encode("utf-8")).hexdigest(),
            }
            for document in documents
        ],
        "question_ids": [str(case.question_id) for case in cases or EVALUATION_CASES],
    }
    encoded = json.dumps(safe_manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _selection_fingerprint(
    *, fixture_fingerprint: str, documents: tuple[BenchmarkDocument, ...], cases: tuple[Any, ...], modes: tuple[RetrievalMode, ...],
) -> str:
    """Bind recovery to the exact safe selection, not CLI ordering or text."""
    payload = {
        "fixture_fingerprint": fixture_fingerprint,
        "document_ids": sorted(item.document_id for item in documents),
        "question_ids": sorted(str(item.question_id) for item in cases),
        "modes": sorted(item.value for item in modes),
        "scorer_version": "deterministic-v2",
        "trace_schema_version": "s4.6a-v1",
        "runner_schema_version": RUNNER_SCHEMA_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _normalised_upload_payload(raw: bytes) -> bytes:
    """Apply the reviewed fixture-to-upload normalization exactly once."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("fixture_document_not_utf8") from exc
    text = text.replace("\r\n", "\n")
    if text.endswith("\n"):
        text = text[:-1]
    return text.encode("utf-8")


def _document_dual_hashes(fixture: EvaluationFixture | None, documents: tuple[BenchmarkDocument, ...]) -> list[dict[str, str]]:
    """Safe state metadata: raw fixture bytes and actual normalized upload bytes."""
    if fixture is None:
        return []
    rows: list[dict[str, str]] = []
    for document in documents:
        raw = (fixture.root / "documents" / document.filename).read_bytes()
        normalized = _normalised_upload_payload(raw)
        # The JSON fixture and exact upload bytes must both satisfy the frozen
        # body contract.  Do not silently upload a differently-normalized body.
        if normalized != document.content.encode("utf-8"):
            raise ValueError("fixture_upload_payload_mismatch")
        rows.append({"fixture_document_id": document.document_id, "fixture_file_sha256": hashlib.sha256(raw).hexdigest(), "upload_payload_sha256": hashlib.sha256(normalized).hexdigest()})
    return rows


def _load_completed_results(output_dir: Path) -> list[dict[str, Any]]:
    """Read only the durable report rows; absent/invalid reports mean none."""
    try:
        payload = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return []
    rows = payload.get("results") if isinstance(payload, dict) else None
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _complete_question_ids(results: list[dict[str, Any]], modes: tuple[RetrievalMode, ...]) -> list[str]:
    expected = {mode.value for mode in modes}
    by_question: dict[str, set[str]] = {}
    for row in results:
        question_id, mode = str(row.get("question_id") or ""), str(row.get("mode") or "")
        if question_id and mode in expected:
            by_question.setdefault(question_id, set()).add(mode)
    return sorted(question_id for question_id, found in by_question.items() if found == expected)


def _ready_documents_from_state(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item for item in state.get("documents", [])
        if item.get("request_state") == "ready" and isinstance(item.get("document_id"), str)
    ]


async def _verify_ready_documents(client: Any, documents: list[dict[str, Any]]) -> bool:
    """Fail closed if a recovery scope no longer points at ready documents."""
    for item in documents:
        status, body, _ = await _api_json(client, "GET", f"/api/documents/{item['document_id']}")
        if status != 200 or body.get("status") != "ready":
            return False
        if body.get("logical_key") != item.get("logical_key") or body.get("namespace") != "default":
            return False
    return True


async def _run_or_resume_evaluation(
    *,
    run_id: str,
    output_dir: Path,
    state: dict[str, Any],
    metadata: dict[str, Any],
    fixture: EvaluationFixture | None,
    documents: tuple[BenchmarkDocument, ...],
    cases: tuple[Any, ...] | None,
) -> bool:
    """Evaluate only missing complete question pairs for a verified saved scope."""
    ready = _ready_documents_from_state(state)
    if len(ready) != len(documents):
        metadata["error_summary"] = "recovery_scope_documents_not_ready"
        return False
    allowed_ids = sorted(str(item["document_id"]) for item in ready)
    saved_scope = state.get("scope")
    if saved_scope is not None and (
        not isinstance(saved_scope, dict)
        or saved_scope.get("scope_verified") is not True
        or sorted(saved_scope.get("allowed_document_ids") or []) != allowed_ids
    ):
        metadata["error_summary"] = "saved_scope_mismatch"
        return False
    scope = EvaluationScope.from_uploaded_document_ids(run_id, allowed_ids)
    state["scope"] = {
        "scope_verified": scope.is_verified(),
        "allowed_document_ids": allowed_ids,
    }
    evaluation = state.setdefault("evaluation", {})
    evaluation.update({"status": "running", "interruption": None})
    _write_ingestion_state(output_dir, state)

    graph: KnowledgeGraphService | None = None
    try:
        runner, _, graph, chat, embeddings = await _build_real_runner(scope)
        runner = RAGEvaluationRunner(
            runner.agent,
            cases or runner.cases,
            scope=scope,
            offline=False,
            ingestion_trace_root=PROJECT_ROOT / ".runtime" / "evaluation",
        )
        payload = await runner.run(
            run_id=run_id,
            modes=("vector_only", "graph_rag"),
            output_root=output_dir.parent,
            existing_results=_load_completed_results(output_dir),
        )
        all_sources_scoped = all(
            source.get("document_id") in scope.allowed_document_ids
            for row in payload["results"]
            for source in row["sources"]
        )
        vector_only_graph_calls = sum(
            int(row.get("model_call_counts", {}).get("graph_calls") or 0)
            for row in payload["results"] if row["mode"] == "vector_only"
        )
        expected_result_count = len(cases or runner.cases) * 2
        completed_ids = _complete_question_ids(payload["results"], (RetrievalMode.VECTOR_ONLY, RetrievalMode.GRAPH_RAG))
        evaluation.update({
            "status": "completed" if len(payload["results"]) == expected_result_count else "interrupted",
            "completed_question_ids": completed_ids,
            "completed_result_count": len(payload["results"]),
            "expected_result_count": expected_result_count,
        })
        metadata["evaluation"] = {
            "results_count": len(payload["results"]),
            "all_sources_scoped": all_sources_scoped,
            "vector_only_graph_calls": vector_only_graph_calls,
            "chat_logical_calls_current_process": chat.call_count,
            "embedding_query_logical_calls_current_process": embeddings.query_calls,
            "chat_retry_observability": "SDK-internal retries are not externally observable",
        }
        if len(payload["results"]) != expected_result_count or not all_sources_scoped or vector_only_graph_calls:
            metadata["error_summary"] = "evaluation_scope_or_coverage_failed"
            return False
        return True
    except Exception as exc:
        # The per-question report is already atomically persisted by the
        # runner.  Preserve scope and exact IDs for --recover-run.
        evaluation.update({
            "status": "interrupted",
            "completed_question_ids": _complete_question_ids(
                _load_completed_results(output_dir), (RetrievalMode.VECTOR_ONLY, RetrievalMode.GRAPH_RAG)
            ),
            "completed_result_count": len(_load_completed_results(output_dir)),
            "interruption": _safe_error(exc),
        })
        metadata["error_summary"] = "evaluation_interrupted"
        metadata["interruption_type"] = _safe_error(exc)
        return False
    finally:
        _write_ingestion_state(output_dir, state)
        if graph is not None:
            await graph.close()


def _safe_cleanup_targets(state: dict[str, Any]) -> tuple[str, ...]:
    """Return only exact ready UUIDs persisted by this run's ingestion state."""
    ready = _ready_documents_from_state(state)
    targets = exact_cleanup_document_ids(ready)
    try:
        targets = tuple(str(uuid.UUID(document_id)) for document_id in targets)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("cleanup requires UUID document IDs") from exc
    scope = state.get("scope")
    if isinstance(scope, dict) and scope.get("allowed_document_ids") is not None:
        if sorted(scope.get("allowed_document_ids") or []) != sorted(targets):
            raise ValueError("cleanup scope does not match exact ready documents")
    return targets


async def _cleanup_saved_run(run_id: str) -> int:
    """Explicitly delete only IDs durably recorded for one interrupted run."""
    output_dir = PROJECT_ROOT / ".runtime" / "evaluation" / run_id
    try:
        state = _load_ingestion_state(output_dir)
        if state.get("run_id") != run_id:
            return 1
        targets = _safe_cleanup_targets(state)
    except ValueError:
        return 1
    if not targets:
        return 1
    metadata_path = output_dir / "safe-run-metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        metadata = {"run_id": run_id, "cleanup": []}
    metadata.setdefault("cleanup", [])
    state["lifecycle"] = "cleanup_running"
    _write_ingestion_state(output_dir, state)

    async with httpx.AsyncClient(timeout=httpx.Timeout(UPLOAD_RESPONSE_TIMEOUT_SECONDS, connect=10)) as client:
        for document_id in targets:
            item = next(item for item in state["documents"] if item.get("document_id") == document_id)
            if item.get("request_state") == "deleted":
                continue
            status, body, elapsed = await _api_json(client, "DELETE", f"/api/documents/{document_id}")
            cleanup = {
                "document_id": document_id,
                "http_status": status,
                "elapsed_ms": elapsed,
                "operation_id": body.get("operation_id"),
                "status": body.get("status"),
            }
            _update_upload_state(item, request_state="cleanup_pending", cleanup_operation_id=cleanup["operation_id"])
            _write_ingestion_state(output_dir, state)
            if status < 200 or status >= 300 or not cleanup["operation_id"]:
                cleanup["error_summary"] = body.get("detail") or "delete_failed"
            else:
                operation = await _wait_operation(client, cleanup["operation_id"], output_dir)
                cleanup["operation_status"] = operation.get("status")
                if cleanup["operation_status"] != "succeeded":
                    cleanup["error_summary"] = "cleanup_not_succeeded"
            _update_upload_state(
                item,
                request_state="deleted" if cleanup.get("operation_status") == "succeeded" else "cleanup_failed",
                terminal_status=cleanup.get("operation_status"),
                error_summary=cleanup.get("error_summary"),
            )
            metadata["cleanup"] = [row for row in metadata["cleanup"] if row.get("document_id") != document_id]
            metadata["cleanup"].append(cleanup)
            _write_ingestion_state(output_dir, state)
            atomic_json(metadata_path, metadata)

    succeeded = all(
        item.get("request_state") == "deleted"
        for item in state["documents"]
        if item.get("document_id") in targets
    )
    state["lifecycle"] = "cleaned" if succeeded else "cleanup_pending"
    _write_ingestion_state(output_dir, state)
    metadata["cleanup_requested_explicitly"] = True
    metadata["cleanup_completed"] = succeeded
    atomic_json(metadata_path, metadata)
    return 0 if succeeded else 1


async def _run_real(
    run_id: str,
    fixture: EvaluationFixture | None = None,
    *,
    selection_fingerprint: str | None = None,
    fixture_fingerprint: str | None = None,
) -> int:
    output_dir = PROJECT_ROOT / ".runtime" / "evaluation" / run_id
    if output_dir.exists() and any(output_dir.iterdir()):
        # A run directory is evidence.  A caller must use explicit recovery
        # rather than overwrite it with a second upload attempt.
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "mode": "authorized_real_s4",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "uploaded_documents": [],
        "operations": [],
        "scope_verified": False,
        "cleanup": [],
        "selection_fingerprint": selection_fingerprint,
        # Provider identity is useful for comparing runs, while endpoint and
        # credentials remain intentionally absent from persisted evidence.
        "chat_provider": settings.chat_config.provider,
        "chat_model": settings.chat_config.model,
        "embedding_provider": settings.embedding_config.provider,
        "embedding_model": settings.embedding_config.model,
    }
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
            "schema_version": 2,
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "lifecycle": "ingesting",
            "fixture": {
                "document_count": len(documents),
                "case_count": len(cases or EVALUATION_CASES),
                # Bind recovery to the immutable, fully validated fixture;
                # selection is separately bound below and must never widen it.
                "fingerprint": fixture_fingerprint or _fixture_fingerprint(documents, cases),
                "document_ids": [item.document_id for item in documents],
                "question_ids": [str(item.question_id) for item in (cases or EVALUATION_CASES)],
                "dual_hashes": _document_dual_hashes(fixture, documents),
                "fixture_source": (
                    str(fixture.root.relative_to(PROJECT_ROOT)).replace("\\", "/")
                    if fixture is not None and fixture.root.is_relative_to(PROJECT_ROOT) else None
                ),
            },
            "selection_fingerprint": selection_fingerprint,
            "documents": [
                _new_upload_state(run_id, document, f"s4-eval-{run_id}-{document.document_id}")
                for document in documents
            ],
            "cleanup_document_ids": [],
            "scope": None,
            "evaluation": {"status": "pending", "completed_question_ids": [], "completed_result_count": 0},
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
                headers={
                    "X-Operation-Id": state["operation_id"],
                    "X-Evaluation-Trace-Run-Id": run_id,
                },
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
            ingestion_state["lifecycle"] = "evaluating"
            success = await _run_or_resume_evaluation(
                run_id=run_id,
                output_dir=output_dir,
                state=ingestion_state,
                metadata=metadata,
                fixture=fixture,
                documents=documents,
                cases=cases,
            )
            metadata["scope_verified"] = bool((ingestion_state.get("scope") or {}).get("scope_verified"))
            metadata["allowed_document_ids_count"] = len((ingestion_state.get("scope") or {}).get("allowed_document_ids") or [])
            ingestion_state["lifecycle"] = "evaluation_completed" if success else "interrupted"
            _write_ingestion_state(output_dir, ingestion_state)
            atomic_json(output_dir / "safe-run-metadata.json", metadata)
        elif not metadata.get("error_summary"):
            metadata["error_summary"] = "ingestion_incomplete"
            ingestion_state["lifecycle"] = "interrupted"
            _write_ingestion_state(output_dir, ingestion_state)

        # Preserve exact document IDs after any interruption.  Cleanup is
        # automatic only after a complete comparison, or explicit via
        # --cleanup-run.  This makes --recover-run safe and meaningful.
        cleanup_ids = exact_cleanup_document_ids(created) if success and created else ()
        for document_id in cleanup_ids:
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

        if created and not success:
            metadata["cleanup_deferred"] = {
                "reason": "evaluation_or_ingestion_interrupted",
                "document_ids_count": len(created),
                "next_action": "use --recover-run to resume or --cleanup-run for explicit cleanup",
            }
            atomic_json(output_dir / "safe-run-metadata.json", metadata)

        if metadata["cleanup"]:
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

    cleanup_ok = success and len(metadata["cleanup"]) == len(created) and all(item.get("operation_status") == "succeeded" and item.get("chroma_vector_count") == 0 and not any(item.get("neo4j", {}).values()) and item.get("mongo_tombstone_retained") for item in metadata["cleanup"])
    metadata["completed_at_utc"] = datetime.now(UTC).isoformat()
    metadata["overall_pass"] = bool(success and cleanup_ok and not metadata.get("error_summary"))
    atomic_json(output_dir / "safe-run-metadata.json", metadata)
    return 0 if metadata["overall_pass"] else 1


async def _recover_real_ingestion(
    run_id: str,
    fixture: EvaluationFixture | None,
    *,
    selection_fingerprint: str | None = None,
    fixture_fingerprint: str | None = None,
) -> int:
    """Observe pending ingestion then resume only missing evaluation pairs.

    This mode never replays multipart POST requests.  It requires the same
    reviewed fixture and exact saved scope; if either is unavailable it stops
    without widening retrieval or deleting anything.
    """
    output_dir = PROJECT_ROOT / ".runtime" / "evaluation" / run_id
    try:
        state = _load_ingestion_state(output_dir)
    except ValueError:
        return 1
    if state.get("run_id") != run_id:
        return 1
    documents = fixture.documents if fixture else tuple(
        BenchmarkDocument(name, filename, text) for name, filename, text in SYNTHETIC_DOCUMENTS
    )
    cases = fixture.cases if fixture else None
    expected_fixture = state.get("fixture") or {}
    if expected_fixture.get("fingerprint") != (fixture_fingerprint or _fixture_fingerprint(documents, cases)):
        return 1
    if selection_fingerprint is not None and state.get("selection_fingerprint") != selection_fingerprint:
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
        if not all(item.get("request_state") == "ready" for item in state["documents"]):
            state["lifecycle"] = "interrupted"
            _write_ingestion_state(output_dir, state)
            return 1
        if not await _verify_ready_documents(client, _ready_documents_from_state(state)):
            state["lifecycle"] = "interrupted"
            _write_ingestion_state(output_dir, state)
            return 1
        metadata_path = output_dir / "safe-run-metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            metadata = {"run_id": run_id, "cleanup": []}
        completed = await _run_or_resume_evaluation(
            run_id=run_id,
            output_dir=output_dir,
            state=state,
            metadata=metadata,
            fixture=fixture,
            documents=documents,
            cases=cases,
        )
        atomic_json(metadata_path, metadata)
    # A recovered run only becomes eligible for automatic cleanup after every
    # result pair is present.  Interrupted runs remain intact for another
    # --recover-run or an explicit --cleanup-run.
    if not completed:
        state["lifecycle"] = "interrupted"
        _write_ingestion_state(output_dir, state)
        return 1
    return await _cleanup_saved_run(run_id)


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
    parser.add_argument("--recover-run", metavar="RUN_ID", help="resume only missing question pairs; never replay POST")
    parser.add_argument("--cleanup-run", metavar="RUN_ID", help="explicitly clean exact IDs saved by an interrupted run")
    parser.add_argument("--rescore", metavar="RESULTS_JSON")
    parser.add_argument("--benchmark-dir", metavar="DIR", help="read a reviewed, immutable benchmark fixture from DIR")
    parser.add_argument("--document-ids", metavar="IDS", help="comma-separated reviewed fixture document IDs")
    parser.add_argument("--question-ids", metavar="IDS", help="comma-separated reviewed fixture question IDs")
    parser.add_argument("--s4-baseline", action="store_true", help="use the historic built-in 4-document S4 fixture")
    parser.add_argument("--run-id")
    parser.add_argument("--v2-fake", action="store_true", help="run Phase F deterministic fake-only variants")
    parser.add_argument("--variants", help="comma-separated Phase F variants")
    parser.add_argument("--final-top-k", type=int, default=8, help="Phase F final context K")
    parser.add_argument("--v2-retrieval-trace", action="store_true", help="write Phase F ID-only retrieval diagnostics")
    parser.add_argument("--v2-graph-trace", action="store_true", help="write Phase F graph-only diagnostics")
    args = parser.parse_args(argv)
    if args.v2_fake:
        forbidden = args.real or args.authorized_s4 or args.recover_run or args.cleanup_run or args.rescore or args.benchmark_dir or args.document_ids or args.question_ids or args.s4_baseline
        if forbidden:
            parser.error("--v2-fake is offline-only and cannot be combined with real/S4 options")
        from services.hybrid_retrieval_v2_evaluation import VARIANTS, run_fake_evaluation
        variants = tuple(part.strip() for part in (args.variants or ",".join(VARIANTS)).split(",") if part.strip())
        run_id = args.run_id or f"phase-f-fake-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        try:
            run_fake_evaluation(PROJECT_ROOT / ".runtime" / "evaluation", run_id, variants, args.final_top_k,
                                retrieval_trace=args.v2_retrieval_trace, graph_trace=args.v2_graph_trace)
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            print(f"Phase F fake evaluation failed safely: {_safe_error(exc)}", file=sys.stderr)
            return 1
        print(f"Phase F fake-only evaluation completed: {PROJECT_ROOT / '.runtime' / 'evaluation' / run_id}")
        return 0
    if args.rescore:
        if args.offline or args.real or args.authorized_s4 or args.run_id or args.benchmark_dir or args.document_ids or args.question_ids or args.s4_baseline or args.recover_run or args.cleanup_run:
            parser.error("--rescore cannot be combined with evaluation execution options")
        return _rescore_v2(args.rescore)
    if args.recover_run and args.cleanup_run:
        parser.error("choose only one of --recover-run or --cleanup-run")
    if args.cleanup_run:
        if not args.real or not args.authorized_s4 or args.offline or args.run_id or args.benchmark_dir or args.document_ids or args.question_ids or args.s4_baseline:
            parser.error("--cleanup-run requires --real --authorized-s4 and no evaluation options")
        return asyncio.run(_cleanup_saved_run(args.cleanup_run))
    if args.recover_run:
        if not args.real or not args.authorized_s4 or args.offline or args.run_id:
            parser.error("--recover-run requires --real --authorized-s4 and cannot use --run-id")
        # Recovering a subset must not silently widen to the default fixture.
        # Read only the prior safe state; no network or database access occurs here.
        try:
            saved = _load_ingestion_state(PROJECT_ROOT / ".runtime" / "evaluation" / args.recover_run)
            saved_fixture = saved.get("fixture") if isinstance(saved.get("fixture"), dict) else {}
            if not args.benchmark_dir and saved_fixture.get("fixture_source"):
                args.benchmark_dir = str(saved_fixture["fixture_source"])
            if args.document_ids is None and isinstance(saved_fixture.get("document_ids"), list):
                args.document_ids = ",".join(str(item) for item in saved_fixture["document_ids"])
            if args.question_ids is None and isinstance(saved_fixture.get("question_ids"), list):
                args.question_ids = ",".join(str(item) for item in saved_fixture["question_ids"])
        except ValueError:
            return 1
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
    full_fixture_fingerprint: str | None = None
    if not args.s4_baseline:
        supplied = Path(args.benchmark_dir) if args.benchmark_dir else DEFAULT_ENTERPRISE_BENCHMARK_DIR
        fixture_root = supplied if supplied.is_absolute() else PROJECT_ROOT / supplied
        try:
            full_fixture = load_evaluation_fixture(fixture_root)
            full_fixture_fingerprint = _fixture_fingerprint(full_fixture.documents, full_fixture.cases)
            fixture = select_evaluation_subset(
                full_fixture,
                document_ids=_parse_selection(args.document_ids, kind="document"),
                question_ids=_parse_selection(args.question_ids, kind="question"),
            )
            # Validate raw-file and actual-upload byte identity before any
            # provider, HTTP, or storage operation can be created.
            _document_dual_hashes(fixture, fixture.documents)
        except (OSError, ValueError) as exc:
            print(f"benchmark fixture validation failed safely: {_safe_error(exc)}", file=sys.stderr)
            return 1
    modes = _parse_modes(args.mode)
    selection_fingerprint = (
        _selection_fingerprint(
            fixture_fingerprint=full_fixture_fingerprint or _fixture_fingerprint(fixture.documents, fixture.cases),
            documents=fixture.documents, cases=fixture.cases, modes=modes,
        ) if fixture else None
    )
    if args.offline:
        try:
            payload = run_offline_sync(
                run_id, modes, PROJECT_ROOT / ".runtime" / "evaluation", fixture=fixture,
            )
        except (ValueError, OSError) as exc:
            print(f"offline evaluation failed safely: {_safe_error(exc)}", file=sys.stderr)
            return 1
        print(f"offline fake-only evaluation completed: {PROJECT_ROOT / '.runtime' / 'evaluation' / run_id}")
        atomic_json(PROJECT_ROOT / ".runtime" / "evaluation" / run_id / "safe-run-metadata.json", {
            "run_id": run_id, "offline": True, "fixture_document_count": len(fixture.documents) if fixture else 4,
            "fixture_question_count": len(fixture.cases) if fixture else len(EVALUATION_CASES),
            "selection_fingerprint": selection_fingerprint,
            "selected_document_ids": [item.document_id for item in fixture.documents] if fixture else [],
            "selected_question_ids": [item.question_id for item in fixture.cases] if fixture else [],
            "dual_hashes": _document_dual_hashes(fixture, fixture.documents) if fixture else [],
        })
        for mode, summary in payload["summary"].items():
            print(f"{mode}: questions={summary['questions']} accuracy={summary['question_accuracy']:.3f}")
        return 0
    if args.recover_run:
        return asyncio.run(_recover_real_ingestion(
            args.recover_run,
            fixture,
            selection_fingerprint=selection_fingerprint,
            fixture_fingerprint=full_fixture_fingerprint,
        ))
    return asyncio.run(_run_real(
        run_id,
        fixture,
        selection_fingerprint=selection_fingerprint,
        fixture_fingerprint=full_fixture_fingerprint,
    ))


if __name__ == "__main__":
    raise SystemExit(main())
