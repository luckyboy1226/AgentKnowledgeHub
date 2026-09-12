"""One-shot, authorized D03 ingestion timeout/recovery verification.

This tool intentionally has no fixture discovery and no QA path: it accepts
only the reviewed D03 fixture, sends one POST, then observes the caller-owned
operation ID.  Runtime evidence contains identifiers and counts only.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = PROJECT_ROOT / "python"
# Keep provider configuration resolution identical to the API and the main
# evaluation runner without reading or displaying the local .env file.
os.chdir(PYTHON_ROOT)
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from config import settings  # noqa: E402
from providers.factory import create_embedding_provider  # noqa: E402
from services.knowledge_graph import KnowledgeGraphService  # noqa: E402
from services.rag_evaluation import atomic_json  # noqa: E402
from services.vector_store import VectorStoreService  # noqa: E402


API_BASE_URL = "http://127.0.0.1:8080"
FIXTURE_RELATIVE_PATH = Path("benchmarks/enterprise_20docs_60q_expanded/documents/D03_application_architecture_team.txt")
EXPECTED_SHA256 = "0BF9FAB8202E29EBE58854E65A60EF7F980905BE7C5F564B903E2920F0A6E6EC"
READ_TIMEOUT_SECONDS = 180
OBSERVATION_DEADLINE_SECONDS = 15 * 60
TERMINAL_OPERATION_STATUSES = {"succeeded", "failed", "cleanup_pending", "needs_reconciliation"}
SOURCE_HASH_PATHS = (
    Path("python/api/main.py"),
    Path("python/services/document_update_coordinator.py"),
    Path("scripts/run-rag-eval.py"),
    Path("scripts/verify-benchmark-recovery-d03.py"),
)


def _safe_error(error: object) -> str:
    return type(error).__name__[:80]


def _run_id() -> str:
    return f"benchmark-recovery-d03-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"


def _source_hashes() -> dict[str, str]:
    return {
        path.as_posix(): hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest()
        for path in SOURCE_HASH_PATHS
    }


def _ensure_local_no_proxy() -> None:
    values = [value.strip() for value in os.environ.get("NO_PROXY", "").split(",") if value.strip()]
    for host in ("localhost", "127.0.0.1", "::1"):
        if host not in values:
            values.append(host)
    os.environ["NO_PROXY"] = ",".join(values)
    os.environ["no_proxy"] = os.environ["NO_PROXY"]


async def _api_json(client: httpx.AsyncClient, method: str, path: str, **kwargs: Any) -> tuple[int, dict[str, Any], float]:
    started = time.monotonic()
    try:
        response = await client.request(method, f"{API_BASE_URL}{path}", **kwargs)
    except httpx.HTTPError as exc:
        return 0, {"error_summary": _safe_error(exc)}, round((time.monotonic() - started) * 1000, 3)
    elapsed = round((time.monotonic() - started) * 1000, 3)
    try:
        body = response.json()
    except ValueError:
        body = {"error_summary": "non_json_response"}
    return response.status_code, body if isinstance(body, dict) else {"items": body}, elapsed


async def _graph_counts(graph: KnowledgeGraphService, document_id: str) -> dict[str, int]:
    async def count(cypher: str) -> int:
        rows = await graph.execute_cypher(cypher, {"document_id": document_id})
        return int(rows[0].get("count", 0)) if rows else 0

    return {
        "document_versions": await count("MATCH (dv:DocumentVersion {document_id: $document_id}) RETURN count(dv) AS count"),
        "mentions": await count("MATCH (:DocumentVersion {document_id: $document_id})-[m:MENTIONS]->() RETURN count(m) AS count"),
        "evidence": await count("MATCH ()-[r]->() WHERE r.document_id = $document_id RETURN count(r) AS count"),
    }


async def _safe_baseline() -> tuple[dict[str, Any], VectorStoreService, KnowledgeGraphService, Any]:
    """Read counts only; no provider invocation and no business writes."""
    from pymongo import MongoClient

    mongo_client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=10_000)
    database = mongo_client[settings.mongodb_database]
    vectors = VectorStoreService(create_embedding_provider(settings))
    graph = KnowledgeGraphService()
    await vectors.init()
    await graph.init()
    total_vectors = int((await vectors.get_stats()).get("total_vectors", 0))
    graph_rows = await graph.execute_cypher("MATCH (e:Entity) RETURN count(e) AS count")
    relation_rows = await graph.execute_cypher("MATCH ()-[r]->() RETURN count(r) AS count")
    baseline = {
        "mongo": {
            "documents": database["documents"].count_documents({}),
            "document_versions": database["document_versions"].count_documents({}),
            "document_operations": database["document_operations"].count_documents({}),
        },
        "chroma_total_vectors": total_vectors,
        "neo4j": {
            "entities": int(graph_rows[0].get("count", 0)) if graph_rows else 0,
            "relationships": int(relation_rows[0].get("count", 0)) if relation_rows else 0,
        },
    }
    return baseline, vectors, graph, mongo_client


def _storage_baseline_restored(before: dict[str, Any], after: dict[str, Any]) -> bool:
    """Mongo tombstones are intentionally durable; vector/graph counts are not."""
    return (
        before.get("chroma_total_vectors") == after.get("chroma_total_vectors")
        and before.get("neo4j") == after.get("neo4j")
    )


async def _observe_operation(
    client: httpx.AsyncClient,
    operation_id: str,
    output_dir: Path,
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    deadline = time.monotonic() + OBSERVATION_DEADLINE_SECONDS
    while time.monotonic() < deadline:
        status, body, elapsed = await _api_json(client, "GET", f"/api/document-operations/{operation_id}")
        observation = {
            "at_utc": datetime.now(UTC).isoformat(),
            "http_status": status,
            "elapsed_ms": elapsed,
            "operation_id": operation_id,
            "status": body.get("status"),
            "document_id": body.get("document_id"),
            "version": body.get("version"),
            "document_status": body.get("document_status"),
            "completed_steps": body.get("completed_steps", []),
            "error_phase": body.get("error_phase"),
            "error_category": body.get("error_category"),
            "error_type": body.get("error_type"),
            "chunk_index": body.get("chunk_index"),
            "error_summary": body.get("error_summary"),
        }
        observations.append(observation)
        atomic_json(output_dir / "operation-observations.json", {"observations": observations})
        # Journal creation trails multipart receipt.  The first 404 is a
        # grace-period observation, never a basis for a second POST.
        if status == 200 and body.get("status") in TERMINAL_OPERATION_STATUSES:
            return observation
        if status not in {200, 404}:
            return observation
        await asyncio.sleep(2)
    return {
        "operation_id": operation_id,
        "status": "ambiguous",
        "error_summary": "operation_observation_deadline_exceeded",
    }


def _run_recovery_observer(run_id: str) -> dict[str, Any]:
    """Exercise the B01 recovery CLI; it sends GET only and never POSTs."""
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run-rag-eval.py"),
            "--real",
            "--authorized-s4",
            "--recover-run",
            run_id,
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=OBSERVATION_DEADLINE_SECONDS + 30,
        check=False,
    )
    # stdout/stderr can contain environment-specific paths, so retain only
    # process outcome rather than persisting raw command output.
    return {"executed": True, "exit_code": completed.returncode}


async def run() -> int:
    _ensure_local_no_proxy()
    fixture = PROJECT_ROOT / FIXTURE_RELATIVE_PATH
    content = fixture.read_bytes()
    content_hash = hashlib.sha256(content).hexdigest().upper()
    if content_hash != EXPECTED_SHA256:
        print("fixture SHA-256 mismatch; refusing to send document", file=sys.stderr)
        return 1

    run_id = _run_id()
    output_dir = PROJECT_ROOT / ".runtime" / "evaluation" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    operation_id = str(uuid.uuid4())
    logical_key = run_id
    state = {
        "run_id": run_id,
        "documents": [{
            "run_id": run_id,
            "fixture_document_id": "D03",
            "logical_key": logical_key,
            "safe_filename": fixture.name,
            "content_hash": content_hash.lower(),
            "operation_id": operation_id,
            "request_state": "planned",
            "request_started_at": None,
            "last_observed_status": None,
            "document_id": None,
            "version": None,
            "terminal_status": None,
            "cleanup_operation_id": None,
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "error_summary": None,
        }],
        "cleanup_document_ids": [],
    }
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "fixture_document_id": "D03",
        "logical_key": logical_key,
        "operation_id": operation_id,
        "content_hash": content_hash,
        "post_calls": 0,
        "qa_calls": 0,
        "other_benchmark_documents": 0,
        "runner_level_provider_retries": 0,
        "provider_sdk_retry_observability": "not_exposed_by_api",
        "source_hashes_before": _source_hashes(),
    }
    atomic_json(output_dir / "ingestion-state.json", state)
    atomic_json(output_dir / "safe-run-metadata.json", metadata)

    baseline, vectors, graph, mongo_client = await _safe_baseline()
    atomic_json(output_dir / "baseline.json", baseline)
    observations: list[dict[str, Any]] = []
    cleanup: dict[str, Any] = {}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(READ_TIMEOUT_SECONDS, connect=10)) as client:
            state["documents"][0].update({
                "request_state": "request_sent",
                "request_started_at": datetime.now(UTC).isoformat(),
                "updated_at": datetime.now(UTC).isoformat(),
            })
            atomic_json(output_dir / "ingestion-state.json", state)
            metadata["post_calls"] = 1
            atomic_json(output_dir / "safe-run-metadata.json", metadata)
            status, body, elapsed = await _api_json(
                client,
                "POST",
                "/api/documents",
                headers={"X-Operation-Id": operation_id},
                data={"namespace": "default", "logical_key": logical_key},
                files={"file": (fixture.name, content, "text/plain")},
            )
            timed_out = status == 0 and body.get("error_summary") == "ReadTimeout"
            upload = {
                "http_status": status,
                "elapsed_ms": elapsed,
                "operation_id": operation_id,
                "document_id": body.get("document_id"),
                "version": body.get("version"),
                "status": body.get("status"),
                "content_hash": content_hash,
                "response_timeout": timed_out,
                "error_summary": body.get("detail") if status >= 400 else body.get("error_summary"),
            }
            atomic_json(output_dir / "upload-result.json", upload)
            state_item = state["documents"][0]
            state_item.update({
                "request_state": "ambiguous" if timed_out else "response_received",
                "last_observed_status": body.get("status"),
                "document_id": body.get("document_id"),
                "version": body.get("version"),
                "error_summary": upload["error_summary"],
                "updated_at": datetime.now(UTC).isoformat(),
            })
            atomic_json(output_dir / "ingestion-state.json", state)

            # Confirm that persisted operation state can be reopened by the
            # production runner without replaying the original multipart POST.
            metadata["recover_run"] = _run_recovery_observer(run_id)
            atomic_json(output_dir / "safe-run-metadata.json", metadata)
            operation = await _observe_operation(client, operation_id, output_dir, observations)
            document_id = operation.get("document_id") or state_item.get("document_id")
            state_item.update({
                "last_observed_status": operation.get("status"),
                "document_id": document_id,
                "version": operation.get("version") or state_item.get("version"),
                "terminal_status": operation.get("status") if operation.get("status") in TERMINAL_OPERATION_STATUSES else None,
                "request_state": "ready" if operation.get("status") == "succeeded" and document_id else ("failed" if operation.get("status") in {"failed", "needs_reconciliation"} else "ambiguous"),
                "error_summary": operation.get("error_summary"),
                "updated_at": datetime.now(UTC).isoformat(),
            })
            if state_item["request_state"] == "ready":
                state["cleanup_document_ids"] = [str(document_id)]
            atomic_json(output_dir / "ingestion-state.json", state)

            if state_item["request_state"] == "ready" and document_id:
                document_status, document, _ = await _api_json(client, "GET", f"/api/documents/{document_id}")
                versions_status, versions, _ = await _api_json(client, "GET", f"/api/documents/{document_id}/versions")
                version_row = next((item for item in versions.get("items", versions if isinstance(versions, list) else []) if item.get("version") == 1), {})
                metadata["ingestion"] = {
                    "document_status_http": document_status,
                    "document_status": document.get("status"),
                    "chunks": version_row.get("chunk_count"),
                    "entities": body.get("entities_count"),
                    "relations": body.get("relations_count"),
                    "version_content_hash_matches": versions_status == 200 and version_row.get("content_hash") == content_hash.lower(),
                    "chat_logical_calls_estimate": version_row.get("chunk_count"),
                    "embedding_logical_calls_estimate": 1,
                }
                operation_row = mongo_client[settings.mongodb_database]["document_operations"].find_one(
                    {"operation_id": operation_id},
                    {"_id": 0, "processing_metadata": 1},
                ) or {}
                operation_metadata = operation_row.get("processing_metadata") or {}
                metadata["ingestion"].update({
                    "chunks": operation_metadata.get("chunk_count", metadata["ingestion"]["chunks"]),
                    "entities": operation_metadata.get("entity_count", metadata["ingestion"]["entities"]),
                    "relations": operation_metadata.get("relation_count", metadata["ingestion"]["relations"]),
                    "dropped_relations": operation_metadata.get("dropped_relation_count"),
                })

                if (
                    document.get("logical_key") == logical_key
                    and document.get("status") == "ready"
                    and version_row.get("content_hash") == content_hash.lower()
                ):
                    delete_status, delete_body, delete_elapsed = await _api_json(client, "DELETE", f"/api/documents/{document_id}")
                    cleanup = {
                        "document_id": document_id,
                        "http_status": delete_status,
                        "elapsed_ms": delete_elapsed,
                        "operation_id": delete_body.get("operation_id"),
                        "status": delete_body.get("status"),
                    }
                    state_item["cleanup_operation_id"] = cleanup["operation_id"]
                    state_item["request_state"] = "cleanup_pending"
                    atomic_json(output_dir / "ingestion-state.json", state)
                    if cleanup["operation_id"]:
                        cleanup_operation = await _observe_operation(client, cleanup["operation_id"], output_dir, observations)
                        cleanup["operation_status"] = cleanup_operation.get("status")
                    else:
                        cleanup["error_summary"] = "delete_operation_id_missing"
                    if cleanup.get("operation_status") == "succeeded":
                        state_item["request_state"] = "deleted"
                        state_item["terminal_status"] = "succeeded"
                    else:
                        state_item["request_state"] = "cleanup_failed"
                        state_item["terminal_status"] = cleanup.get("operation_status")
                    atomic_json(output_dir / "ingestion-state.json", state)
                else:
                    cleanup = {"document_id": document_id, "error_summary": "cleanup_identity_verification_failed"}
            else:
                cleanup = {"document_id": document_id, "error_summary": "no_verified_document_for_cleanup"}

            atomic_json(output_dir / "cleanup-report.json", cleanup)
            if document_id:
                cleanup["chroma_vector_count"] = len(vectors.list_document_vector_ids(document_id, None))
                cleanup["neo4j"] = await _graph_counts(graph, document_id)
                mongo_status, mongo_document, _ = await _api_json(client, "GET", f"/api/documents/{document_id}")
                cleanup["mongo_status"] = mongo_document.get("status") if mongo_status == 200 else None
                cleanup["mongo_tombstone_retained"] = cleanup["mongo_status"] == "deleted"
            final, final_vectors, final_graph, final_mongo_client = await _safe_baseline()
            await final_graph.close()
            final_mongo_client.close()
            metadata["final"] = final
            metadata["source_hashes_after"] = _source_hashes()
            metadata["source_hashes_unchanged"] = metadata["source_hashes_before"] == metadata["source_hashes_after"]
            metadata["late_success_recovered"] = bool(timed_out and state_item.get("request_state") in {"ready", "deleted"})
            metadata["baseline_restored"] = _storage_baseline_restored(baseline, final)
            metadata["post_calls_exactly_one"] = metadata["post_calls"] == 1
            metadata["completed_at_utc"] = datetime.now(UTC).isoformat()
            atomic_json(output_dir / "cleanup-report.json", cleanup)
            atomic_json(output_dir / "safe-run-metadata.json", metadata)
            summary = {
                "run_id": run_id,
                "logical_key": logical_key,
                "operation_id": operation_id,
                "document_id": document_id,
                "post_calls": metadata["post_calls"],
                "response_timeout": timed_out,
                "late_success_recovered": metadata["late_success_recovered"],
                "recover_run": metadata["recover_run"],
                "operation_final_status": operation.get("status"),
                "cleanup_status": cleanup.get("operation_status"),
                "baseline_restored": metadata["baseline_restored"],
                "qa_calls": 0,
                "other_benchmark_documents": 0,
            }
            atomic_json(output_dir / "summary.json", summary)
            return 0 if (
                summary["post_calls"] == 1
                and operation.get("status") == "succeeded"
                and cleanup.get("operation_status") == "succeeded"
                and cleanup.get("chroma_vector_count") == 0
                and not any(cleanup.get("neo4j", {}).values())
                and cleanup.get("mongo_tombstone_retained")
                and summary["baseline_restored"]
            ) else 1
    finally:
        await graph.close()
        mongo_client.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
