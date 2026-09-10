"""Run one persisted, document-scoped S3 incremental-update verification.

The tool deliberately has no retry loop for POST/PUT/DELETE.  Every network
step is written atomically before the next one starts so a disconnected shell
cannot turn an already-delivered write into a duplicate request.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

V1 = "王明是星河公司的后端工程师。王明负责知识检索项目。该项目使用 Chroma。"
V2 = "王明是星河公司的技术负责人。王明负责智能知识平台。该平台使用 Chroma 和 Neo4j。"
QUESTION = "王明现在担任什么角色？他负责哪个项目？项目使用哪些存储技术？"
PREFIX = "codex-incremental-test-"


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def safe_error(error: object) -> str:
    return " ".join(str(error).split())[:240] or type(error).__name__


def safe_source(value: object) -> str:
    return str(value or "").replace("\\", "/").rsplit("/", 1)[-1]


def safe_response(response: dict[str, Any]) -> dict[str, Any]:
    """Persist only public API fields; never save provider/raw HTTP material."""
    allowed = {
        "operation_id", "document_id", "version", "status", "changed", "content_hash",
        "namespace", "logical_key", "file_name", "chunks_count", "entities_count",
        "relations_count", "status_url", "completed_steps", "error_summary",
    }
    return {key: response.get(key) for key in sorted(allowed) if key in response}


def multipart_body(filename: str, content: bytes, fields: dict[str, str]) -> tuple[bytes, str]:
    boundary = "----AgentKnowledgeHubIncrementalVerification"
    parts: list[bytes] = []
    for key, value in fields.items():
        parts.extend((
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode(),
            value.encode("utf-8"), b"\r\n",
        ))
    parts.extend((
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode(),
        b"Content-Type: text/plain; charset=utf-8\r\n\r\n",
        content, b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ))
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def http_call(
    base_url: str, method: str, path: str, *, body: bytes | None = None, content_type: str | None = None
) -> dict[str, Any]:
    started = time.monotonic()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}", data=body, method=method,
        headers={"Content-Type": content_type} if content_type else {},
    )
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=300) as reply:
            raw = reply.read().decode("utf-8")
            parsed = json.loads(raw)
            return {"http_status": reply.status, "elapsed_seconds": round(time.monotonic() - started, 3), "json": parsed}
    except urllib.error.HTTPError as error:
        return {"http_status": error.code, "elapsed_seconds": round(time.monotonic() - started, 3), "error_summary": "HTTP request failed"}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        return {"http_status": None, "elapsed_seconds": round(time.monotonic() - started, 3), "error_summary": type(error).__name__}


def record_api_step(output_dir: Path, step: str, call: dict[str, Any]) -> dict[str, Any]:
    response = call.pop("json", None)
    record = dict(call)
    if isinstance(response, dict):
        record["response"] = safe_response(response)
    write_json_atomic(output_dir / f"{step}.json", record)
    return response if isinstance(response, dict) else {}


def step_summary(call: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    return {
        "http_status": call.get("http_status"),
        "elapsed_seconds": call.get("elapsed_seconds"),
        **safe_response(response),
    }


def mongo_snapshot(exclude_document_id: str | None = None) -> dict[str, Any]:
    from config import settings
    from pymongo import MongoClient

    client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=10_000)
    try:
        database = client[settings.mongodb_database]
        query = {"document_id": {"$ne": exclude_document_id}} if exclude_document_id else {}
        document_ids = sorted(
            str(row["document_id"]) for row in database.documents.find(query, {"document_id": 1, "_id": 0})
            if row.get("document_id")
        )
        return {
            "documents": database.documents.count_documents(query),
            "document_versions": database.document_versions.count_documents(query),
            "document_operations": database.document_operations.count_documents({}),
            "document_id_digest": hashlib.sha256("\n".join(document_ids).encode()).hexdigest(),
        }
    finally:
        client.close()


def chroma_snapshot(document_id: str | None = None) -> dict[str, Any]:
    import chromadb
    from config import settings
    from services.vector_store import VectorStoreService

    host = VectorStoreService.chroma_http_host(settings.chroma_host)
    collection = chromadb.HttpClient(host=host, port=settings.chroma_port).get_collection("knowledge_chunks")
    if document_id:
        rows = collection.get(where={"document_id": document_id}, include=["metadatas"])
        metadata = rows.get("metadatas", []) or []
        return {
            "vectors": len(rows.get("ids", []) or []),
            "versions": sorted({int(row.get("document_version", 0)) for row in metadata if row.get("document_version") is not None}),
            "current_versions": sorted({int(row.get("document_version", 0)) for row in metadata if row.get("is_current") and row.get("status") == "ready"}),
            "ready_current_vectors": sum(bool(row.get("is_current")) and row.get("status") == "ready" for row in metadata),
            "inactive_vectors": sum(not bool(row.get("is_current")) for row in metadata),
        }
    return {"vectors": collection.count()}


async def neo4j_snapshot(document_id: str | None = None) -> dict[str, Any]:
    from config import settings
    from neo4j import AsyncGraphDatabase

    driver = AsyncGraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
    try:
        async with driver.session() as session:
            if document_id:
                queries = {
                    "document_versions": "MATCH (dv:DocumentVersion {document_id: $id}) RETURN count(dv) AS value",
                    "mentions": "MATCH (:DocumentVersion {document_id: $id})-[r:MENTIONS]->() RETURN count(r) AS value",
                    "evidence": "MATCH ()-[r]->() WHERE r.document_id = $id RETURN count(r) AS value",
                    "current_evidence": "MATCH ()-[r]->() WHERE r.document_id = $id AND r.is_current = true AND r.status = 'ready' RETURN count(r) AS value",
                    "current_versions": "MATCH ()-[r]->() WHERE r.document_id = $id AND r.is_current = true AND r.status = 'ready' RETURN collect(DISTINCT r.document_version) AS value",
                }
                params = {"id": document_id}
            else:
                queries = {
                    "document_versions": "MATCH (dv:DocumentVersion) RETURN count(dv) AS value",
                    "mentions": "MATCH (:DocumentVersion)-[r:MENTIONS]->() RETURN count(r) AS value",
                    "evidence": "MATCH ()-[r]->() WHERE r.document_id IS NOT NULL RETURN count(r) AS value",
                    "entities": "MATCH (e:Entity) RETURN count(e) AS value",
                }
                params = {}
            snapshot: dict[str, Any] = {}
            for name, query in queries.items():
                result = await session.run(query, params)
                record = await result.single()
                value = record["value"] if record else 0
                snapshot[name] = sorted(int(item) for item in value) if name == "current_versions" else int(value)
            return snapshot
    finally:
        await driver.close()


def api_document(base_url: str, document_id: str) -> dict[str, Any]:
    return http_call(base_url, "GET", f"/api/documents/{document_id}").get("json", {})


def api_operation(base_url: str, operation_id: str | None) -> dict[str, Any]:
    if not operation_id:
        return {}
    return http_call(base_url, "GET", f"/api/document-operations/{operation_id}").get("json", {})


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def state_snapshot(document_id: str | None = None, *, exclude_document_id: str | None = None) -> dict[str, Any]:
    return {
        "mongo": mongo_snapshot(exclude_document_id),
        "chroma": chroma_snapshot(document_id),
        "neo4j": asyncio.run(neo4j_snapshot(document_id)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--output-root", default=".runtime/incremental-verification")
    args = parser.parse_args()

    logical_key = f"{PREFIX}{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output_dir = Path(args.output_root) / logical_key
    summary: dict[str, Any] = {"logical_key": logical_key, "namespace": "default", "overall_pass": False, "steps": {}}
    document_id: str | None = None
    deletion_attempted = False

    write_json_atomic(output_dir / "started.json", {"logical_key": logical_key, "namespace": "default"})
    try:
        baseline = state_snapshot()
        write_json_atomic(output_dir / "baseline.json", baseline)

        body, content_type = multipart_body("codex-incremental-v1.txt", V1.encode(), {"logical_key": logical_key, "namespace": "default"})
        first_call = http_call(args.base_url, "POST", "/api/documents", body=body, content_type=content_type)
        first = record_api_step(output_dir, "v1-create", first_call)
        document_id = first.get("document_id")
        summary["steps"]["v1"] = step_summary(first_call, first)
        assert_true(first.get("status") == "succeeded" and first.get("changed") is True and first.get("version") == 1, "V1 create did not succeed as version 1")
        assert_true(bool(document_id), "V1 create did not return document_id")
        operation = api_operation(args.base_url, first.get("operation_id"))
        assert_true(operation.get("status") == "succeeded", "V1 operation is not succeeded")
        document = api_document(args.base_url, document_id)
        assert_true(document.get("logical_key") == logical_key and document.get("current_version") == 1 and document.get("status") == "ready", "V1 registry state is invalid")
        after_v1 = state_snapshot(document_id)
        write_json_atomic(output_dir / "after-v1.json", after_v1)
        assert_true(after_v1["chroma"]["ready_current_vectors"] > 0, "V1 has no ready/current vectors")
        assert_true(after_v1["neo4j"]["document_versions"] == 1 and after_v1["neo4j"]["mentions"] > 0 and after_v1["neo4j"]["evidence"] > 0, "V1 graph provenance is incomplete")

        duplicate_call = http_call(args.base_url, "POST", "/api/documents", body=body, content_type=content_type)
        duplicate = record_api_step(output_dir, "v1-duplicate", duplicate_call)
        summary["steps"]["duplicate"] = step_summary(duplicate_call, duplicate)
        assert_true(duplicate.get("changed") is False and duplicate.get("document_id") == document_id and duplicate.get("version") == 1, "duplicate V1 was not idempotent")
        after_duplicate = state_snapshot(document_id)
        write_json_atomic(output_dir / "after-v1-duplicate.json", after_duplicate)
        assert_true(after_duplicate["chroma"] == after_v1["chroma"] and after_duplicate["neo4j"] == after_v1["neo4j"], "duplicate V1 changed document-scoped stored data")
        duplicate_operation = api_operation(args.base_url, duplicate.get("operation_id"))
        assert_true(duplicate_operation.get("completed_steps") == ["mongo_reserved", "unchanged"], "duplicate V1 did not finish at the Registry hash check")

        body, content_type = multipart_body("codex-incremental-v2.txt", V2.encode(), {})
        update_call = http_call(args.base_url, "PUT", f"/api/documents/{document_id}", body=body, content_type=content_type)
        update = record_api_step(output_dir, "v2-update", update_call)
        summary["steps"]["v2"] = step_summary(update_call, update)
        assert_true(update.get("status") == "succeeded" and update.get("changed") is True and update.get("version") == 2, "V2 update did not succeed as version 2")
        assert_true(api_operation(args.base_url, update.get("operation_id")).get("status") == "succeeded", "V2 operation is not succeeded")
        document = api_document(args.base_url, document_id)
        assert_true(document.get("current_version") == 2 and document.get("status") == "ready", "V2 did not become current")
        versions = http_call(args.base_url, "GET", f"/api/documents/{document_id}/versions").get("json", [])
        assert_true(len(versions) == 2 and versions[0].get("is_current") is False and versions[1].get("is_current") is True, "V1/V2 registry current flags are invalid")
        after_v2 = state_snapshot(document_id)
        write_json_atomic(output_dir / "after-v2.json", after_v2)
        assert_true(after_v2["chroma"]["versions"] == [1, 2] and after_v2["chroma"]["current_versions"] == [2], "V2 Chroma current state is invalid")
        assert_true(after_v2["neo4j"]["document_versions"] == 2 and after_v2["neo4j"]["current_evidence"] > 0 and after_v2["neo4j"]["current_versions"] == [2], "V2 graph current state is invalid")

        qa_payload = json.dumps({"question": QUESTION, "session_id": f"{logical_key}-qa", "user_id": "incremental-verification"}, ensure_ascii=False).encode("utf-8")
        qa_call = http_call(args.base_url, "POST", "/api/qa/ask", body=qa_payload, content_type="application/json")
        qa = qa_call.get("json", {})
        qa_record = {"http_status": qa_call.get("http_status"), "elapsed_seconds": qa_call.get("elapsed_seconds"), "response": qa}
        write_json_atomic(output_dir / "qa.json", qa_record)
        answer = str(qa.get("answer", ""))
        sources = qa.get("sources", []) if isinstance(qa.get("sources"), list) else []
        summary["qa"] = {"http_status": qa_call.get("http_status"), "elapsed_seconds": qa_call.get("elapsed_seconds"), "answer_summary": answer[:500], "sources": [{"document_id": item.get("document_id"), "document_version": item.get("document_version"), "source": safe_source(item.get("source", ""))} for item in sources if isinstance(item, dict)]}
        assert_true(qa_call.get("http_status") == 200, "QA did not return HTTP 200")
        assert_true(all(term in answer for term in ("技术负责人", "智能知识平台", "Chroma", "Neo4j")), "QA answer does not contain V2 facts")
        assert_true("后端工程师" not in answer and "知识检索项目" not in answer, "QA answer contains inactive V1 facts")
        assert_true(any(item.get("document_id") == document_id and item.get("document_version") == 2 for item in sources if isinstance(item, dict)), "QA sources do not include V2 provenance")

        current = api_document(args.base_url, document_id)
        assert_true(current.get("logical_key") == logical_key and logical_key.startswith(PREFIX), "refusing delete: test identity verification failed")
        delete_call = http_call(args.base_url, "DELETE", f"/api/documents/{document_id}")
        deleted = record_api_step(output_dir, "delete", delete_call)
        deletion_attempted = True
        summary["steps"]["delete"] = step_summary(delete_call, deleted)
        assert_true(deleted.get("status") == "succeeded", "delete did not complete")
        assert_true(api_operation(args.base_url, deleted.get("operation_id")).get("status") == "succeeded", "delete operation is not succeeded")
        document = api_document(args.base_url, document_id)
        assert_true(document.get("status") == "deleted", "document is not logically deleted")
        after_delete = state_snapshot(document_id)
        write_json_atomic(output_dir / "after-delete.json", after_delete)
        assert_true(after_delete["chroma"]["vectors"] == 0 and after_delete["neo4j"]["document_versions"] == 0 and after_delete["neo4j"]["mentions"] == 0 and after_delete["neo4j"]["evidence"] == 0, "delete did not precisely clean Chroma/Neo4j provenance")
        repeated_delete_call = http_call(args.base_url, "DELETE", f"/api/documents/{document_id}")
        repeated_delete = record_api_step(output_dir, "delete-duplicate", repeated_delete_call)
        summary["steps"]["delete_duplicate"] = step_summary(repeated_delete_call, repeated_delete)
        assert_true(repeated_delete.get("status") == "deleted" and repeated_delete.get("changed") is False, "duplicate delete is not idempotent")

        final = state_snapshot(exclude_document_id=document_id)
        write_json_atomic(output_dir / "final.json", final)
        summary["baseline"] = baseline
        summary["final"] = final
        summary["document_id"] = document_id
        summary["other_documents_unchanged"] = {
            "document_count": final["mongo"]["documents"] == baseline["mongo"]["documents"],
            "version_count": final["mongo"]["document_versions"] == baseline["mongo"]["document_versions"],
            "document_id_digest": final["mongo"]["document_id_digest"] == baseline["mongo"]["document_id_digest"],
            "chroma_total": final["chroma"] == baseline["chroma"],
            "neo4j_total": final["neo4j"] == baseline["neo4j"],
        }
        assert_true(all(summary["other_documents_unchanged"].values()), "cleanup changed data outside the test document")
        summary["overall_pass"] = True
    except Exception as error:
        summary["error_summary"] = safe_error(error)
        if document_id and not deletion_attempted:
            cleanup = record_api_step(output_dir, "cleanup-after-failure", http_call(args.base_url, "DELETE", f"/api/documents/{document_id}"))
            summary["cleanup_after_failure"] = safe_response(cleanup)
    finally:
        summary["document_id"] = document_id
        write_json_atomic(output_dir / "summary.json", summary)
    return 0 if summary.get("overall_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
