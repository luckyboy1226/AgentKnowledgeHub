"""Safe, resumable embedding-index migration utility.

``--dry-run`` is intentionally read-only.  ``--execute`` is deliberately
gated behind an explicit authorization flag: it supports the future approved
migration without making a configuration probe capable of writing vectors.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))
# Settings intentionally resolves ``.env`` relative to the backend directory,
# matching the API and existing evaluation scripts.  This is configuration
# discovery only; the script never reads or prints credential values.
os.chdir(PROJECT_ROOT / "python")

from config.settings import settings  # noqa: E402
from providers.factory import create_embedding_provider  # noqa: E402
from services.vector_store import VectorStoreService  # noqa: E402


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def collection_name(value: str) -> str:
    if not VectorStoreService._COLLECTION_NAME.fullmatch(value):
        raise argparse.ArgumentTypeError("invalid collection name")
    return value


def safe_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    allowed = {"hnsw:space", "embedding_provider", "embedding_model", "embedding_dimensions", "embedding_space_id", "schema_version"}
    return {key: metadata[key] for key in allowed if key in (metadata or {})}


def dry_run(source: str, target: str, run_id: str, batch_size: int) -> dict[str, Any]:
    import chromadb

    client = chromadb.HttpClient(host=VectorStoreService.chroma_http_host(settings.chroma_host), port=settings.chroma_port)
    source_collection = client.get_collection(source)
    source_count = source_collection.count()
    rows = source_collection.get(include=["embeddings", "documents", "metadatas"])
    ids = rows.get("ids", [])
    vectors = rows.get("embeddings", [])
    documents = rows.get("documents", [])
    metadatas = rows.get("metadatas", [])
    dimensions = sorted({len(vector) for vector in vectors if hasattr(vector, "__len__")})
    versioned = sum(1 for metadata in metadatas if isinstance(metadata, dict) and metadata.get("document_id"))
    current_ready = sum(1 for metadata in metadatas if isinstance(metadata, dict) and metadata.get("is_current") is True and metadata.get("status") == "ready")
    target_exists = True
    try:
        target_collection = client.get_collection(target)
        target_metadata = safe_metadata(target_collection.metadata)
    except Exception:
        target_exists, target_metadata = False, {}
    target_identity_matches = target_exists and target_metadata == {
        "hnsw:space": "cosine",
        "embedding_provider": settings.embedding_config.provider,
        "embedding_model": settings.embedding_config.model,
        "embedding_dimensions": settings.embedding_dimensions,
        "embedding_space_id": settings.resolved_embedding_space_id,
        "schema_version": 1,
    }
    valid_texts = [text for text in documents if isinstance(text, str) and text.strip()]
    target_profile_matches = (
        settings.embedding_config.provider == "qwen"
        and settings.embedding_config.model == "qwen3.7-text-embedding-flash"
        and settings.embedding_dimensions == 1024
        and settings.resolved_embedding_space_id == "qwen:qwen3.7-text-embedding-flash:1024"
    )
    return {
        "run_id": run_id,
        "mode": "dry_run",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_collection": source,
        "source_vector_count": source_count,
        "source_vector_dimensions": dimensions,
        "source_valid_chunk_text_count": len(valid_texts),
        "source_legacy_vector_count": max(0, len(ids) - versioned),
        "source_versioned_vector_count": versioned,
        "source_current_ready_vector_count": current_ready,
        "target_collection": target,
        "target_exists": target_exists,
        "target_identity_metadata": target_metadata,
        "target_identity_matches": target_identity_matches,
        "target_embedding_provider": settings.embedding_config.provider,
        "target_embedding_model": settings.embedding_config.model,
        "target_embedding_dimensions": settings.embedding_dimensions,
        "target_embedding_space_id": settings.resolved_embedding_space_id,
        "target_profile_matches_qwen37_flash_1024": target_profile_matches,
        "estimated_embedding_text_count": len(valid_texts),
        "estimated_total_characters": sum(len(text) for text in valid_texts),
        "batch_size": batch_size,
        "estimated_embedding_batches": (len(valid_texts) + batch_size - 1) // batch_size,
        "ready_for_authorized_execution": bool(valid_texts) and target_profile_matches and (not target_exists or target_identity_matches),
    }


def safe_error_category(exc: Exception) -> str:
    """Classify failures without persisting provider/server text."""
    name = type(exc).__name__.lower()
    if "permission" in name or "auth" in name:
        return "provider_authentication"
    if "timeout" in name:
        return "provider_timeout"
    if "network" in name or "connect" in name:
        return "provider_network"
    if "dimension" in str(exc).lower():
        return "embedding_dimension"
    return "unknown"


def state_path(run_id: str) -> Path:
    return PROJECT_ROOT / ".runtime" / "embedding-migration" / run_id / "migration-state.json"


def load_state(run_id: str) -> dict[str, Any] | None:
    path = state_path(run_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def make_state(source: str, target: str, run_id: str, ids: list[str], batch_size: int) -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    return {
        "run_id": run_id,
        "source_collection": source,
        "target_collection": target,
        "source_embedding_space_id": "legacy_or_source_identity",
        "target_embedding_space_id": settings.resolved_embedding_space_id,
        "target_dimensions": settings.embedding_dimensions,
        "planned_vector_ids": list(ids),
        "completed_vector_ids": [],
        "failed_vector_ids": [],
        "batch_size": batch_size,
        "current_batch": 0,
        "status": "planned",
        "safe_error_category": "",
        "started_at_utc": now,
        "updated_at_utc": now,
    }


def target_metadata() -> dict[str, Any]:
    return {
        "hnsw:space": "cosine",
        "embedding_provider": settings.embedding_config.provider,
        "embedding_model": settings.embedding_config.model,
        "embedding_dimensions": settings.embedding_dimensions,
        "embedding_space_id": settings.resolved_embedding_space_id,
        "schema_version": 1,
    }


async def execute_migration(source: str, target: str, run_id: str, batch_size: int, recover: bool) -> dict[str, Any]:
    """Run an explicitly-authorized, source-preserving batched migration.

    The state file contains identifiers and safe categories only.  Source
    vectors are read only; successful target batches are recorded atomically
    before the next batch begins so recovery never resubmits confirmed IDs.
    """
    import chromadb

    phase = "preflight"
    state: dict[str, Any] | None = None
    try:
        if source == target:
            raise RuntimeError("source_target_collection_must_differ")
        if not (
            settings.embedding_config.provider == "qwen"
            and settings.embedding_config.model == "qwen3.7-text-embedding-flash"
            and settings.embedding_dimensions == 1024
            and settings.resolved_embedding_space_id == "qwen:qwen3.7-text-embedding-flash:1024"
        ):
            raise RuntimeError("target_embedding_profile_not_configured")

        phase = "source_read"
        client = chromadb.HttpClient(host=VectorStoreService.chroma_http_host(settings.chroma_host), port=settings.chroma_port)
        source_collection = client.get_collection(source)
        rows = source_collection.get(include=["documents", "metadatas"])
        ids = [str(value) for value in rows.get("ids", [])]
        documents = rows.get("documents", [])
        metadatas = rows.get("metadatas", [])
        if not (len(ids) == len(documents) == len(metadatas)):
            raise RuntimeError("source_collection_invalid")

        phase = "state_persist"
        state = load_state(run_id) if recover else None
        if state is None:
            state = make_state(source, target, run_id, ids, batch_size)
            atomic_json(state_path(run_id), state)
        elif state.get("source_collection") != source or state.get("target_collection") != target:
            raise RuntimeError("migration_state_identity_mismatch")

        phase = "target_collection_initialize"
        target_collection = client.get_or_create_collection(name=target, metadata=target_metadata())
        if safe_metadata(target_collection.metadata) != target_metadata():
            raise RuntimeError("target_collection_identity_mismatch")

        phase = "provider_initialize"
        provider = create_embedding_provider(settings)
        completed = set(state.get("completed_vector_ids", []))
        indexed = {vector_id: (document, metadata) for vector_id, document, metadata in zip(ids, documents, metadatas)}
        pending = [vector_id for vector_id in state["planned_vector_ids"] if vector_id not in completed]
        state["status"] = "processing"
        atomic_json(state_path(run_id), state)
        for start in range(0, len(pending), batch_size):
            batch_ids = pending[start : start + batch_size]
            batch_rows = [indexed[vector_id] for vector_id in batch_ids]
            texts = [text for text, _ in batch_rows]
            if not all(isinstance(text, str) and text.strip() for text in texts):
                state["failed_vector_ids"].extend(batch_ids)
                state["status"] = "failed"
                state["safe_error_category"] = "invalid_source_text"
                state["updated_at_utc"] = datetime.now(UTC).isoformat()
                atomic_json(state_path(run_id), state)
                raise RuntimeError("invalid_source_text")
            try:
                phase = "embedding_request"
                embeddings = await provider.aembed_documents(texts)
                if any(len(vector) != settings.embedding_dimensions for vector in embeddings):
                    raise RuntimeError("embedding_dimension_mismatch")
                phase = "target_upsert"
                target_collection.upsert(
                    ids=batch_ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=[dict(metadata or {}) for _, metadata in batch_rows],
                )
            except Exception as exc:
                state["failed_vector_ids"].extend(batch_ids)
                state["status"] = "failed"
                state["safe_error_category"] = safe_error_category(exc)
                state["updated_at_utc"] = datetime.now(UTC).isoformat()
                atomic_json(state_path(run_id), state)
                raise
            completed.update(batch_ids)
            state["completed_vector_ids"] = sorted(completed)
            state["current_batch"] = start // batch_size + 1
            state["updated_at_utc"] = datetime.now(UTC).isoformat()
            atomic_json(state_path(run_id), state)
        state["status"] = "succeeded"
        state["updated_at_utc"] = datetime.now(UTC).isoformat()
        atomic_json(state_path(run_id), state)
        verification = {
            "run_id": run_id,
            "source_collection": source,
            "target_collection": target,
            "expected_vector_count": len(state["planned_vector_ids"]),
            "completed_vector_count": len(state["completed_vector_ids"]),
            "target_vector_count": target_collection.count(),
            "target_identity_matches": safe_metadata(target_collection.metadata) == target_metadata(),
            "target_dimensions": settings.embedding_dimensions,
        }
        atomic_json(state_path(run_id).with_name("verification.json"), verification)
        return verification
    except Exception as exc:
        safe_failure = {"run_id": run_id, "status": "failed", "phase": phase, "error_type": type(exc).__name__, "safe_error_category": safe_error_category(exc)}
        atomic_json(state_path(run_id).with_name("safe-failure.json"), safe_failure)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only embedding index migration preflight")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorized-embedding-migration", action="store_true")
    parser.add_argument("--source-collection", type=collection_name, default="knowledge_chunks")
    parser.add_argument("--target-collection", type=collection_name, default="knowledge_chunks_qwen37_flash_1024")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--recover-run")
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.batch_size > 10:
        parser.error("batch-size must be 1..10")
    if args.dry_run and (args.execute or args.authorized_embedding_migration or args.recover_run):
        parser.error("--dry-run cannot be combined with migration execution flags")
    if not args.dry_run and not args.execute:
        parser.error("choose --dry-run or --execute")
    if args.execute and not args.authorized_embedding_migration:
        parser.error("--execute requires --authorized-embedding-migration")
    try:
        if args.dry_run:
            report = dry_run(args.source_collection, args.target_collection, args.run_id, args.batch_size)
        else:
            report = asyncio.run(
                execute_migration(
                    args.source_collection, args.target_collection, args.run_id, args.batch_size, bool(args.recover_run)
                )
            )
    except Exception as exc:
        print(f"dry-run failed safely: {type(exc).__name__}", file=sys.stderr)
        return 1
    if args.dry_run:
        output = PROJECT_ROOT / ".runtime" / "embedding-migration" / args.run_id / "dry-run.json"
        atomic_json(output, report)
        summary_keys = ("source_collection", "source_vector_count", "target_collection", "target_exists", "ready_for_authorized_execution")
    else:
        atomic_json(state_path(args.run_id).with_name("safe-summary.json"), report)
        summary_keys = ("source_collection", "target_collection", "expected_vector_count", "completed_vector_count", "target_vector_count")
    print(json.dumps({key: report[key] for key in summary_keys}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
