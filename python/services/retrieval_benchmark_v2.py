"""Build the reviewable Phase G0 retrieval ground truth without touching v1.

The source benchmark remains authoritative for question wording and semantics.
This module only maps its already-reviewed source basenames to stable document
keys; it never uses an LLM or runtime document UUID.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


BENCHMARK_ID = "enterprise_20docs_retrieval"
BENCHMARK_VERSION = "2.0.0"


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _hash_files(root: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path for path in root.rglob("*") if path.is_file()):
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        digest.update(item.read_bytes())
    return digest.hexdigest()


def build_retrieval_benchmark_v2(source_root: Path, output_root: Path) -> dict[str, Any]:
    """Create a separate frozen candidate and a per-question review report.

    A non-abstention question is marked manual-review only when source mapping
    is absent or ambiguous. This is deliberately conservative: no relevant
    document is inferred from answer wording, entity overlap, or a model.
    """
    documents_payload = json.loads((source_root / "benchmark_documents.json").read_text(encoding="utf-8"))
    questions_payload = json.loads((source_root / "benchmark_questions.json").read_text(encoding="utf-8"))
    documents = documents_payload.get("documents")
    questions = questions_payload.get("questions")
    if not isinstance(documents, list) or not isinstance(questions, list):
        raise ValueError("invalid_source_benchmark")
    by_source: dict[str, list[str]] = {}
    for document in documents:
        if not isinstance(document, dict) or not document.get("id") or not document.get("filename"):
            raise ValueError("invalid_source_document")
        by_source.setdefault(str(document["filename"]), []).append(str(document["id"]))

    ground_truth: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    for question in questions:
        if not isinstance(question, dict):
            raise ValueError("invalid_source_question")
        question_id = str(question.get("question_id") or "")
        expected_sources = tuple(str(item) for item in question.get("expected_sources", []) if str(item))
        answerable = not bool(question.get("requires_abstention", False))
        mapped = [by_source.get(source, []) for source in expected_sources]
        relevant_documents = tuple(item for source_ids in mapped for item in source_ids)
        mapping_complete = all(len(source_ids) == 1 for source_ids in mapped)
        # An abstention question may deliberately name a source that proves
        # the requested fact is absent. That mapping is still deterministic;
        # reserve manual review for answerable questions without a complete
        # source-to-logical-document derivation.
        needs_manual_review = answerable and (not expected_sources or not mapping_complete)
        path = question.get("expected_relation_path", [])
        expected_edges = [list(edge) + (["forward"] if len(edge) == 3 else []) for edge in path if isinstance(edge, list) and len(edge) in (3, 4)]
        row = {
            "question_id": question_id,
            "category": str(question.get("category") or ""),
            "answerable": answerable,
            "relevant_documents": list(dict.fromkeys(relevant_documents)),
            "expected_sources": list(expected_sources),
            "expected_edges": expected_edges,
            "expected_path": expected_edges,
            "relevant_chunks": None,
            "needs_manual_review": needs_manual_review,
        }
        ground_truth.append(row)
        review.append({
            "question_id": question_id,
            "question": str(question.get("question") or ""),
            "category": row["category"],
            "answerable": answerable,
            "relevant_documents": row["relevant_documents"],
            "derivation": "expected_sources basename -> benchmark_documents.id" if expected_sources else "requires_abstention -> no relevant document",
            "needs_manual_review": needs_manual_review,
        })

    source_hash = _hash_files(source_root)
    payload = {
        "benchmark_id": BENCHMARK_ID,
        "benchmark_version": BENCHMARK_VERSION,
        "source_benchmark": {"path": source_root.name, "sha256": source_hash},
        "relevance_level": "document",
        "runtime_document_uuid_policy": "forbidden; map logical document key/source basename after ingestion",
        "coverage_gaps": ["exact_identifier", "semantic_paraphrase", "cross_document", "version_conflict"],
        "questions": ground_truth,
    }
    manifest = {
        "benchmark_id": BENCHMARK_ID,
        "benchmark_version": BENCHMARK_VERSION,
        "document_count": len(documents),
        "question_count": len(ground_truth),
        "source_benchmark_sha256": source_hash,
        "schema": {"relevant_documents": "stable logical document keys", "relevant_chunks": "not available", "expected_edges": "subject,predicate,object,direction"},
        "category_coverage": {category: sum(row["category"] == category for row in ground_truth) for category in sorted({row["category"] for row in ground_truth})},
        "coverage_gaps": payload["coverage_gaps"],
        "needs_manual_review_count": sum(row["needs_manual_review"] for row in ground_truth),
    }
    _atomic_json(output_root / "ground_truth.json", payload)
    _atomic_json(output_root / "benchmark_manifest.json", manifest)
    _atomic_json(output_root / "ground_truth_review.json", {"benchmark_id": BENCHMARK_ID, "benchmark_version": BENCHMARK_VERSION, "review": review})
    lines = ["# Retrieval Ground Truth Review", "", f"Benchmark: `{BENCHMARK_ID}` v`{BENCHMARK_VERSION}`", "", "| Question | Category | Answerable | Relevant logical documents | Derivation | Manual review |", "|---|---|---|---|---|---|"]
    for item in review:
        question = item["question"].replace("|", "\\|").replace("\n", " ")
        documents_value = ", ".join(item["relevant_documents"]) or "—"
        lines.append(f"| {item['question_id']}: {question} | {item['category']} | {item['answerable']} | {documents_value} | {item['derivation']} | {item['needs_manual_review']} |")
    _atomic_text(output_root / "ground_truth_review.md", "\n".join(lines) + "\n")
    return manifest
