"""Fake-only contracts for durable benchmark-ingestion observation."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

from services.rag_evaluation import BenchmarkDocument


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _runner_module():
    path = PROJECT_ROOT / "scripts" / "run-rag-eval.py"
    spec = importlib.util.spec_from_file_location("benchmark_operation_recovery", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_upload_state_is_safe_and_preallocates_operation_identity(tmp_path):
    runner = _runner_module()
    document = BenchmarkDocument("D03", "D03_group.txt", "synthetic content must not persist")
    state = runner._new_upload_state("run-1", document, "s4-eval-run-1-D03")

    assert state["request_state"] == "planned"
    assert state["operation_id"]
    assert state["safe_filename"] == "D03_group.txt"
    assert len(state["content_hash"]) == 64
    assert "synthetic content" not in json.dumps(state)


def test_late_success_becomes_exact_cleanup_manifest(tmp_path):
    runner = _runner_module()
    state = {"run_id": "run-1", "documents": [
        runner._new_upload_state("run-1", BenchmarkDocument("D03", "D03.txt", "body"), "s4-eval-run-1-D03"),
    ]}
    item = state["documents"][0]
    runner._record_upload_observation(item, {
        "status": "succeeded", "document_id": "doc-late", "version": 1,
    })
    runner._refresh_cleanup_manifest(state)
    runner._write_ingestion_state(tmp_path, state)

    recovered = runner._load_ingestion_state(tmp_path)
    assert recovered["documents"][0]["request_state"] == "ready"
    assert recovered["cleanup_document_ids"] == ["doc-late"]


class _Response:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


class _PollingClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_operation_poll_treats_initial_not_found_as_ambiguous_not_retry(tmp_path, monkeypatch):
    runner = _runner_module()
    client = _PollingClient([
        _Response(404, {"detail": "Document operation not found"}),
        _Response(200, {"operation_id": "op-1", "document_id": "doc-late", "status": "succeeded"}),
    ])

    async def no_wait(_seconds):
        return None

    monkeypatch.setattr(runner.asyncio, "sleep", no_wait)
    observed = await runner._wait_operation(client, "op-1", tmp_path)
    assert observed["status"] == "succeeded"
    assert len(client.calls) == 2
    assert all(call[0] == "GET" for call in client.calls)


@pytest.mark.asyncio
async def test_recovery_observes_existing_operation_without_post(tmp_path, monkeypatch):
    runner = _runner_module()
    output_dir = tmp_path / ".runtime" / "evaluation" / "run-recover"
    output_dir.mkdir(parents=True)
    state = {"run_id": "run-recover", "documents": [
        runner._new_upload_state("run-recover", BenchmarkDocument("D03", "D03.txt", "body"), "s4-eval-run-recover-D03"),
    ], "cleanup_document_ids": []}
    state["documents"][0]["request_state"] = "ambiguous"
    runner._write_ingestion_state(output_dir, state)

    client = _PollingClient([
        _Response(200, {"operation_id": state["documents"][0]["operation_id"], "document_id": "doc-late", "version": 1, "status": "succeeded"}),
    ])

    class _AsyncClient:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return client
        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner.httpx, "AsyncClient", _AsyncClient)
    assert await runner._recover_real_ingestion("run-recover") == 0
    persisted = runner._load_ingestion_state(output_dir)
    assert persisted["cleanup_document_ids"] == ["doc-late"]
    assert all(call[0] == "GET" for call in client.calls)


def test_failed_or_ambiguous_observation_never_claims_ready():
    runner = _runner_module()
    item = runner._new_upload_state("run-1", BenchmarkDocument("D03", "D03.txt", "body"), "s4-eval-run-1-D03")
    runner._record_upload_observation(item, {"operation_id": item["operation_id"], "status": "ambiguous"})
    assert item["request_state"] == "ambiguous"
    runner._record_upload_observation(item, {"operation_id": item["operation_id"], "status": "failed"})
    assert item["request_state"] == "failed"
