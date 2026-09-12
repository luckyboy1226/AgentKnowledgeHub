"""Offline API contract tests; all document dependencies are in-memory fakes."""

from __future__ import annotations

import inspect
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

import api.main as api
from services.document_processor import ProcessingTimeoutError
from services.document_update_coordinator import OperationBusyError


class FakeRegistry:
    def __init__(self):
        self.operations = object()
        self.operation_locks = object()
        self.rows = {
            "doc-1": {
                "document_id": "doc-1", "namespace": "default", "logical_key": "report.txt",
                "filename": "report.txt", "current_version": 1, "status": "ready",
                "updated_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            }
        }
        self.versions = {"doc-1": [{"version": 1, "is_current": True, "chunk_count": 2, "status": "ready"}]}

    def find(self, document_id):
        return self.rows.get(document_id)

    def versions_for(self, document_id):
        return self.versions.get(document_id, [])

    def list_documents(self):
        return list(self.rows.values())


class FakeJournal:
    def __init__(self):
        self.records = {
            "op-1": {
                "operation_id": "op-1", "operation_type": "create", "document_id": "doc-1",
                "version": 1, "status": "succeeded", "completed_steps": ["mongo_ready"],
                "compensation_steps": [], "error_summary": None,
                "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
                "updated_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            }
        }

    def get(self, operation_id):
        return self.records.get(operation_id)


class FakeCoordinator:
    def __init__(self):
        self.journal = FakeJournal()
        self.calls = []
        self.error = None
        self.create_changed = True
        self.delete_status = "succeeded"

    async def create_document_version(self, **kwargs):
        self.calls.append(("create", kwargs))
        if self.error:
            raise self.error
        return self._result(kwargs, version=1, changed=self.create_changed)

    async def update_document(self, document_id, **kwargs):
        self.calls.append(("update", {"document_id": document_id, **kwargs}))
        if self.error:
            raise self.error
        return self._result(kwargs, document_id=document_id, version=2)

    async def delete_document(self, document_id, **kwargs):
        self.calls.append(("delete", {"document_id": document_id, **kwargs}))
        if self.error:
            raise self.error
        return {"operation_id": kwargs["operation_id"], "status": self.delete_status}

    @staticmethod
    def _result(kwargs, document_id="doc-1", version=1, changed=True):
        return {
            "operation_id": kwargs["operation_id"], "document_id": document_id, "version": version,
            "content_hash": "hash-1", "status": "succeeded", "changed": changed,
            "completed_steps": ["mongo_ready"],
            "processing_metadata": {"chunk_count": 2, "entity_count": 2, "relation_count": 1},
        }


@pytest.fixture
def client(monkeypatch):
    registry, coordinator = FakeRegistry(), FakeCoordinator()
    monkeypatch.setattr(api, "document_registry", registry)
    monkeypatch.setattr(api, "document_coordinator", coordinator)
    api.app.state.document_registry = registry
    api.app.state.document_coordinator = coordinator
    return TestClient(api.app, raise_server_exceptions=False), registry, coordinator


def upload(client, path="report.txt", body=b"valid text", **data):
    return client.post("/api/documents", files={"file": (path, body, "text/plain")}, data=data)


def test_post_first_upload_returns_operation_and_safe_response(client):
    http, _, coordinator = client
    response = upload(http)
    assert response.status_code == 200
    payload = response.json()
    assert payload["document_id"] == "doc-1" and payload["operation_id"]
    assert payload["status_url"].startswith("/api/document-operations/")
    assert coordinator.calls[0][1]["content"] == b"valid text"


def test_operation_response_exposes_only_safe_failure_classification(client):
    http, _, coordinator = client
    coordinator.journal.records["op-1"].update(
        error_phase="extract",
        error_category="provider_timeout",
        error_type="APITimeoutError",
        error_summary="Knowledge extraction failed",
    )
    response = http.get("/api/document-operations/op-1")
    assert response.status_code == 200
    payload = response.json()
    assert payload["error_phase"] == "extract"
    assert payload["error_category"] == "provider_timeout"
    assert payload["error_type"] == "APITimeoutError"
    assert payload["chunk_index"] is None
    assert "prompt" not in payload and "response" not in payload


def test_post_defaults_namespace(client):
    http, _, coordinator = client
    assert upload(http).json()["namespace"] == "default"
    assert coordinator.calls[0][1]["namespace"] == "default"


def test_post_passes_explicit_logical_key(client):
    http, _, coordinator = client
    assert upload(http, logical_key="customer-policy").status_code == 200
    assert coordinator.calls[0][1]["logical_key"] == "customer-policy"


def test_post_forwards_client_generated_operation_id(client):
    http, _, coordinator = client
    operation_id = "4e1a5cfb-58a8-4fa1-9a7a-9f0783376c0a"
    response = http.post(
        "/api/documents",
        headers={"X-Operation-Id": operation_id},
        files={"file": ("report.txt", b"valid text", "text/plain")},
    )
    assert response.status_code == 200
    assert coordinator.calls[0][1]["operation_id"] == operation_id


def test_post_rejects_invalid_client_generated_operation_id(client):
    http, _, coordinator = client
    response = http.post(
        "/api/documents",
        headers={"X-Operation-Id": "not-a-uuid"},
        files={"file": ("report.txt", b"valid text", "text/plain")},
    )
    assert response.status_code == 422 and not coordinator.calls


def test_post_rejects_empty_file(client):
    http, _, _ = client
    assert upload(http, body=b"").status_code == 400


def test_post_rejects_oversize_file(client, monkeypatch):
    http, _, _ = client
    monkeypatch.setattr(api, "MAX_DOCUMENT_UPLOAD_BYTES", 4)
    assert upload(http, body=b"12345").status_code == 413


def test_post_rejects_unsupported_type(client):
    http, _, _ = client
    assert upload(http, path="report.exe").status_code == 400


def test_post_normalizes_traversal_filename(client):
    http, _, coordinator = client
    assert upload(http, path=r"..\..\private.txt").status_code == 200
    assert coordinator.calls[0][1]["filename"] == "private.txt"


def test_post_unchanged_response_is_explicit(client):
    http, _, coordinator = client
    coordinator.create_changed = False
    response = upload(http)
    assert response.status_code == 200 and response.json()["changed"] is False


def test_put_creates_next_version(client):
    http, _, coordinator = client
    response = http.put("/api/documents/doc-1", files={"file": ("next.txt", b"next", "text/plain")})
    assert response.status_code == 200 and response.json()["version"] == 2
    assert coordinator.calls[0][0] == "update"


def test_put_path_document_id_cannot_be_overridden_by_form(client):
    http, _, coordinator = client
    response = http.put("/api/documents/doc-1", files={"file": ("next.txt", b"next", "text/plain")}, data={"document_id": "other"})
    assert response.status_code == 200 and coordinator.calls[0][1]["document_id"] == "doc-1"


def test_put_returns_not_found_for_unknown_document(client):
    http, _, _ = client
    assert http.put("/api/documents/missing", files={"file": ("next.txt", b"next", "text/plain")}).status_code == 404


def test_put_returns_conflict_for_active_document(client):
    http, registry, _ = client
    registry.rows["doc-1"]["status"] = "processing"
    assert http.put("/api/documents/doc-1", files={"file": ("next.txt", b"next", "text/plain")}).status_code == 409


def test_put_maps_lease_conflict_to_409(client):
    http, _, coordinator = client
    coordinator.error = OperationBusyError("busy")
    assert http.put("/api/documents/doc-1", files={"file": ("next.txt", b"next", "text/plain")}).status_code == 409


def test_delete_calls_document_scoped_coordinator(client):
    http, _, coordinator = client
    response = http.delete("/api/documents/doc-1")
    assert response.status_code == 200 and coordinator.calls[0][0] == "delete"
    assert coordinator.calls[0][1]["document_id"] == "doc-1"


def test_delete_returns_not_found(client):
    http, _, _ = client
    assert http.delete("/api/documents/missing").status_code == 404


def test_delete_is_idempotent_for_deleted_document(client):
    http, registry, coordinator = client
    registry.rows["doc-1"]["status"] = "deleted"
    response = http.delete("/api/documents/doc-1")
    assert response.status_code == 200 and response.json()["changed"] is False and not coordinator.calls


def test_delete_conflicts_while_processing(client):
    http, registry, _ = client
    registry.rows["doc-1"]["status"] = "processing"
    assert http.delete("/api/documents/doc-1").status_code == 409


def test_delete_exposes_cleanup_pending(client):
    http, _, coordinator = client
    coordinator.delete_status = "cleanup_pending"
    assert http.delete("/api/documents/doc-1").json()["status"] == "cleanup_pending"


def test_get_document(client):
    http, _, _ = client
    assert http.get("/api/documents/doc-1").json()["document_id"] == "doc-1"


def test_get_versions(client):
    http, _, _ = client
    assert http.get("/api/documents/doc-1/versions").json()[0]["version"] == 1


def test_get_status(client):
    http, _, _ = client
    assert http.get("/api/documents/doc-1/status").json()["status"] == "ready"


def test_get_operation_returns_safe_fields(client):
    http, _, _ = client
    payload = http.get("/api/document-operations/op-1").json()
    assert payload["operation_id"] == "op-1" and payload["document_status"] == "ready" and "vector_ids" not in payload


def test_get_operation_not_found(client):
    http, _, _ = client
    assert http.get("/api/document-operations/missing").status_code == 404


def test_provider_unavailable_maps_to_503_without_secret(client):
    http, _, coordinator = client
    coordinator.error = RuntimeError("token=do-not-leak")
    response = upload(http)
    assert response.status_code == 503 and "do-not-leak" not in response.text


def test_processing_timeout_maps_to_504(client):
    http, _, coordinator = client
    coordinator.error = ProcessingTimeoutError("timeout")
    assert upload(http).status_code == 504


def test_legacy_upload_url_is_a_compatibility_alias(client):
    http, _, coordinator = client
    response = http.post("/api/ingest/upload", files={"file": ("legacy.txt", b"body", "text/plain")})
    assert response.status_code == 200 and coordinator.calls[0][0] == "create"


def test_legacy_admin_update_rejects_file_paths_without_coordinator_call(client):
    http, _, coordinator = client
    response = http.post("/api/admin/update", json={"file_path": r"C:\\sensitive\\report.pdf"})

    assert response.status_code == 410
    assert "/api/documents" in response.json()["detail"]
    assert not coordinator.calls


def test_legacy_filename_delete_is_retired_without_coordinator_call(client):
    http, _, coordinator = client
    response = http.delete("/api/ingest/documents/report.txt")

    assert response.status_code == 410
    assert "/api/documents" in response.json()["detail"]
    assert not coordinator.calls


def test_legacy_batch_uses_plain_default_identity_values(client):
    http, _, coordinator = client
    response = http.post(
        "/api/ingest/batch",
        files=[("files", ("one.txt", b"one", "text/plain")), ("files", ("two.txt", b"two", "text/plain"))],
    )
    assert response.status_code == 200
    assert all(call[1]["namespace"] == "default" and call[1]["logical_key"] is None for call in coordinator.calls)


def test_legacy_document_list_keeps_frontend_fields_and_adds_registry_fields(client):
    http, _, _ = client
    row = http.get("/api/ingest/documents").json()[0]
    assert {"id", "name", "size", "upload_time", "chunks_count", "document_id", "version", "status", "changed"}.issubset(row)
    assert row["legacy"] is False


def test_api_signature_has_no_local_path_parameter():
    assert "file_path" not in inspect.signature(api.upload_document).parameters
    assert "file_path" not in inspect.signature(api.update_registered_document).parameters


def test_fake_coordinator_receives_only_uploaded_bytes(client):
    http, _, coordinator = client
    upload(http, body=b"bytes only")
    call = coordinator.calls[0][1]
    assert call["content"] == b"bytes only" and "file_path" not in call


def test_no_unprotected_recover_or_reconcile_route_is_published():
    paths = {route.path for route in api.app.routes}
    assert not any("reconcile" in path or "recover" in path for path in paths)


def test_lifecycle_builder_reuses_injected_services_without_creating_clients(tmp_path):
    registry, vectors, graph, provider = FakeRegistry(), object(), object(), object()
    coordinator = api.build_document_coordinator(registry, vectors, graph, provider, temp_root=tmp_path)
    assert coordinator.registry is registry and coordinator.vector_store is vectors and coordinator.knowledge_graph is graph
    assert coordinator.processor.parser.llm is provider and coordinator.processor.extractor.llm is provider


def test_lifecycle_builder_has_no_close_side_effect_on_injected_clients(tmp_path):
    registry, vectors, graph, provider = FakeRegistry(), object(), object(), object()
    coordinator = api.build_document_coordinator(registry, vectors, graph, provider, temp_root=tmp_path)
    assert not hasattr(coordinator, "close")
