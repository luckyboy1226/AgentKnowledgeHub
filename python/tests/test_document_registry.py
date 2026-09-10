"""Offline contract tests for the S3 Mongo document registry."""

from __future__ import annotations

import inspect
from copy import deepcopy

import pytest

from services.document_registry import DocumentRegistry, RegistryError, safe_error, safe_key


def _matches(row, query):
    return all(row.get(key) == value for key, value in query.items())


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def sort(self, key, direction):
        return iter(sorted(self.rows, key=lambda row: row[key], reverse=direction < 0))

    def __iter__(self):
        return iter(self.rows)


class FakeCollection:
    def __init__(self):
        self.rows = []
        self.indexes = []
        self.fail = False

    def create_index(self, keys, unique=False):
        self.indexes.append((keys, unique))

    def find_one(self, query, projection=None, sort=None):
        if self.fail:
            raise RuntimeError("database transport failure")
        rows = [row for row in self.rows if _matches(row, query)]
        if sort:
            key, direction = sort[0]
            rows.sort(key=lambda row: row[key], reverse=direction < 0)
        return deepcopy(rows[0]) if rows else None

    def find(self, query, projection=None):
        if self.fail:
            raise RuntimeError("fake Mongo failure")
        return FakeCursor([deepcopy(row) for row in self.rows if _matches(row, query)])

    def insert_one(self, row):
        if self.fail:
            raise RuntimeError("fake Mongo failure")
        self.rows.append(deepcopy(row))

    def update_one(self, query, update):
        if self.fail:
            raise RuntimeError("fake Mongo failure")
        for row in self.rows:
            if _matches(row, query):
                row.update(deepcopy(update["$set"]))
                return

    def update_many(self, query, update):
        for row in self.rows:
            if _matches(row, query):
                row.update(deepcopy(update["$set"]))


class FakeDatabase:
    def __init__(self):
        self.collections = {
            "documents": FakeCollection(),
            "document_versions": FakeCollection(),
            "document_operations": FakeCollection(),
            "document_operation_locks": FakeCollection(),
        }

    def __getitem__(self, name):
        return self.collections[name]


@pytest.fixture
def registry():
    return DocumentRegistry(FakeDatabase())


def reserve_ready(registry, content=b"first"):
    document, version, changed = registry.reserve("policy.txt", content)
    assert changed
    registry.transition(document["document_id"], version["version"], "ready", chunk_count=2)
    return document, version


def test_first_reservation_creates_default_version_one(registry):
    document, version, changed = registry.reserve("policy.txt", b"first")
    assert changed and version["version"] == 1
    assert document["namespace"] == "default" and version["status"] == "processing"


def test_explicit_logical_key_is_preserved(registry):
    document, _, _ = registry.reserve("display.txt", b"one", logical_key="customer-policy")
    assert document["logical_key"] == "customer-policy"


def test_missing_logical_key_uses_safe_filename(registry):
    document, _, _ = registry.reserve("../../policy.txt", b"one")
    assert document["logical_key"] == ".._.._policy.txt"


def test_document_id_stays_stable_across_new_content(registry):
    first, _ = reserve_ready(registry)
    second, version, changed = registry.reserve("renamed.txt", b"second", logical_key="policy.txt")
    assert changed and version["version"] == 2
    assert second["document_id"] == first["document_id"]


def test_namespace_and_logical_key_form_the_logical_identity(registry):
    first, _, _ = registry.reserve("policy.txt", b"one", logical_key="policy", namespace="north")
    second, _, _ = registry.reserve("policy.txt", b"one", logical_key="policy", namespace="south")
    assert first["document_id"] != second["document_id"]


def test_same_hash_is_unchanged(registry):
    first, version = reserve_ready(registry)
    again, duplicate, changed = registry.reserve("policy.txt", b"first")
    assert not changed and again["document_id"] == first["document_id"]
    assert duplicate["version"] == version["version"]


def test_same_hash_does_not_create_second_version(registry):
    document, _ = reserve_ready(registry)
    registry.reserve("policy.txt", b"first")
    assert len(registry.versions_for(document["document_id"])) == 1


def test_different_hash_creates_version_two(registry):
    document, _ = reserve_ready(registry)
    _, version, changed = registry.reserve("policy.txt", b"second")
    assert changed and version["version"] == 2
    assert registry.find(document["document_id"])["status"] == "processing"


def test_processing_moves_to_ready_with_chunk_count(registry):
    document, version, _ = registry.reserve("policy.txt", b"one")
    ready = registry.transition(document["document_id"], version["version"], "ready", chunk_count=3)
    assert ready["status"] == "ready" and ready["chunk_count"] == 3 and ready["is_current"]


def test_processing_moves_to_failed_with_safe_error(registry):
    document, version, _ = registry.reserve("policy.txt", b"one")
    failed = registry.transition(
        document["document_id"], version["version"], "failed", error="Bearer token-value"
    )
    assert failed["status"] == "failed" and "token-value" not in failed["error_summary"]


def test_new_ready_version_switches_current(registry):
    document, first = reserve_ready(registry)
    _, second, _ = registry.reserve("policy.txt", b"second")
    registry.transition(document["document_id"], second["version"], "ready", chunk_count=4)
    versions = registry.versions_for(document["document_id"])
    assert [row["is_current"] for row in versions] == [False, True]
    assert registry.find(document["document_id"])["current_version"] == 2
    assert first["version"] == 1


def test_invalid_transition_is_rejected(registry):
    document, version, _ = registry.reserve("policy.txt", b"one")
    with pytest.raises(RegistryError, match="Invalid"):
        registry.transition(document["document_id"], version["version"], "deleted")


def test_versions_are_sorted_ascending(registry):
    document, _ = reserve_ready(registry)
    _, second, _ = registry.reserve("policy.txt", b"second")
    registry.transition(document["document_id"], second["version"], "ready")
    assert [row["version"] for row in registry.versions_for(document["document_id"])] == [1, 2]


def test_error_summary_is_redacted_and_limited():
    text = "token=" + "example-value " + "x" * 400
    result = safe_error(text)
    assert "example-value" not in result and len(result) <= 300


def test_path_traversal_text_is_normalized():
    assert safe_key("..\\..\\private.txt") == ".._.._private.txt"


def test_create_next_version_requires_known_document(registry):
    with pytest.raises(RegistryError, match="not found"):
        registry.create_next_version("missing", "policy.txt", b"one")


def test_create_next_version_uses_existing_identity(registry):
    document, _ = reserve_ready(registry)
    same, version, changed = registry.create_next_version(document["document_id"], "new.txt", b"second")
    assert changed and version["version"] == 2 and same["document_id"] == document["document_id"]


def test_expected_unique_indexes_are_declared(registry):
    registry.ensure_indexes()
    assert registry.documents.indexes == [([("namespace", 1), ("logical_key", 1)], True)]
    assert registry.versions.indexes == [([("document_id", 1), ("version", 1)], True)]
    assert registry.operations.indexes[0] == ([("operation_id", 1)], True)
    assert registry.operation_locks.indexes[0] == ([("document_id", 1)], True)


def test_mongo_failure_becomes_safe_registry_error(registry):
    registry.documents.fail = True
    with pytest.raises(RegistryError) as error:
        registry.reserve("policy.txt", b"one")
    assert "transport failure" not in str(error.value)


def test_deleting_and_deleted_are_explicit_transitions(registry):
    document, version = reserve_ready(registry)
    deleting = registry.mark_deleting(document["document_id"])
    deleted = registry.mark_deleted(document["document_id"], version["version"])
    assert deleting["status"] == "deleting" and deleted["status"] == "deleted"
    assert registry.find(document["document_id"])["deleted_at"] is not None


def test_failed_update_keeps_prior_ready_document_available(registry):
    document, _ = reserve_ready(registry)
    _, second, _ = registry.reserve("policy.txt", b"second")
    registry.transition(document["document_id"], second["version"], "failed", error="parse failed")
    refreshed = registry.find(document["document_id"])
    assert refreshed["status"] == "ready" and refreshed["current_version"] == 1


def test_upload_keeps_legacy_signature_and_accepts_logical_key():
    from api.main import upload_document

    parameters = inspect.signature(upload_document).parameters
    assert "file" in parameters and "logical_key" in parameters
    assert getattr(parameters["logical_key"].default, "default", None) is None


def test_new_document_endpoints_do_not_accept_local_path_parameters():
    from api.main import app

    document_routes = {
        route.path: route for route in app.routes if getattr(route, "path", "").startswith("/api/documents/")
    }
    assert "/api/documents/{document_id}" in document_routes
    assert "file_path" not in str(document_routes["/api/documents/{document_id}"].dependant)
