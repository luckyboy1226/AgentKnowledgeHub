"""Offline Saga tests: all dependencies are in-memory fakes, never real stores."""

from __future__ import annotations

from copy import deepcopy
from datetime import timedelta

import pytest

from agents.doc_parser_agent import DocType, DocumentChunk
from agents.knowledge_extract_agent import Entity, Relation
from services.document_registry import utcnow
from services.document_update_coordinator import (
    DocumentUpdateCoordinator,
    OperationBusyError,
    PreparedDocument,
)


class FakeRegistry:
    def __init__(self):
        self.documents = {}
        self.versions = {}
        self.counter = 0
        self.fail_ready = False

    def reserve(self, filename, content, logical_key=None, namespace="default"):
        key = (namespace, logical_key or filename)
        document = next((item for item in self.documents.values() if item["key"] == key), None)
        content_hash = f"hash:{content.decode(errors='ignore')}"
        if document is None:
            self.counter += 1
            document = {"document_id": f"doc-{self.counter}", "key": key, "current_version": None, "status": "processing"}
            self.documents[document["document_id"]] = document
        versions = self.versions.setdefault(document["document_id"], [])
        existing = next((row for row in versions if row["content_hash"] == content_hash), None)
        if existing:
            return deepcopy(document), deepcopy(existing), False
        record = {"document_id": document["document_id"], "version": len(versions) + 1, "content_hash": content_hash, "status": "processing", "is_current": False}
        versions.append(record)
        return deepcopy(document), deepcopy(record), True

    def create_next_version(self, document_id, filename, content):
        document = self.documents[document_id]
        return self.reserve(filename, content, document["key"][1], document["key"][0])

    def find(self, document_id):
        return deepcopy(self.documents.get(document_id))

    def transition(self, document_id, version, target, *, chunk_count=0, error=None):
        if target == "ready" and self.fail_ready:
            raise RuntimeError("mongo ready failure")
        row = next(row for row in self.versions[document_id] if row["version"] == version)
        row["status"] = target
        if target == "ready":
            for old in self.versions[document_id]:
                old["is_current"] = False
            row["is_current"] = True
            row["chunk_count"] = chunk_count
            self.documents[document_id]["current_version"] = version
            self.documents[document_id]["status"] = "ready"
        elif target == "failed":
            row["error_summary"] = str(error)[:30]
            self.documents[document_id]["status"] = "ready" if self.documents[document_id]["current_version"] else "failed"
        elif target == "deleted":
            self.documents[document_id]["status"] = "deleted"
        return deepcopy(row)

    def mark_deleting(self, document_id, version):
        self.documents[document_id]["status"] = "deleting"
        return self.transition(document_id, version, "deleting")

    def mark_deleted(self, document_id, version):
        return self.transition(document_id, version, "deleted")

    def seed_current(self, filename="same.txt", content=b"old"):
        document, row, _ = self.reserve(filename, content)
        self.transition(document["document_id"], row["version"], "ready", chunk_count=1)
        return document["document_id"]


class FakeLocks:
    def __init__(self):
        self.rows = {}

    def find_one(self, query):
        row = self.rows.get(query["document_id"])
        return deepcopy(row) if row else None


class FakeJournal:
    def __init__(self):
        self.records = {}
        self.lock_rows = {}
        self.locks = FakeLocks()

    def get(self, operation_id):
        row = self.records.get(operation_id)
        return deepcopy(row) if row else None

    def create(self, record):
        self.records.setdefault(record["operation_id"], deepcopy(record))
        return self.get(record["operation_id"])

    def update(self, operation_id, **fields):
        self.records[operation_id].update(fields)
        return self.get(operation_id)

    def complete_step(self, operation_id, step, **fields):
        record = self.records[operation_id]
        if step not in record["completed_steps"]:
            record["completed_steps"].append(step)
        record.update(fields)
        return self.get(operation_id)

    def complete_compensation(self, operation_id, step):
        record = self.records[operation_id]
        if step not in record["compensation_steps"]:
            record["compensation_steps"].append(step)
        return self.get(operation_id)

    def acquire_lock(self, document_id, operation_id, lease_seconds=300):
        now = utcnow()
        owner = self.lock_rows.get(document_id)
        if owner and owner["operation_id"] != operation_id and owner["expires_at"] > now:
            return False
        row = {"document_id": document_id, "operation_id": operation_id, "expires_at": now + timedelta(seconds=lease_seconds)}
        self.lock_rows[document_id] = row
        self.locks.rows[document_id] = row
        return True

    def release_lock(self, document_id, operation_id):
        owner = self.lock_rows.get(document_id)
        if not owner or owner["operation_id"] != operation_id:
            return False
        del self.lock_rows[document_id]
        self.locks.rows.pop(document_id, None)
        return True

    def incomplete(self):
        return [self.get(op_id) for op_id, row in self.records.items() if row["status"] in {"running", "compensating", "cleanup_pending", "needs_reconciliation"}]


class FakeProcessor:
    def __init__(self):
        self.fail = False
        self.calls = 0

    async def prepare(self, *, content, filename, document_id, version, content_hash, operation_id):
        self.calls += 1
        if self.fail:
            raise RuntimeError("processor failure")
        return PreparedDocument(
            chunks=[DocumentChunk("one", "legacy", 0, DocType.TEXT, {"source": filename})],
            entities=[Entity("Alice", "Person", "Owner")],
            relations=[Relation("Alice", "owns", "Project", 0.9)],
        )


class FakeVersionStore:
    def __init__(self, name, events):
        self.name = name
        self.events = events
        self.staged = {}
        self.current = {}
        self.fail = set()

    def _maybe_fail(self, action):
        if action in self.fail:
            raise RuntimeError(f"{self.name} {action} failure")

    async def stage_document_version(self, document_id, version, *args, **kwargs):
        self._maybe_fail("stage")
        self.events.append(f"{self.name}.stage")
        self.staged[(document_id, version)] = True
        return 1

    async def activate_document_version(self, document_id, version):
        self._maybe_fail("activate")
        self.events.append(f"{self.name}.activate:{version}")
        self.current[document_id] = version
        return 1

    async def deactivate_document_version(self, document_id, version):
        self._maybe_fail("deactivate")
        self.events.append(f"{self.name}.deactivate:{version}")
        if self.current.get(document_id) == version:
            self.current.pop(document_id)
        return 1

    async def delete_document_version(self, document_id, version):
        self._maybe_fail("delete_version")
        self.events.append(f"{self.name}.delete_version:{version}")
        self.staged.pop((document_id, version), None)
        return 1

    async def delete_document(self, document_id):
        self._maybe_fail("delete_document")
        self.events.append(f"{self.name}.delete_document")
        for key in list(self.staged):
            if key[0] == document_id:
                del self.staged[key]
        self.current.pop(document_id, None)
        return 1


class FakeVector(FakeVersionStore):
    def __init__(self, events):
        super().__init__("vector", events)

    async def stage_document_version(self, document_id, version, chunks, content_hash, **kwargs):
        await super().stage_document_version(document_id, version)
        self.ids = {key: [f"{key[0]}:v{key[1]}:c0"] for key in self.staged}

    def list_document_vector_ids(self, document_id, version):
        return self.ids.get((document_id, version), [])


class FakeGraph(FakeVersionStore):
    def __init__(self, events):
        super().__init__("graph", events)

    async def stage_document_version(self, document_id, version, content_hash, source, entities, relations):
        await super().stage_document_version(document_id, version)
        self.evidence = {key: [{"evidence_key": f"e:{key[0]}:{key[1]}"}] for key in self.staged}

    async def list_document_evidence(self, document_id, version):
        return self.evidence.get((document_id, version), [])


@pytest.fixture
def setup():
    registry = FakeRegistry()
    journal = FakeJournal()
    events = []
    vector = FakeVector(events)
    graph = FakeGraph(events)
    processor = FakeProcessor()
    coordinator = DocumentUpdateCoordinator(registry, vector, graph, processor, journal, lease_seconds=1)
    return coordinator, registry, journal, vector, graph, processor, events


@pytest.mark.asyncio
async def test_create_success_follows_strict_saga_order(setup):
    coordinator, registry, journal, vector, graph, processor, events = setup
    result = await coordinator.create_document_version(filename="a.txt", content=b"new", operation_id="op-1")
    assert result["status"] == "succeeded"
    assert events == ["vector.stage", "graph.stage", "vector.activate:1", "graph.activate:1"]
    assert registry.find("doc-1")["current_version"] == 1


@pytest.mark.asyncio
async def test_same_hash_returns_unchanged_without_processing(setup):
    coordinator, _, _, _, _, processor, _ = setup
    await coordinator.create_document_version(filename="a.txt", content=b"same", operation_id="op-1")
    result = await coordinator.create_document_version(filename="a.txt", content=b"same", operation_id="op-2")
    assert result["changed"] is False and processor.calls == 1


@pytest.mark.asyncio
async def test_update_increments_version(setup):
    coordinator, registry, _, _, _, _, _ = setup
    document_id = registry.seed_current()
    result = await coordinator.update_document(document_id, filename="same.txt", content=b"new", operation_id="op-2")
    assert result["version"] == 2 and registry.find(document_id)["current_version"] == 2


@pytest.mark.asyncio
async def test_processor_failure_marks_version_failed(setup):
    coordinator, registry, journal, _, _, processor, _ = setup
    processor.fail = True
    with pytest.raises(RuntimeError, match="processor failure"):
        await coordinator.create_document_version(filename="a.txt", content=b"new", operation_id="op-1")
    assert journal.get("op-1")["status"] == "failed"
    assert registry.versions["doc-1"][0]["status"] == "failed"


@pytest.mark.asyncio
async def test_vector_stage_failure_has_no_graph_compensation(setup):
    coordinator, _, journal, vector, graph, _, events = setup
    vector.fail.add("stage")
    with pytest.raises(RuntimeError, match="vector stage failure"):
        await coordinator.create_document_version(filename="a.txt", content=b"new", operation_id="op-1")
    assert "graph.stage" not in events and journal.get("op-1")["compensation_steps"] == []


@pytest.mark.asyncio
async def test_graph_stage_failure_rolls_back_only_vector_stage(setup):
    coordinator, _, journal, _, graph, _, events = setup
    graph.fail.add("stage")
    with pytest.raises(RuntimeError, match="graph stage failure"):
        await coordinator.create_document_version(filename="a.txt", content=b"new", operation_id="op-1")
    assert "vector.delete_version:1" in events
    assert "graph.delete_version:1" not in events
    assert "vector_stage_deleted" in journal.get("op-1")["compensation_steps"]


@pytest.mark.asyncio
async def test_vector_activate_failure_compensates_stages(setup):
    coordinator, _, journal, vector, _, _, events = setup
    vector.fail.add("activate")
    with pytest.raises(RuntimeError, match="vector activate failure"):
        await coordinator.create_document_version(filename="a.txt", content=b"new", operation_id="op-1")
    steps = journal.get("op-1")["compensation_steps"]
    assert "graph_stage_deleted" in steps and "vector_stage_deleted" in steps


@pytest.mark.asyncio
async def test_graph_activate_failure_restores_previous_vector_current(setup):
    coordinator, registry, journal, vector, graph, _, events = setup
    document_id = registry.seed_current()
    vector.current[document_id] = 1
    graph.current[document_id] = 1
    graph.fail.add("activate")
    with pytest.raises(RuntimeError, match="graph activate failure"):
        await coordinator.update_document(document_id, filename="same.txt", content=b"new", operation_id="op-2")
    assert vector.current[document_id] == 1
    assert "previous_vector_restored" in journal.get("op-2")["compensation_steps"]
    assert events.index("vector.deactivate:2") < events.index("vector.activate:1")


@pytest.mark.asyncio
async def test_mongo_ready_failure_compensates_both_activated_stores(setup):
    coordinator, registry, journal, vector, graph, _, _ = setup
    document_id = registry.seed_current()
    vector.current[document_id] = 1
    graph.current[document_id] = 1
    registry.fail_ready = True
    with pytest.raises(RuntimeError, match="mongo ready failure"):
        await coordinator.update_document(document_id, filename="same.txt", content=b"new", operation_id="op-2")
    steps = journal.get("op-2")["compensation_steps"]
    assert {"vector_deactivated", "graph_deactivated", "previous_vector_restored", "previous_graph_restored"}.issubset(steps)


@pytest.mark.asyncio
async def test_old_current_is_retained_after_failed_update(setup):
    coordinator, registry, _, vector, graph, _, _ = setup
    document_id = registry.seed_current()
    vector.current[document_id] = graph.current[document_id] = 1
    graph.fail.add("activate")
    with pytest.raises(RuntimeError):
        await coordinator.update_document(document_id, filename="same.txt", content=b"new")
    assert registry.find(document_id)["current_version"] == 1


@pytest.mark.asyncio
async def test_failed_new_version_is_not_current(setup):
    coordinator, registry, _, _, graph, _, _ = setup
    graph.fail.add("stage")
    with pytest.raises(RuntimeError):
        await coordinator.create_document_version(filename="a.txt", content=b"new")
    assert not registry.versions["doc-1"][0]["is_current"]


@pytest.mark.asyncio
async def test_compensation_only_uses_completed_steps(setup):
    coordinator, _, journal, vector, graph, _, events = setup
    graph.fail.add("stage")
    with pytest.raises(RuntimeError):
        await coordinator.create_document_version(filename="a.txt", content=b"new", operation_id="op-1")
    assert "vector.delete_version:1" in events and "graph.deactivate:1" not in events
    assert journal.get("op-1")["completed_steps"] == ["mongo_reserved", "prepared", "vector_staged"]


@pytest.mark.asyncio
async def test_compensation_failure_needs_reconciliation(setup):
    coordinator, _, journal, vector, graph, _, _ = setup
    graph.fail.add("stage")
    vector.fail.add("delete_version")
    with pytest.raises(RuntimeError):
        await coordinator.create_document_version(filename="a.txt", content=b"new", operation_id="op-1")
    assert journal.get("op-1")["status"] == "needs_reconciliation"


@pytest.mark.asyncio
async def test_same_operation_retry_does_not_duplicate_writes(setup):
    coordinator, _, _, _, _, processor, events = setup
    await coordinator.create_document_version(filename="a.txt", content=b"new", operation_id="op-1")
    await coordinator.create_document_version(filename="a.txt", content=b"different", operation_id="op-1")
    assert processor.calls == 1 and events.count("vector.stage") == 1


@pytest.mark.asyncio
async def test_same_document_concurrent_operation_is_rejected(setup):
    coordinator, registry, journal, _, _, _, _ = setup
    document_id = registry.seed_current()
    assert journal.acquire_lock(document_id, "owner", 300)
    with pytest.raises(OperationBusyError):
        await coordinator.update_document(document_id, filename="same.txt", content=b"new", operation_id="other")


@pytest.mark.asyncio
async def test_different_documents_can_update_independently(setup):
    coordinator, registry, _, _, _, _, _ = setup
    first = registry.seed_current("one.txt", b"old")
    second = registry.seed_current("two.txt", b"old")
    await coordinator.update_document(first, filename="one.txt", content=b"new1")
    await coordinator.update_document(second, filename="two.txt", content=b"new2")
    assert registry.find(first)["current_version"] == 2 and registry.find(second)["current_version"] == 2


def test_expired_lock_can_be_recovered(setup):
    _, _, journal, _, _, _, _ = setup
    journal.lock_rows["doc-a"] = {"document_id": "doc-a", "operation_id": "old", "expires_at": utcnow() - timedelta(seconds=1)}
    journal.locks.rows["doc-a"] = journal.lock_rows["doc-a"]
    assert journal.acquire_lock("doc-a", "new", 1)


def test_non_owner_cannot_release_lock(setup):
    _, _, journal, _, _, _, _ = setup
    journal.acquire_lock("doc-a", "owner", 1)
    assert not journal.release_lock("doc-a", "other")
    assert journal.release_lock("doc-a", "owner")


@pytest.mark.asyncio
async def test_delete_success_is_two_phase_and_precise(setup):
    coordinator, registry, journal, vector, graph, _, events = setup
    document_id = registry.seed_current()
    vector.current[document_id] = graph.current[document_id] = 1
    result = await coordinator.delete_document(document_id, operation_id="delete-1")
    assert result["status"] == "succeeded" and registry.find(document_id)["status"] == "deleted"
    assert events == ["vector.deactivate:1", "graph.deactivate:1", "vector.delete_document", "graph.delete_document"]
    assert "mongo_deleted" in journal.get("delete-1")["completed_steps"]


@pytest.mark.asyncio
async def test_delete_cleanup_failure_stays_logically_deleted(setup):
    coordinator, registry, journal, vector, graph, _, _ = setup
    document_id = registry.seed_current()
    vector.current[document_id] = graph.current[document_id] = 1
    graph.fail.add("delete_document")
    result = await coordinator.delete_document(document_id, operation_id="delete-1")
    assert result["status"] == "cleanup_pending" and registry.find(document_id)["status"] == "deleted"
    assert journal.get("delete-1")["status"] == "cleanup_pending"


@pytest.mark.asyncio
async def test_deleted_document_is_not_restored_when_cleanup_pending(setup):
    coordinator, registry, _, vector, graph, _, _ = setup
    document_id = registry.seed_current()
    vector.current[document_id] = graph.current[document_id] = 1
    vector.fail.add("delete_document")
    await coordinator.delete_document(document_id)
    assert registry.find(document_id)["status"] == "deleted"


@pytest.mark.asyncio
async def test_recover_cleanup_pending_completes_exact_cleanup(setup):
    coordinator, registry, journal, vector, graph, _, _ = setup
    document_id = registry.seed_current()
    vector.current[document_id] = graph.current[document_id] = 1
    graph.fail.add("delete_document")
    await coordinator.delete_document(document_id, operation_id="delete-1")
    graph.fail.clear()
    recovered = await coordinator.recover_operation("delete-1")
    assert recovered["status"] == "succeeded" and journal.get("delete-1")["status"] == "succeeded"


@pytest.mark.asyncio
async def test_reconcile_ignores_succeeded_operations(setup):
    coordinator, _, journal, _, _, _, _ = setup
    await coordinator.create_document_version(filename="a.txt", content=b"new", operation_id="op-1")
    assert await coordinator.reconcile_incomplete_operations() == []
    assert journal.get("op-1")["status"] == "succeeded"


@pytest.mark.asyncio
async def test_reconcile_recovers_expired_running_operation(setup):
    coordinator, _, journal, vector, graph, _, _ = setup
    journal.records["op-x"] = {
        "operation_id": "op-x", "operation_type": "update", "document_id": "doc-x", "version": 2,
        "previous_current_version": 1, "status": "running", "completed_steps": ["mongo_reserved", "vector_staged"],
        "compensation_steps": [], "error_summary": None,
    }
    journal.lock_rows["doc-x"] = {"document_id": "doc-x", "operation_id": "old", "expires_at": utcnow() - timedelta(seconds=1)}
    journal.locks.rows["doc-x"] = journal.lock_rows["doc-x"]
    recovered = await coordinator.reconcile_incomplete_operations()
    assert recovered[0]["status"] == "failed"
    assert "vector.delete_version:2" in vector.events
    assert "graph.delete_version:2" not in graph.events


@pytest.mark.asyncio
async def test_recover_operation_after_interruption_is_idempotent(setup):
    coordinator, _, journal, vector, _, _, _ = setup
    journal.records["op-x"] = {
        "operation_id": "op-x", "operation_type": "update", "document_id": "doc-x", "version": 2,
        "previous_current_version": None, "status": "compensating", "completed_steps": ["mongo_reserved", "vector_staged"],
        "compensation_steps": [], "error_summary": "interrupted",
    }
    result = await coordinator.recover_operation("op-x")
    again = await coordinator.recover_operation("op-x")
    assert result["status"] == "failed" and again["status"] == "failed"
    assert vector.events.count("vector.delete_version:2") == 1


@pytest.mark.asyncio
async def test_error_summary_is_safe_and_journal_has_no_payload(setup):
    coordinator, _, journal, _, _, processor, _ = setup
    processor.fail = True
    with pytest.raises(RuntimeError):
        await coordinator.create_document_version(filename="a.txt", content=b"body must not persist", operation_id="op-1")
    operation = journal.get("op-1")
    assert "body must not persist" not in str(operation)
    assert "content" not in operation and "embedding" not in operation and "prompt" not in operation


@pytest.mark.asyncio
async def test_compensation_parameters_are_exact_document_and_version(setup):
    coordinator, registry, _, vector, graph, _, _ = setup
    document_id = registry.seed_current()
    graph.fail.add("stage")
    with pytest.raises(RuntimeError):
        await coordinator.update_document(document_id, filename="same.txt", content=b"new")
    assert (document_id, 2) not in vector.staged
    assert (document_id, 2) not in graph.staged


@pytest.mark.asyncio
async def test_legacy_document_is_not_touched_by_other_document_saga(setup):
    coordinator, registry, _, vector, graph, _, _ = setup
    legacy_id = registry.seed_current("legacy.txt", b"old")
    vector.current[legacy_id] = graph.current[legacy_id] = 1
    await coordinator.create_document_version(filename="new.txt", content=b"new")
    assert vector.current[legacy_id] == 1 and graph.current[legacy_id] == 1


@pytest.mark.asyncio
async def test_operation_journal_records_vector_and_graph_locators(setup):
    coordinator, _, journal, _, _, _, _ = setup
    await coordinator.create_document_version(filename="a.txt", content=b"new", operation_id="op-1")
    record = journal.get("op-1")
    assert record["vector_ids"] == ["doc-1:v1:c0"]
    assert record["graph_evidence_keys"] == ["e:doc-1:1"]


@pytest.mark.asyncio
async def test_reconcile_does_not_process_unexpired_running_operation(setup):
    coordinator, _, journal, _, _, _, _ = setup
    journal.records["op-x"] = {
        "operation_id": "op-x", "operation_type": "update", "document_id": "doc-x", "version": 2,
        "previous_current_version": None, "status": "running", "completed_steps": [], "compensation_steps": [],
    }
    journal.lock_rows["doc-x"] = {"document_id": "doc-x", "operation_id": "op-x", "expires_at": utcnow() + timedelta(seconds=60)}
    journal.locks.rows["doc-x"] = journal.lock_rows["doc-x"]
    assert await coordinator.reconcile_incomplete_operations() == []
