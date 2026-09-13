"""Offline contracts for the embedding-index migration command."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def load_tool():
    path = Path(__file__).resolve().parents[2] / "scripts" / "migrate-embedding-index.py"
    spec = importlib.util.spec_from_file_location("embedding_migration_tool", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migration_state_contains_only_safe_identifiers():
    tool = load_tool()
    state = tool.make_state("source", "target", "run-1", ["id-1"], 10)
    assert state["planned_vector_ids"] == ["id-1"]
    assert "documents" not in state and "embeddings" not in state
    assert state["status"] == "planned"


def test_metadata_summary_omits_unapproved_fields():
    tool = load_tool()
    metadata = tool.safe_metadata({"embedding_model": "model", "unsafe": "secret"})
    assert metadata == {"embedding_model": "model"}


def test_error_categories_do_not_persist_provider_messages():
    tool = load_tool()
    assert tool.safe_error_category(TimeoutError("private provider response")) == "provider_timeout"
