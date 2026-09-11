"""Offline guardrails for the retired pre-S3 update path."""

from __future__ import annotations

import inspect

import pytest

from agents.knowledge_update_agent import DeprecatedUpdatePathError, KnowledgeUpdateAgent
from orchestrator.graph import build_knowledge_graph_workflow
import api.main as api


@pytest.mark.asyncio
async def test_retired_agent_rejects_change_without_touching_injected_stores():
    """The compatibility shell cannot revive delete-and-rebuild behavior."""

    vector_store = object()
    knowledge_graph = object()
    agent = KnowledgeUpdateAgent(vector_store=vector_store, knowledge_graph=knowledge_graph)

    with pytest.raises(DeprecatedUpdatePathError, match="DocumentUpdateCoordinator"):
        await agent.process_change(object())

    assert not hasattr(agent, "vector_store")
    assert not hasattr(agent, "knowledge_graph")


def test_retired_agent_module_has_no_direct_storage_write_code():
    source = inspect.getsource(__import__("agents.knowledge_update_agent", fromlist=["*"]))
    assert "delete_by_doc_id" not in source
    assert "delete_by_source" not in source
    assert "VectorStoreService" not in source
    assert "KnowledgeGraphService" not in source


def test_workflow_factory_exposes_qa_only_not_legacy_mutation_graphs():
    workflows = build_knowledge_graph_workflow(chat_provider=object())
    assert set(workflows) == {"qa"}


def test_formal_document_api_module_does_not_import_legacy_update_agent():
    assert "KnowledgeUpdateAgent" not in inspect.getsource(api)

