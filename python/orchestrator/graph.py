"""Checkpointed QA workflow.

Formal document creation, update, and deletion are intentionally outside this
LangGraph module. They use the versioned FastAPI document endpoints and
``DocumentUpdateCoordinator`` so Mongo, Chroma, and Neo4j participate in the
same S3 Saga.
"""

from __future__ import annotations

import asyncio
import logging
from functools import wraps
from typing import Annotated, Any

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from agents.qa_agent import QAAgent, QAResult
from providers.chat import ChatProvider
from services.knowledge_graph import KnowledgeGraphService
from services.memory_service import MemoryService
from services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


class IdempotentNode:
    """Apply bounded retry and request-id idempotency to a QA workflow node."""

    def __init__(self, node_name: str, max_retries: int = 3, timeout_seconds: float = 60.0):
        self.node_name = node_name
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds

    def __call__(self, func):
        node_name = self.node_name
        max_retries = self.max_retries
        timeout_seconds = self.timeout_seconds

        @wraps(func)
        async def wrapper(state: dict) -> dict:
            request_id = state.get("request_id", "")
            executed_key = f"_executed_{node_name}"
            executed_set = state.get(executed_key, set()) or set()
            if request_id and request_id in executed_set:
                logger.info("[Idempotent] node %s already ran for request_id=%s", node_name, request_id)
                return {}

            last_error: str | None = None
            for attempt in range(1, max_retries + 1):
                try:
                    result = await asyncio.wait_for(func(state), timeout=timeout_seconds)
                    if request_id:
                        result[executed_key] = set(executed_set) | {request_id}
                    return result
                except asyncio.TimeoutError:
                    last_error = f"node {node_name} timed out on attempt {attempt}"
                except Exception as exc:
                    last_error = f"node {node_name} failed on attempt {attempt}: {type(exc).__name__}"
                logger.warning("%s", last_error)
                if attempt < max_retries:
                    await asyncio.sleep(min(2**attempt, 10))
            raise RuntimeError(f"node {node_name} exhausted retries: {last_error}")

        return wrapper


class QAState(dict):
    """State owned by the checkpointed question-answering workflow."""

    question: str
    session_id: str | None
    user_id: str | None
    result: QAResult | None
    messages: Annotated[list, add_messages]
    request_id: str
    _executed_answer: set


def build_knowledge_graph_workflow(
    chat_provider: ChatProvider,
    vector_store: VectorStoreService | None = None,
    knowledge_graph: KnowledgeGraphService | None = None,
    memory_service: MemoryService | None = None,
    checkpointer: Any = None,
) -> dict[str, Any]:
    """Build the QA workflow; S3 document mutations use the Coordinator only."""

    qa_agent = QAAgent(
        chat_provider=chat_provider,
        vector_store=vector_store,
        knowledge_graph=knowledge_graph,
        memory_service=memory_service,
    )
    return {"qa": _build_qa_graph(qa_agent, checkpointer=checkpointer)}


def _build_qa_graph(qa_agent: QAAgent, checkpointer: Any = None) -> Any:
    @IdempotentNode("answer", max_retries=2, timeout_seconds=180)
    async def process_question(state: dict) -> dict:
        question = state.get("question", "")
        session_id = state.get("session_id")
        user_id = state.get("user_id")
        if session_id or user_id:
            qa_agent.set_session_context(session_id=session_id, user_id=user_id)
        return {"result": await qa_agent.answer(question)}

    graph = StateGraph(QAState)
    graph.add_node("answer", process_question)
    graph.set_entry_point("answer")
    graph.add_edge("answer", END)
    return graph.compile(checkpointer=checkpointer)
