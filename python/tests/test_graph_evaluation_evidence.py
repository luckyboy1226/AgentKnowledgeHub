"""Fake-only S4.2b contracts for provenance-complete graph evaluation context."""

from __future__ import annotations

import pytest

from agents.qa_agent import RetrievedContext, RetrievalMode, QAAgent
from services.rag_evaluation import build_offline_runner


def _edge(
    subject: str,
    predicate: str,
    object_: str,
    *,
    direction: str = "forward",
    document_id: str = "0f04b95f-4e95-444b-a6e4-7c403a712303",
    version: int = 1,
    source: str = "s4-eval-projects.txt",
    evidence_key: str = "evidence-1",
):
    return {
        "subject": subject, "predicate": predicate, "object": object_, "direction": direction,
        "document_id": document_id, "document_version": version, "source": source,
        "evidence_key": evidence_key,
    }


def test_scoped_graph_evidence_formats_a_directed_single_hop_with_provenance():
    evidence = [_edge("北极星检索项目", "DEPENDS_ON", "Atlas 事件服务")]
    text = QAAgent._format_scoped_graph_evidence(evidence)

    assert "北极星检索项目 --DEPENDS_ON--> Atlas 事件服务" in text
    assert "方向: forward" in text
    assert "来源: s4-eval-projects.txt, v1" in text
    assert "evidence: evidence-1" in text


def test_scoped_graph_evidence_keeps_each_hop_and_intermediate_entity():
    evidence = [
        _edge("北极星检索项目", "DEPENDS_ON", "Atlas 事件服务", evidence_key="depends"),
        _edge("赵启", "RESPONSIBLE_FOR", "Atlas 事件服务", direction="reverse", evidence_key="owner"),
    ]
    text = QAAgent._format_scoped_graph_evidence(evidence)

    assert "北极星检索项目 --DEPENDS_ON--> Atlas 事件服务" in text
    assert "赵启 --RESPONSIBLE_FOR--> Atlas 事件服务" in text
    assert text.count("Atlas 事件服务") == 2
    assert "方向: reverse" in text


def test_provides_index_and_depends_on_are_never_rewritten_into_each_other():
    provides = QAAgent._format_scoped_graph_evidence([
        _edge("北极星", "PROVIDES_INDEX", "天枢", evidence_key="provides"),
    ])
    depends = QAAgent._format_scoped_graph_evidence([
        _edge("北极星", "DEPENDS_ON", "Atlas", evidence_key="depends"),
    ])

    assert "北极星 --PROVIDES_INDEX--> 天枢" in provides
    assert "DEPENDS_ON" not in provides
    assert "北极星 --DEPENDS_ON--> Atlas" in depends
    assert "PROVIDES_INDEX" not in depends


@pytest.mark.parametrize("record", [
    {},
    {"evidence_edges": []},
    {"evidence_edges": [{"subject": "A"}]},
    {"evidence_edges": [_edge("A", "OWNS", "B", document_id="foreign-id")]},
    {"evidence_edges": [_edge("A", "OWNS", "B", version=0)]},
])
def test_scoped_graph_evidence_fails_closed_without_complete_allowed_provenance(record):
    allowed = frozenset({"0f04b95f-4e95-444b-a6e4-7c403a712303"})
    assert QAAgent._scoped_graph_evidence(record, allowed) is None


def test_evaluation_rerank_removes_the_unconditional_graph_boost_only_for_scoped_runs():
    vector = RetrievedContext("vector", "v.txt", 0.9, "vector")
    graph = RetrievedContext("graph", "g.txt", 0.8, "graph")

    ranked = QAAgent._hybrid_rerank([vector, graph], evaluation_scoped=True)

    assert ranked[0].retrieval_type == "vector"
    assert graph.score == 0.8


@pytest.mark.asyncio
async def test_scoped_graph_context_reaches_prompt_as_directed_evidence_only():
    runner = build_offline_runner()
    case = next(case for case in runner.cases if case.question_id == "Q03")
    plan = await runner.agent.build_evaluation_query_plan(
        case.question, run_id="evidence", question_id=case.question_id
    )

    result = await runner.agent.answer_with_evaluation_plan(
        plan, RetrievalMode.GRAPH_RAG, scope=runner.scope
    )

    graph_contexts = [context for context in result.contexts if context.retrieval_type == "graph"]
    assert graph_contexts
    assert all("图谱证据（逐边原样陈述" in context.content for context in graph_contexts)
    assert all(context.metadata["graph_evidence"] for context in graph_contexts)
    assert "'relations':" not in runner.agent.llm.final_prompts[-1]
    assert "foreign evidence" not in runner.agent.llm.final_prompts[-1]
    assert "legacy evidence" not in runner.agent.llm.final_prompts[-1]


class _EmptyScopedGraph:
    def __init__(self) -> None:
        self.calls = 0

    @staticmethod
    def safe_source(value: object) -> str:
        return str(value)

    async def get_neighbors(self, *_args, **_kwargs):
        self.calls += 1
        return []


@pytest.mark.asyncio
async def test_missing_graph_evidence_is_allowed_to_remain_empty_without_fallback():
    runner = build_offline_runner()
    empty_graph = _EmptyScopedGraph()
    runner.agent.knowledge_graph = empty_graph
    case = next(case for case in runner.cases if case.question_id == "Q03")
    plan = await runner.agent.build_evaluation_query_plan(
        case.question, run_id="empty-evidence", question_id=case.question_id
    )

    result = await runner.agent.answer_with_evaluation_plan(
        plan, RetrievalMode.GRAPH_RAG, scope=runner.scope
    )

    assert empty_graph.calls == len(plan.entities)
    assert not [context for context in result.contexts if context.retrieval_type == "graph"]
    assert "图谱证据（逐边原样陈述" not in runner.agent.llm.final_prompts[-1]
