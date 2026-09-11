"""Offline-safe, deterministic Vector RAG versus GraphRAG evaluation helpers."""

from __future__ import annotations

import asyncio
import csv
import json
import math
import os
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from uuid import NAMESPACE_URL, uuid5

from agents.qa_agent import EvaluationQueryPlan, QAAgent, RetrievalMode
from services.evaluation_scope import EvaluationScope, require_verified_scope


_ABSOLUTE_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|[\\/]{1,2})")
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")
_ABSTENTION_MARKERS = ("无法回答", "无法确定", "信息不足", "未提供", "未找到", "没有相关")


@dataclass(frozen=True)
class EvaluationCase:
    question_id: str
    question: str
    category: str
    required_keywords: tuple[str, ...]
    expected_sources: tuple[str, ...]
    forbidden_keywords: tuple[str, ...] = ()
    requires_abstention: bool = False
    graph_advantage_expected: bool = False
    offline_entities: tuple[str, ...] = ()
    offline_answer: str = ""


# The 12 cases intentionally have overlapping categories (for example,
# constraints can also require multi-hop traversal); the case count remains 12.
EVALUATION_CASES: tuple[EvaluationCase, ...] = (
    EvaluationCase("Q01", "陈航在什么部门担任什么职位？", "single_hop", ("数据平台部", "数据工程师"), ("s4-eval-org.txt",), offline_entities=("陈航",), offline_answer="陈航是数据平台部的数据工程师。"),
    EvaluationCase("Q02", "北极星检索项目由谁负责，使用什么技术？", "single_hop", ("陈航", "Chroma"), ("s4-eval-projects.txt",), offline_entities=("北极星", "陈航"), offline_answer="北极星检索项目由陈航负责，使用 Chroma。"),
    EvaluationCase("Q03", "北极星依赖的服务由谁负责？", "multi_hop", ("Atlas", "赵启"), ("s4-eval-projects.txt",), graph_advantage_expected=True, offline_entities=("北极星", "Atlas"), offline_answer="北极星依赖 Atlas 事件服务，该服务由赵启负责。"),
    EvaluationCase("Q04", "天枢知识平台的共同负责人分别来自哪个部门？", "multi_hop", ("林岚", "数据平台部", "赵启", "应用架构部"), ("s4-eval-org.txt", "s4-eval-collaboration.txt"), graph_advantage_expected=True, offline_entities=("天枢", "林岚", "赵启"), offline_answer="天枢由林岚和赵启共同负责；林岚来自数据平台部，赵启来自应用架构部。"),
    EvaluationCase("Q05", "北极星如何与天枢知识平台关联？", "multi_hop", ("北极星", "检索索引", "天枢"), ("s4-eval-collaboration.txt",), graph_advantage_expected=True, offline_entities=("北极星", "天枢"), offline_answer="北极星检索项目为天枢知识平台提供检索索引。"),
    EvaluationCase("Q06", "哪个项目同时使用 Neo4j 和 Chroma？", "constraint", ("天枢", "Neo4j", "Chroma"), ("s4-eval-collaboration.txt",), graph_advantage_expected=True, offline_entities=("天枢",), offline_answer="天枢知识平台同时使用 Neo4j 和 Chroma。"),
    EvaluationCase("Q07", "谁同时满足数据平台部成员和项目负责人？", "constraint_multi_hop", ("陈航", "北极星"), ("s4-eval-org.txt", "s4-eval-projects.txt"), graph_advantage_expected=True, offline_entities=("陈航", "数据平台部", "北极星"), offline_answer="陈航属于数据平台部，并负责北极星检索项目。"),
    EvaluationCase("Q08", "赵启负责的服务与哪个项目存在依赖关系？", "multi_hop", ("Atlas", "北极星"), ("s4-eval-projects.txt",), graph_advantage_expected=True, offline_entities=("赵启", "Atlas"), offline_answer="赵启负责 Atlas 事件服务，北极星检索项目依赖该服务。"),
    EvaluationCase("Q09", "晨星报表项目使用哪些技术？", "distractor_fact", ("PostgreSQL", "Elasticsearch"), ("s4-eval-distractors.txt",), offline_entities=("晨星"), offline_answer="晨星报表项目使用 PostgreSQL 和 Elasticsearch。"),
    EvaluationCase("Q10", "周宁是否负责北极星检索项目？", "distractor_negation", ("否", "周宁", "晨星"), ("s4-eval-org.txt", "s4-eval-distractors.txt"), forbidden_keywords=("周宁负责北极星",), graph_advantage_expected=True, offline_entities=("周宁", "北极星"), offline_answer="否，周宁不负责北极星；周宁参与晨星报表的数据分析。"),
    EvaluationCase("Q11", "南斗项目的负责人是谁？", "unanswerable", (), (), requires_abstention=True, offline_answer="资料未提供南斗项目负责人，无法回答。"),
    EvaluationCase("Q12", "星河云智公司的 CEO 是谁？", "unanswerable", (), (), requires_abstention=True, offline_answer="现有资料未提供星河云智 CEO 信息，无法确定。"),
)


# Complete reviewed S4 corpus. The authorized real runner sends no document
# text other than these strings.
SYNTHETIC_DOCUMENTS: tuple[tuple[str, str, str], ...] = (
    ("org", "s4-eval-org.txt", "星河云智公司有数据平台部、应用架构部和运营部。林岚是数据平台部负责人；陈航是数据工程师，隶属数据平台部；赵启是应用架构部负责人；周宁是运营部分析师。"),
    ("projects", "s4-eval-projects.txt", "陈航负责北极星检索项目。北极星使用 Chroma，依赖 Atlas 事件服务。赵启负责 Atlas 事件服务。"),
    ("collaboration", "s4-eval-collaboration.txt", "数据平台部与应用架构部共同交付天枢知识平台。林岚和赵启共同负责天枢。天枢使用 Neo4j 与 Chroma；北极星为天枢提供检索索引。"),
    ("distractors", "s4-eval-distractors.txt", "晨星报表项目使用 PostgreSQL 与 Elasticsearch。周宁参与晨星报表的数据分析；晨星与北极星、天枢不存在依赖关系。"),
)


_OFFLINE_SOURCES = frozenset(source for case in EVALUATION_CASES for source in case.expected_sources)


def offline_document_id(source: str) -> str:
    """Return a deterministic UUID-shaped fake upload identity for one fixture source."""
    return str(uuid5(NAMESPACE_URL, f"agentknowledgehub-s4/{source}"))


def safe_source(value: object) -> str:
    """Return a display-only source name and never retain an absolute path."""
    text = str(value or "").replace("\\", "/")
    return text.rsplit("/", 1)[-1]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _nearest_rank(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil((percentile / 100) * len(ordered)) - 1)
    return round(ordered[index], 3)


def _counter_value(owner: Any, attribute: str) -> int | None:
    value = getattr(owner, attribute, None)
    return int(value) if isinstance(value, int) else None


class DeterministicScorer:
    """Score answers from fixtures; no model judge is involved."""

    @staticmethod
    def score(
        case: EvaluationCase,
        *,
        answer: str,
        sources: list[dict[str, Any]],
        mode: RetrievalMode,
        graph_context_count: int,
    ) -> dict[str, Any]:
        normalized_answer = answer.casefold()
        matched = [keyword for keyword in case.required_keywords if keyword.casefold() in normalized_answer]
        forbidden = [keyword for keyword in case.forbidden_keywords if keyword.casefold() in normalized_answer]
        source_names = {str(source.get("source", "")) for source in sources}
        source_hit = (
            not sources if case.requires_abstention else set(case.expected_sources).issubset(source_names)
        )
        abstention = any(marker in answer for marker in _ABSTENTION_MARKERS)
        abstention_correct = abstention if case.requires_abstention else None
        facts_correct = len(matched) == len(case.required_keywords) and not forbidden
        graph_participated = graph_context_count > 0
        graph_participation_pass = (
            mode is not RetrievalMode.GRAPH_RAG
            or case.requires_abstention
            or graph_participated
        )
        correct = facts_correct and source_hit and graph_participation_pass
        if case.requires_abstention:
            correct = bool(abstention_correct and source_hit and graph_participation_pass)
        return {
            "required_keyword_hits": matched,
            "required_keyword_total": len(case.required_keywords),
            "keyword_fact_score": round(len(matched) / len(case.required_keywords), 3)
            if case.required_keywords else 1.0,
            "forbidden_keyword_hits": forbidden,
            "source_hit": source_hit,
            "abstention_correct": abstention_correct,
            "graph_participated": graph_participated,
            "graph_participation_pass": graph_participation_pass,
            "question_correct": correct,
        }


class RAGEvaluationRunner:
    """Run one or both modes from shared plans and write safe local reports."""

    def __init__(
        self,
        agent: QAAgent,
        cases: Iterable[EvaluationCase] = EVALUATION_CASES,
        *,
        scope: EvaluationScope | None = None,
        offline: bool = True,
    ) -> None:
        if agent.memory_service is not None:
            raise ValueError("Evaluation runner requires QAAgent(memory_service=None)")
        self.agent = agent
        self.cases = tuple(cases)
        self.scope = require_verified_scope(scope)
        self.offline = offline

    async def run(
        self, *, run_id: str, modes: Iterable[RetrievalMode | str], output_root: str | Path
    ) -> dict[str, Any]:
        if not _SAFE_RUN_ID.fullmatch(run_id):
            raise ValueError("run_id must be a safe identifier")
        selected_modes = tuple(RetrievalMode(mode) for mode in modes)
        if not selected_modes:
            raise ValueError("at least one evaluation mode is required")
        results: list[dict[str, Any]] = []
        output_dir = Path(output_root) / run_id
        payload: dict[str, Any] = {
            "run_id": run_id,
            "offline": self.offline,
            "scope": {
                "run_id": self.scope.run_id,
                "allowed_document_ids_count": len(self.scope.allowed_document_ids),
                "scope_verified": self.scope.is_verified(),
            },
            "modes": [mode.value for mode in selected_modes],
            "results": results,
            "summary": {},
            "notice": "Fake-only offline output validates framework behavior; it is not a real model comparison."
            if self.offline else "Authorized S4 synthetic-only real evaluation. Results are not a general production benchmark.",
        }
        for case in self.cases:
            plan_before = _counter_value(self.agent.llm, "call_count")
            plan = await self.agent.build_evaluation_query_plan(
                case.question, run_id=run_id, question_id=case.question_id, top_k=5
            )
            plan_after = _counter_value(self.agent.llm, "call_count")
            for mode in selected_modes:
                results.append(await self._run_case(case, plan, mode, plan_before, plan_after))
                payload["summary"] = self._summarize(results)
                self.write_reports(output_dir, payload)
        return payload

    async def _run_case(
        self,
        case: EvaluationCase,
        plan: EvaluationQueryPlan,
        mode: RetrievalMode,
        plan_before: int | None,
        plan_after: int | None,
    ) -> dict[str, Any]:
        chat_before = _counter_value(self.agent.llm, "call_count")
        vector_before = _counter_value(self.agent.vector_store, "search_calls")
        embedding_before = _counter_value(getattr(self.agent.vector_store, "embeddings", None), "query_calls")
        graph_before = _counter_value(self.agent.knowledge_graph, "calls")
        started = time.monotonic()
        try:
            result = await self.agent.answer_with_evaluation_plan(plan, mode, scope=self.scope)
            elapsed_ms = round((time.monotonic() - started) * 1000, 3)
            sources = self._safe_sources(result.contexts)
            vector_count = sum(context.retrieval_type == "vector" for context in result.contexts)
            graph_count = sum(context.retrieval_type == "graph" for context in result.contexts)
            score = DeterministicScorer.score(
                case, answer=result.answer, sources=sources, mode=mode, graph_context_count=graph_count
            )
            return {
                "run_id": plan.run_id,
                "mode": mode.value,
                "question_id": case.question_id,
                "category": case.category,
                "success": True,
                "http_status": None,
                "latency_ms": elapsed_ms,
                "answer": result.answer,
                "sources": sources,
                "vector_context_count": vector_count,
                "graph_context_count": graph_count,
                "allowed_document_ids_count": result.scope_diagnostics["allowed_document_ids_count"],
                "vector_scope_rejected_count": result.scope_diagnostics["vector_scope_rejected_count"],
                "graph_scope_rejected_count": result.scope_diagnostics["graph_scope_rejected_count"],
                "scope_verified": result.scope_diagnostics["scope_verified"],
                "model_call_counts": self._call_counts(
                    plan_before, plan_after, chat_before, vector_before, embedding_before, graph_before
                ),
                "deterministic_score": score,
                "failure_summary": None,
            }
        except Exception as exc:
            return {
                "run_id": plan.run_id, "mode": mode.value, "question_id": case.question_id,
                "category": case.category, "success": False, "http_status": None,
                "latency_ms": round((time.monotonic() - started) * 1000, 3), "answer": "",
                "sources": [], "vector_context_count": 0, "graph_context_count": 0,
                "allowed_document_ids_count": len(self.scope.allowed_document_ids),
                "vector_scope_rejected_count": 0, "graph_scope_rejected_count": 0,
                "scope_verified": self.scope.is_verified(),
                "model_call_counts": self._call_counts(plan_before, plan_after, chat_before, vector_before, embedding_before, graph_before),
                "deterministic_score": None, "failure_summary": type(exc).__name__,
            }

    def _call_counts(
        self, plan_before: int | None, plan_after: int | None, chat_before: int | None,
        vector_before: int | None, embedding_before: int | None, graph_before: int | None,
    ) -> dict[str, int | None]:
        current_chat = _counter_value(self.agent.llm, "call_count")
        current_vector = _counter_value(self.agent.vector_store, "search_calls")
        current_embedding = _counter_value(getattr(self.agent.vector_store, "embeddings", None), "query_calls")
        current_graph = _counter_value(self.agent.knowledge_graph, "calls")
        return {
            "shared_preprocess_chat": None if plan_before is None or plan_after is None else plan_after - plan_before,
            "answer_chat": None if chat_before is None or current_chat is None else current_chat - chat_before,
            "vector_search": None if vector_before is None or current_vector is None else current_vector - vector_before,
            "embedding_query": None if embedding_before is None or current_embedding is None else current_embedding - embedding_before,
            "graph_calls": None if graph_before is None or current_graph is None else current_graph - graph_before,
        }

    @staticmethod
    def _safe_sources(contexts: Iterable[Any]) -> list[dict[str, Any]]:
        return [
            {
                "source": safe_source(getattr(context, "source", "")),
                "document_id": str(getattr(context, "metadata", {}).get("document_id") or "") or None,
                "document_version": getattr(context, "metadata", {}).get("document_version"),
                "retrieval_type": str(getattr(context, "retrieval_type", "")),
                "score": round(float(getattr(context, "score", 0.0)), 4),
            }
            for context in contexts
        ]

    @staticmethod
    def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for mode in sorted({result["mode"] for result in results}):
            rows = [result for result in results if result["mode"] == mode]
            successful = [row for row in rows if row["success"]]
            correct = [row for row in successful if row.get("deterministic_score", {}).get("question_correct")]
            multi_hop = [row for row in successful if "multi_hop" in row["category"]]
            abstentions = [row for row in successful if row["category"] == "unanswerable"]
            graph_rows = [row for row in successful if row["graph_context_count"] > 0]
            latencies = [float(row["latency_ms"]) for row in successful]
            summary[mode] = {
                "questions": len(rows),
                "success_rate": round(len(successful) / len(rows), 3) if rows else 0.0,
                "question_accuracy": round(len(correct) / len(rows), 3) if rows else 0.0,
                "keyword_fact_hit_rate": round(
                    statistics.fmean(row["deterministic_score"]["keyword_fact_score"] for row in successful), 3
                ) if successful else 0.0,
                "source_hit_rate": round(sum(bool(row["deterministic_score"]["source_hit"]) for row in successful) / len(rows), 3) if rows else 0.0,
                "multi_hop_accuracy": round(sum(bool(row["deterministic_score"]["question_correct"]) for row in multi_hop) / len(multi_hop), 3) if multi_hop else None,
                "abstention_accuracy": round(sum(bool(row["deterministic_score"]["abstention_correct"]) for row in abstentions) / len(abstentions), 3) if abstentions else None,
                "graph_participation_rate": round(len(graph_rows) / len(rows), 3) if mode == RetrievalMode.GRAPH_RAG.value and rows else None,
                "latency_avg_ms": round(statistics.fmean(latencies), 3) if latencies else None,
                "latency_p50_ms": _nearest_rank(latencies, 50),
                "latency_p95_ms": _nearest_rank(latencies, 95),
            }
        return summary

    @staticmethod
    def write_reports(output_dir: Path, payload: dict[str, Any]) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(output_dir / "results.json", payload)
        fieldnames = [
            "run_id", "mode", "question_id", "category", "success", "http_status", "latency_ms",
            "answer", "sources", "vector_context_count", "graph_context_count", "allowed_document_ids_count",
            "vector_scope_rejected_count", "graph_scope_rejected_count", "scope_verified", "model_call_counts",
            "deterministic_score", "failure_summary",
        ]
        temporary = output_dir / "results.csv.tmp"
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for result in payload["results"]:
                row = dict(result)
                for name in ("sources", "model_call_counts", "deterministic_score"):
                    row[name] = json.dumps(row[name], ensure_ascii=False, sort_keys=True)
                writer.writerow(row)
        os.replace(temporary, output_dir / "results.csv")
        lines = ["# Offline RAG evaluation", "", payload["notice"], "", "| Mode | Questions | Success | Accuracy | Graph participation |", "|---|---:|---:|---:|---:|"]
        for mode, values in payload["summary"].items():
            lines.append(f"| {mode} | {values['questions']} | {values['success_rate']:.3f} | {values['question_accuracy']:.3f} | {values['graph_participation_rate'] if values['graph_participation_rate'] is not None else '-'} |")
        temporary_md = output_dir / "summary.md.tmp"
        temporary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(temporary_md, output_dir / "summary.md")


class _OfflineMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class OfflineChatProvider:
    """Deterministic fake Chat provider; it has no network-capable dependency."""

    provider_name = "offline-fake"
    model_name = "offline-fake"

    def __init__(self, cases: Iterable[EvaluationCase]) -> None:
        self.cases = {case.question: case for case in cases}
        self.call_count = 0
        self.final_prompts: list[str] = []

    async def ainvoke(self, messages: list[Any], **_: Any) -> _OfflineMessage:
        self.call_count += 1
        prompt = "\n".join(str(getattr(message, "content", "")) for message in messages)
        question = next((item for item in self.cases if item in prompt), "")
        case = self.cases.get(question)
        if "查询意图分类器" in prompt:
            return _OfflineMessage("factoid")
        if "查询改写专家" in prompt and case:
            return _OfflineMessage(json.dumps({"queries": [case.question], "entities": list(case.offline_entities)}, ensure_ascii=False))
        self.final_prompts.append(prompt)
        return _OfflineMessage(case.offline_answer if case else "资料未提供，无法回答。")


class OfflineVectorStore:
    """In-memory fake vector interface used exclusively by ``--offline``."""

    def __init__(self, cases: Iterable[EvaluationCase], embeddings: "OfflineEmbeddingProvider") -> None:
        self.by_question = {case.question: case for case in cases}
        self.embeddings = embeddings
        self.search_calls = 0

    async def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        allowed_document_ids: frozenset[str] | None = None,
        scope_diagnostics: dict[str, int] | None = None,
    ) -> list[tuple[dict[str, Any], float]]:
        del allowed_document_ids, scope_diagnostics
        self.search_calls += 1
        await self.embeddings.aembed_query(query)
        case = self.by_question.get(query)
        if not case or case.requires_abstention:
            return []
        result = [
            ({"content": f"synthetic evidence for {source}", "source": source, "metadata": {"document_id": offline_document_id(source), "document_version": 1}}, 0.9)
            for source in case.expected_sources[:top_k]
        ]
        # Deliberately return foreign and legacy candidates. QA's second scope
        # check proves that a misbehaving store cannot widen an evaluation run.
        result.extend([
            ({"content": "foreign", "source": "foreign.txt", "metadata": {"document_id": str(uuid5(NAMESPACE_URL, "foreign")), "document_version": 1}}, 0.1),
            ({"content": "legacy", "source": "legacy.txt", "metadata": {"doc_id": "old"}}, 0.0),
        ])
        return result


class OfflineEmbeddingProvider:
    """Deterministic in-memory embedding stub used only by the fake VectorStore."""

    dimensions = 3

    def __init__(self) -> None:
        self.query_calls = 0

    async def aembed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        # The vector has no semantic role in this fake store; it makes the
        # offline dependency boundary explicit without using a network client.
        return [float(len(text) % 7), 0.0, 1.0]


class OfflineKnowledgeGraph:
    """In-memory fake graph interface; no driver, URI, or database is created."""

    def __init__(self, cases: Iterable[EvaluationCase]) -> None:
        self.entity_source = {
            entity: case.expected_sources[0]
            for case in cases if case.expected_sources
            for entity in case.offline_entities
        }
        self.calls = 0

    @staticmethod
    def safe_source(value: object) -> str:
        return safe_source(value)

    async def get_neighbors(
        self,
        entity_name: str,
        hops: int = 2,
        *,
        allowed_document_ids: frozenset[str] | None = None,
        scope_diagnostics: dict[str, int] | None = None,
    ) -> list[dict[str, Any]]:
        del hops, allowed_document_ids, scope_diagnostics
        self.calls += 1
        source = self.entity_source.get(entity_name)
        if not source:
            return []
        return [{
            "source": entity_name, "relations": ["RELATED_TO"], "target": "synthetic",
            "target_type": "Concept", "target_desc": "offline evidence",
            "document_id": offline_document_id(source), "document_version": 1,
            "provenance_source": source,
        }, {
            "source": entity_name, "relations": ["RELATED_TO"], "target": "foreign",
            "target_type": "Concept", "target_desc": "foreign evidence",
            "document_id": str(uuid5(NAMESPACE_URL, "foreign-graph")), "document_version": 1,
            "provenance_source": "foreign.txt",
        }, {
            "source": entity_name, "relations": ["RELATED_TO"], "target": "legacy",
            "target_type": "Concept", "target_desc": "legacy evidence",
            "provenance_source": "legacy.txt",
        }]


def build_offline_runner(run_id: str = "offline-fixture") -> RAGEvaluationRunner:
    """Construct a fake-only runner without importing configuration or providers."""
    cases = EVALUATION_CASES
    embeddings = OfflineEmbeddingProvider()
    scope = EvaluationScope.from_uploaded_document_ids(
        run_id, (offline_document_id(source) for source in _OFFLINE_SOURCES)
    )
    return RAGEvaluationRunner(
        QAAgent(
            chat_provider=OfflineChatProvider(cases),
            vector_store=OfflineVectorStore(cases, embeddings),
            knowledge_graph=OfflineKnowledgeGraph(cases),
            memory_service=None,
        ),
        cases,
        scope=scope,
    )


def run_offline_sync(run_id: str, modes: Iterable[RetrievalMode | str], output_root: str | Path) -> dict[str, Any]:
    return asyncio.run(build_offline_runner(run_id).run(run_id=run_id, modes=modes, output_root=output_root))
