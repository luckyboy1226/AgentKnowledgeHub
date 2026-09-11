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
from services.relation_semantics import RELATION_SEMANTICS_VERSION, canonical_predicate, is_canonical_predicate


_ABSOLUTE_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|[\\/]{1,2})")
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")
_ABSTENTION_MARKERS = ("无法回答", "无法确定", "信息不足", "未提供", "未找到", "没有相关")
_ABSTENTION_MARKERS_V2 = _ABSTENTION_MARKERS + ("未提及", "未找到相关", "没有相关信息")
_NEGATION_MARKERS_V2 = ("否", "不负责", "并非", "不是负责人", "没有负责", "并不负责", "无法得出")
SCORER_V2 = "deterministic-v2"


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
    # v2 keeps finite, case-owned rules rather than trying to infer answer
    # semantics with another model.  They are intentionally only used by the
    # evaluation scorer, never by production retrieval or generation.
    negation_target_terms: tuple[str, ...] = ()
    positive_claim_patterns: tuple[str, ...] = ()
    abstention_positive_claim_patterns: tuple[str, ...] = ()
    expected_answer: str = ""
    expected_relation_path: tuple[tuple[str, str, str, str], ...] = ()


@dataclass(frozen=True)
class BenchmarkDocument:
    """Immutable benchmark document loaded from a reviewed fixture."""

    document_id: str
    filename: str
    content: str


@dataclass(frozen=True)
class EvaluationFixture:
    """Validated, fixture-owned corpus and question set for one evaluation."""

    name: str
    version: str
    documents: tuple[BenchmarkDocument, ...]
    cases: tuple[EvaluationCase, ...]
    root: Path


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
    EvaluationCase("Q10", "周宁是否负责北极星检索项目？", "distractor_negation", ("否", "周宁", "晨星"), ("s4-eval-org.txt", "s4-eval-distractors.txt"), forbidden_keywords=("周宁负责北极星",), graph_advantage_expected=True, offline_entities=("周宁", "北极星"), offline_answer="否，周宁不负责北极星；周宁参与晨星报表的数据分析。", negation_target_terms=("周宁", "北极星"), positive_claim_patterns=(r"周宁\s*(?:确实|正是|就是|担任|作为|负责(?:人)?)\s*北极星", r"北极星(?:检索项目)?\s*(?:由|负责人是|负责人为)\s*周宁")),
    EvaluationCase("Q11", "南斗项目的负责人是谁？", "unanswerable", (), (), requires_abstention=True, offline_answer="资料未提供南斗项目负责人，无法回答。", abstention_positive_claim_patterns=(r"南斗(?:项目)?(?:的)?负责人\s*(?:是|为)\s*(?!谁|什么|未知|不详|未提供|无法)",)),
    EvaluationCase("Q12", "星河云智公司的 CEO 是谁？", "unanswerable", (), (), requires_abstention=True, offline_answer="现有资料未提供星河云智 CEO 信息，无法确定。", abstention_positive_claim_patterns=(r"CEO\s*(?:是|为)\s*(?!谁|什么|未知|不详|未提供|无法)",)),
)


# Complete reviewed S4 corpus. The authorized real runner sends no document
# text other than these strings.
SYNTHETIC_DOCUMENTS: tuple[tuple[str, str, str], ...] = (
    ("org", "s4-eval-org.txt", "星河云智公司有数据平台部、应用架构部和运营部。林岚是数据平台部负责人；陈航是数据工程师，隶属数据平台部；赵启是应用架构部负责人；周宁是运营部分析师。"),
    ("projects", "s4-eval-projects.txt", "陈航负责北极星检索项目。北极星使用 Chroma，依赖 Atlas 事件服务。赵启负责 Atlas 事件服务。"),
    ("collaboration", "s4-eval-collaboration.txt", "数据平台部与应用架构部共同交付天枢知识平台。林岚和赵启共同负责天枢。天枢使用 Neo4j 与 Chroma；北极星为天枢提供检索索引。"),
    ("distractors", "s4-eval-distractors.txt", "晨星报表项目使用 PostgreSQL 与 Elasticsearch。周宁参与晨星报表的数据分析；晨星与北极星、天枢不存在依赖关系。"),
)


def _fixture_error(message: str) -> ValueError:
    return ValueError(f"Invalid evaluation fixture: {message}")


def _relation_path(value: object) -> tuple[tuple[str, str, str, str], ...]:
    """Normalize fixture relation tuples; missing direction is explicitly forward."""
    if not isinstance(value, list):
        raise _fixture_error("expected_relation_path must be a list")
    edges: list[tuple[str, str, str, str]] = []
    for raw in value:
        if not isinstance(raw, list) or len(raw) not in (3, 4):
            raise _fixture_error("each expected_relation_path edge must have 3 or 4 strings")
        subject, predicate, target = (str(item).strip() for item in raw[:3])
        direction = str(raw[3]).strip() if len(raw) == 4 else "forward"
        if not subject or not predicate or not target or direction not in {"forward", "reverse"}:
            raise _fixture_error("relation edge has an invalid subject, predicate, object, or direction")
        edges.append((subject, predicate, target, direction))
    return tuple(edges)


def load_evaluation_fixture(root: str | Path) -> EvaluationFixture:
    """Load a reviewed fixture without modifying its documents or questions.

    The loader checks manifest counts and verifies every JSON document's text
    against its corresponding ``documents/*.txt`` file. It intentionally does
    not fill defaults for missing question fields: benchmark drift must fail
    before an evaluation can start.
    """
    fixture_root = Path(root).resolve()
    manifest_path = fixture_root / "benchmark_manifest.json"
    documents_path = fixture_root / "benchmark_documents.json"
    questions_path = fixture_root / "benchmark_questions.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        documents_payload = json.loads(documents_path.read_text(encoding="utf-8"))
        questions_payload = json.loads(questions_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _fixture_error(f"unable to read fixture files ({type(exc).__name__})") from exc
    if not all(isinstance(value, dict) for value in (manifest, documents_payload, questions_payload)):
        raise _fixture_error("fixture roots must be JSON objects")
    raw_documents = documents_payload.get("documents")
    raw_questions = questions_payload.get("questions")
    if not isinstance(raw_documents, list) or not isinstance(raw_questions, list):
        raise _fixture_error("documents and questions must be lists")
    if manifest.get("documents") != len(raw_documents) or manifest.get("questions") != len(raw_questions):
        raise _fixture_error("manifest counts do not match fixture payloads")
    if documents_payload.get("document_count") != len(raw_documents) or questions_payload.get("question_count") != len(raw_questions):
        raise _fixture_error("payload counts do not match fixture lists")

    documents: list[BenchmarkDocument] = []
    filenames: set[str] = set()
    for raw in raw_documents:
        if not isinstance(raw, dict):
            raise _fixture_error("document entry must be an object")
        document_id, filename, content = (str(raw.get(key, "")).strip() for key in ("id", "filename", "content"))
        if not document_id or not filename or not content or filename != Path(filename).name or filename in filenames:
            raise _fixture_error("document id/filename/content is invalid or duplicate")
        try:
            file_content = (fixture_root / "documents" / filename).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise _fixture_error(f"document text missing for {filename}") from exc
        if file_content != content:
            raise _fixture_error(f"document text differs from JSON for {filename}")
        filenames.add(filename)
        documents.append(BenchmarkDocument(document_id, filename, content))

    cases: list[EvaluationCase] = []
    question_ids: set[str] = set()
    required_fields = ("question_id", "category", "question", "expected_answer", "required_keywords", "forbidden_keywords", "expected_sources", "expected_relation_path")
    for raw in raw_questions:
        if not isinstance(raw, dict) or any(field not in raw for field in required_fields):
            raise _fixture_error("question misses a required fixed scoring field")
        question_id = str(raw["question_id"]).strip()
        category = str(raw["category"]).strip()
        question = str(raw["question"]).strip()
        expected_answer = str(raw["expected_answer"]).strip()
        keywords = raw["required_keywords"]
        forbidden = raw["forbidden_keywords"]
        sources = raw["expected_sources"]
        entities = raw.get("expected_entities", [])
        if (not question_id or question_id in question_ids or not category or not question or not expected_answer
                or not all(isinstance(value, list) for value in (keywords, forbidden, sources, entities))):
            raise _fixture_error("question identity or list field is invalid")
        normalized_sources = tuple(str(value).strip() for value in sources)
        if any(source not in filenames for source in normalized_sources):
            raise _fixture_error(f"question {question_id} references an unknown source")
        question_ids.add(question_id)
        cases.append(EvaluationCase(
            question_id=question_id,
            question=question,
            category=category,
            required_keywords=tuple(str(value).strip() for value in keywords),
            expected_sources=normalized_sources,
            forbidden_keywords=tuple(str(value).strip() for value in forbidden),
            requires_abstention=bool(raw.get("requires_abstention", False)),
            graph_advantage_expected=str(raw.get("graph_advantage_expected", "none")).lower() not in {"", "none", "false"},
            offline_entities=tuple(str(value).strip() for value in entities),
            offline_answer=expected_answer,
            expected_answer=expected_answer,
            expected_relation_path=_relation_path(raw["expected_relation_path"]),
        ))
    actual_categories: dict[str, int] = {}
    for case in cases:
        actual_categories[case.category] = actual_categories.get(case.category, 0) + 1
    declared_categories = manifest.get("question_categories")
    if declared_categories is None:
        declared_categories = questions_payload.get("category_distribution")
    if not isinstance(declared_categories, dict):
        raise _fixture_error("fixture misses a category distribution")
    try:
        normalized_declared = {str(name): int(count) for name, count in declared_categories.items()}
    except (TypeError, ValueError) as exc:
        raise _fixture_error("category distribution is invalid") from exc
    if normalized_declared != actual_categories:
        raise _fixture_error("category distribution does not match questions")
    return EvaluationFixture(
        name=str(manifest.get("benchmark_name", "evaluation fixture")),
        version=str(manifest.get("version", "")), documents=tuple(documents), cases=tuple(cases), root=fixture_root,
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


def _summary_category(category: str) -> str:
    """Map historic S4 labels and fixture labels to stable report buckets."""
    normalized = str(category).strip().casefold()
    if normalized in {"unanswerable", "abstention"}:
        return "abstention"
    if normalized.startswith("distractor"):
        return "distractor"
    if normalized.startswith("constraint"):
        return "constraint"
    if normalized == "multi_hop":
        return "multi_hop"
    return "single_hop" if normalized == "single_hop" else normalized


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


class DeterministicScorerV2:
    """Versioned, explainable scoring that keeps answer and source evidence separate.

    The implementation uses only finite fixture rules.  It deliberately does
    not re-use the v1 substring-only forbidden check: a phrase such as
    ``周宁并不负责北极星`` must not be mistaken for the opposite claim.
    """

    @staticmethod
    def _positive_claim_present(case: EvaluationCase, answer: str) -> bool:
        return any(re.search(pattern, answer, flags=re.IGNORECASE) for pattern in case.positive_claim_patterns)

    @staticmethod
    def _abstention_invents_target(case: EvaluationCase, answer: str) -> bool:
        return any(
            re.search(pattern, answer, flags=re.IGNORECASE)
            for pattern in case.abstention_positive_claim_patterns
        )

    @classmethod
    def score(
        cls,
        case: EvaluationCase,
        *,
        answer: str,
        sources: list[dict[str, Any]],
        mode: RetrievalMode,
        graph_context_count: int,
    ) -> dict[str, Any]:
        normalized_answer = answer.casefold()
        source_names = {safe_source(source.get("source", "")) for source in sources}
        expected_source_set = set(case.expected_sources)
        source_coverage = (
            1.0 if case.requires_abstention
            else round(len(expected_source_set & source_names) / len(expected_source_set), 3)
            if expected_source_set else 1.0
        )
        source_complete = source_coverage == 1.0
        graph_participated = graph_context_count > 0

        if case.requires_abstention:
            # A reviewed fixture may define additional finite, answer-safe
            # abstention expressions (for example "未记录").  They remain
            # case-owned deterministic rules, not a model-based judgment.
            abstention_markers = _ABSTENTION_MARKERS_V2 + tuple(case.required_keywords)
            abstention_marker = next((marker for marker in abstention_markers if marker in answer), None)
            invented_target_fact = cls._abstention_invents_target(case, answer)
            abstention_score = 1.0 if abstention_marker and not invented_target_fact else 0.0
            answer_semantic_score = abstention_score
            citation_behavior = "scope_sources_allowed" if sources else "no_sources_required"
            answer_rule = "abstention_marker" if abstention_marker else "missing_abstention_marker"
            if invented_target_fact:
                answer_rule = "invented_target_fact"
            required_hits: list[str] = []
            forbidden_hits: list[str] = []
        elif case.negation_target_terms:
            negation_marker = next((marker for marker in _NEGATION_MARKERS_V2 if marker in answer), None)
            targets_present = all(term in answer for term in case.negation_target_terms)
            positive_claim = cls._positive_claim_present(case, answer)
            answer_semantic_score = 1.0 if negation_marker and targets_present and not positive_claim else 0.0
            abstention_score = None
            citation_behavior = "supporting_sources" if sources else "no_supporting_sources"
            answer_rule = "negation_supported" if answer_semantic_score else "negation_not_established"
            required_hits = [term for term in case.negation_target_terms if term in answer]
            forbidden_hits = ["positive_claim"] if positive_claim else []
        else:
            required_hits = [keyword for keyword in case.required_keywords if keyword.casefold() in normalized_answer]
            forbidden_hits = [keyword for keyword in case.forbidden_keywords if keyword.casefold() in normalized_answer]
            answer_semantic_score = round(len(required_hits) / len(case.required_keywords), 3) if case.required_keywords else 1.0
            if forbidden_hits:
                answer_semantic_score = 0.0
            abstention_score = None
            citation_behavior = "supporting_sources" if sources else "no_supporting_sources"
            answer_rule = "keyword_rules"

        graph_participation_pass = mode is not RetrievalMode.GRAPH_RAG or case.requires_abstention or graph_participated
        # Source coverage and graph participation remain separately observable;
        # neither is allowed to erase a semantically correct answer.
        overall_v2 = answer_semantic_score
        return {
            "scorer_version": SCORER_V2,
            "answer_semantic_score": answer_semantic_score,
            "source_coverage_score": source_coverage,
            "abstention_score": abstention_score,
            "citation_behavior": citation_behavior,
            "overall_v2": overall_v2,
            "question_correct_v2": overall_v2 == 1.0,
            "answer_rule": answer_rule,
            "required_keyword_hits_v2": required_hits,
            "forbidden_keyword_hits_v2": forbidden_hits,
            "source_coverage_complete": source_complete,
            "graph_participated": graph_participated,
            "graph_participation_pass": graph_participation_pass,
        }


def score_relation_fidelity(case: EvaluationCase, contexts: Iterable[Any]) -> dict[str, Any]:
    """Score exact directed fixture relations against graph evidence in Top-K.

    This is deliberately independent from answer correctness. A final answer
    may be fluent while its evidence contains a different predicate, such as
    ``DEPENDS_ON`` instead of ``PROVIDES_INDEX_TO``.
    """
    expected = {
        (subject, canonical_predicate(predicate), object_, direction)
        for subject, predicate, object_, direction in case.expected_relation_path
    }
    observed: set[tuple[str, str, str, str]] = set()
    for context in contexts:
        if getattr(context, "retrieval_type", "") != "graph":
            continue
        metadata = getattr(context, "metadata", {}) or {}
        for edge in metadata.get("graph_evidence", []):
            if not isinstance(edge, dict):
                continue
            direction = str(edge.get("direction", "")).strip()
            subject, predicate, object_ = (str(edge.get(field, "")).strip() for field in ("subject", "predicate", "object"))
            if subject and predicate and object_ and direction in {"forward", "reverse"}:
                observed.add((subject, canonical_predicate(predicate), object_, direction))
    matched = expected & observed
    return {
        "expected_relation_total": len(expected),
        "matched_relation_count": len(matched),
        "relation_fidelity_score": round(len(matched) / len(expected), 3) if expected else None,
        "relation_fidelity_pass": bool(expected) and matched == expected,
        "expected_relations": sorted("|".join(edge) for edge in expected),
        "matched_relations": sorted("|".join(edge) for edge in matched),
    }


def graph_evidence_diagnostics(
    case: EvaluationCase, contexts: Iterable[Any]
) -> dict[str, Any]:
    """Return safe graph edges that actually enter an evaluation prompt.

    This is forward-looking diagnostics only: it never reconstructs or mutates
    historical prompts. A conflict records multiple predicates on one directed
    pair; it does not infer that either predicate is false.
    """
    edges: list[dict[str, Any]] = []
    violations: list[dict[str, str]] = []
    for context in contexts:
        if getattr(context, "retrieval_type", "") != "graph":
            continue
        metadata = getattr(context, "metadata", {}) or {}
        for raw in metadata.get("graph_evidence", []):
            if not isinstance(raw, dict):
                continue
            edge = {
                "subject": " ".join(str(raw.get("subject", "")).split()),
                "predicate": " ".join(str(raw.get("predicate", "")).split()),
                "object": " ".join(str(raw.get("object", "")).split()),
                "direction": " ".join(str(raw.get("direction", "")).split()),
                "document_id": " ".join(str(raw.get("document_id", "")).split()),
                "document_version": raw.get("document_version"),
                "source": safe_source(raw.get("source", "")),
                "evidence_key": " ".join(str(raw.get("evidence_key", "")).split()),
                "raw_predicate": " ".join(str(raw.get("raw_predicate", raw.get("predicate", ""))).split())[:160],
                "relation_semantics_version": " ".join(
                    str(raw.get("relation_semantics_version", "legacy-unversioned")).split()
                ),
            }
            if not all(str(edge[field]).strip() for field in ("subject", "predicate", "object", "direction", "document_id", "source", "evidence_key")):
                violations.append({"type": "incomplete_edge", "evidence_key": edge["evidence_key"]})
                continue
            if edge["direction"] not in {"forward", "reverse"}:
                violations.append({"type": "invalid_direction", "evidence_key": edge["evidence_key"]})
                continue
            if not is_canonical_predicate(edge["predicate"]):
                violations.append({"type": "noncanonical_predicate", "evidence_key": edge["evidence_key"]})
            edges.append(edge)
    edges.sort(key=lambda edge: (edge["document_id"], int(edge["document_version"] or 0), edge["evidence_key"]))
    predicates = sorted({edge["predicate"] for edge in edges})
    directed_predicates: dict[tuple[str, str, str], set[str]] = {}
    for edge in edges:
        directed_predicates.setdefault((edge["subject"], edge["object"], edge["direction"]), set()).add(edge["predicate"])
    conflicts = [
        {"subject": subject, "object": object_, "direction": direction, "predicates": sorted(predicates)}
        for (subject, object_, direction), predicates in sorted(directed_predicates.items())
        if len(predicates) > 1
    ]
    expected = {
        (subject, canonical_predicate(predicate), object_, direction)
        for subject, predicate, object_, direction in case.expected_relation_path
    }
    observed = {
        (edge["subject"], canonical_predicate(edge["predicate"]), edge["object"], edge["direction"])
        for edge in edges
    }
    for relation in sorted(expected - observed):
        violations.append({"type": "expected_relation_not_in_prompt", "relation": "|".join(relation)})
    return {
        "graph_evidence_edges": edges,
        "graph_predicates_used": predicates,
        "relation_fidelity_violations": violations,
        "relation_conflict_detected": bool(conflicts),
        "relation_conflicts": conflicts,
        "relation_semantics_version": RELATION_SEMANTICS_VERSION,
    }


def _case_by_id(question_id: str) -> EvaluationCase:
    for case in EVALUATION_CASES:
        if case.question_id == question_id:
            return case
    raise ValueError(f"unknown evaluation question_id: {question_id}")


def rescore_payload_v2(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a v2 score-only view without mutating a historical run payload."""
    rows: list[dict[str, Any]] = []
    for original in payload.get("results", []):
        if not original.get("success"):
            continue
        case = _case_by_id(str(original.get("question_id", "")))
        mode = RetrievalMode(str(original.get("mode", "")))
        v1 = original.get("deterministic_score") or {}
        v2 = DeterministicScorerV2.score(
            case,
            answer=str(original.get("answer", "")),
            sources=list(original.get("sources") or []),
            mode=mode,
            graph_context_count=int(original.get("graph_context_count") or 0),
        )
        # Do not copy full generated answers into the derived report.
        rows.append({
            "mode": mode.value,
            "question_id": case.question_id,
            "category": case.category,
            "v1_question_correct": bool(v1.get("question_correct")),
            "v1_keyword_fact_score": v1.get("keyword_fact_score"),
            "v1_source_hit": v1.get("source_hit"),
            "v1_abstention_correct": v1.get("abstention_correct"),
            "v2": v2,
            "source_names": sorted({safe_source(source.get("source", "")) for source in original.get("sources", [])}),
        })
    summary: dict[str, dict[str, Any]] = {}
    for mode in sorted({row["mode"] for row in rows}):
        mode_rows = [row for row in rows if row["mode"] == mode]
        summary[mode] = {
            "rows": len(mode_rows),
            "v1_question_accuracy": round(sum(row["v1_question_correct"] for row in mode_rows) / len(mode_rows), 3) if mode_rows else 0.0,
            "v2_question_accuracy": round(sum(row["v2"]["question_correct_v2"] for row in mode_rows) / len(mode_rows), 3) if mode_rows else 0.0,
            "v2_answer_semantic_average": round(statistics.fmean(float(row["v2"]["answer_semantic_score"]) for row in mode_rows), 3) if mode_rows else 0.0,
            "v2_source_coverage_average": round(statistics.fmean(float(row["v2"]["source_coverage_score"]) for row in mode_rows), 3) if mode_rows else 0.0,
        }
    return {
        "source_run_id": payload.get("run_id"),
        "scorer_version": SCORER_V2,
        "notice": "Derived deterministic rescoring only. It does not alter source answers, raw outputs, or the original v1 report.",
        "results": rows,
        "summary": summary,
    }


def write_rescore_reports(output_dir: Path, rescored: dict[str, Any]) -> None:
    """Write score-only v2 reports atomically; generated answers are excluded."""
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "results.json", rescored)
    fieldnames = ("mode", "question_id", "category", "v1_question_correct", "v1_keyword_fact_score", "v1_source_hit", "v1_abstention_correct", "v2", "source_names")
    temporary = output_dir / "results.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for result in rescored["results"]:
            row = dict(result)
            row["v2"] = json.dumps(row["v2"], ensure_ascii=False, sort_keys=True)
            row["source_names"] = json.dumps(row["source_names"], ensure_ascii=False)
            writer.writerow(row)
    os.replace(temporary, output_dir / "results.csv")
    lines = ["# S4 deterministic scorer v2 rescoring", "", rescored["notice"], "", "| Mode | v1 accuracy | v2 accuracy | v2 semantic | v2 source coverage |", "|---|---:|---:|---:|---:|"]
    for mode, values in rescored["summary"].items():
        lines.append(f"| {mode} | {values['v1_question_accuracy']:.3f} | {values['v2_question_accuracy']:.3f} | {values['v2_answer_semantic_average']:.3f} | {values['v2_source_coverage_average']:.3f} |")
    temporary_md = output_dir / "summary.md.tmp"
    temporary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary_md, output_dir / "summary.md")


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
            score_v2 = DeterministicScorerV2.score(
                case, answer=result.answer, sources=sources, mode=mode, graph_context_count=graph_count
            )
            relation_fidelity = score_relation_fidelity(case, result.contexts)
            evidence_diagnostics = graph_evidence_diagnostics(case, result.contexts)
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
                "deterministic_score_v2": score_v2,
                "relation_fidelity": relation_fidelity,
                **evidence_diagnostics,
                "forbidden_fact_violation": bool(score["forbidden_keyword_hits"]),
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
                "deterministic_score": None, "deterministic_score_v2": None,
                "relation_fidelity": None, "forbidden_fact_violation": None,
                "graph_evidence_edges": [], "graph_predicates_used": [],
                "relation_fidelity_violations": [], "relation_conflict_detected": False,
                "relation_conflicts": [], "relation_semantics_version": RELATION_SEMANTICS_VERSION,
                "failure_summary": type(exc).__name__,
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
        categories = ("single_hop", "multi_hop", "constraint", "distractor", "abstention")
        for mode in sorted({result["mode"] for result in results}):
            rows = [result for result in results if result["mode"] == mode]
            successful = [row for row in rows if row["success"]]
            correct = [row for row in successful if row.get("deterministic_score", {}).get("question_correct")]
            correct_v2 = [row for row in successful if row.get("deterministic_score_v2", {}).get("question_correct_v2")]
            multi_hop = [row for row in successful if "multi_hop" in row["category"]]
            abstentions = [row for row in successful if row["category"] in {"unanswerable", "abstention"}]
            graph_called = [row for row in successful if int(row.get("model_call_counts", {}).get("graph_calls") or 0) > 0]
            graph_top_k = [row for row in successful if row["graph_context_count"] > 0]
            latencies = [float(row["latency_ms"]) for row in successful]
            category_breakdown: dict[str, dict[str, Any]] = {}
            for category in categories:
                category_rows = [row for row in rows if _summary_category(row["category"]) == category]
                category_correct = [row for row in category_rows if row.get("deterministic_score_v2", {}).get("question_correct_v2")]
                category_breakdown[category] = {
                    "correct": len(category_correct), "total": len(category_rows),
                    "accuracy": round(len(category_correct) / len(category_rows), 3) if category_rows else None,
                }
            plan_calls = {
                row["question_id"]: int(row.get("model_call_counts", {}).get("shared_preprocess_chat") or 0)
                for row in rows
            }
            summary[mode] = {
                "questions": len(rows),
                "success_rate": round(len(successful) / len(rows), 3) if rows else 0.0,
                "question_accuracy": round(len(correct) / len(rows), 3) if rows else 0.0,
                "question_accuracy_v2": round(len(correct_v2) / len(rows), 3) if rows else 0.0,
                "answer_semantic_score_v2": round(
                    statistics.fmean(row["deterministic_score_v2"]["answer_semantic_score"] for row in successful), 3
                ) if successful else 0.0,
                "keyword_fact_hit_rate": round(
                    statistics.fmean(row["deterministic_score"]["keyword_fact_score"] for row in successful), 3
                ) if successful else 0.0,
                "required_fact_hit_rate": round(
                    statistics.fmean(row["deterministic_score"]["keyword_fact_score"] for row in successful), 3
                ) if successful else 0.0,
                "source_hit_rate": round(sum(bool(row["deterministic_score"]["source_hit"]) for row in successful) / len(rows), 3) if rows else 0.0,
                "source_coverage": round(
                    statistics.fmean(row["deterministic_score_v2"]["source_coverage_score"] for row in successful), 3
                ) if successful else 0.0,
                "multi_hop_accuracy": round(sum(bool(row["deterministic_score"]["question_correct"]) for row in multi_hop) / len(multi_hop), 3) if multi_hop else None,
                "abstention_accuracy": round(sum(bool(row["deterministic_score"]["abstention_correct"]) for row in abstentions) / len(abstentions), 3) if abstentions else None,
                "relation_fidelity": round(statistics.fmean(
                    row["relation_fidelity"]["relation_fidelity_score"]
                    for row in successful if row.get("relation_fidelity", {}).get("relation_fidelity_score") is not None
                ), 3) if any(row.get("relation_fidelity", {}).get("relation_fidelity_score") is not None for row in successful) else None,
                "forbidden_fact_violation_rate": round(sum(bool(row.get("forbidden_fact_violation")) for row in successful) / len(rows), 3) if rows else 0.0,
                "graph_participation_rate": round(len(graph_called) / len(rows), 3) if mode == RetrievalMode.GRAPH_RAG.value and rows else None,
                "graph_evidence_entering_top_k_rate": round(len(graph_top_k) / len(rows), 3) if mode == RetrievalMode.GRAPH_RAG.value and rows else None,
                "latency_avg_ms": round(statistics.fmean(latencies), 3) if latencies else None,
                "latency_p50_ms": _nearest_rank(latencies, 50),
                "latency_p95_ms": _nearest_rank(latencies, 95),
                "chat_logical_calls": sum(plan_calls.values()) + sum(int(row.get("model_call_counts", {}).get("answer_chat") or 0) for row in rows),
                "embedding_logical_calls": sum(int(row.get("model_call_counts", {}).get("embedding_query") or 0) for row in rows),
                "neo4j_logical_calls": sum(int(row.get("model_call_counts", {}).get("graph_calls") or 0) for row in rows),
                "categories": category_breakdown,
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
            "deterministic_score", "deterministic_score_v2", "relation_fidelity", "graph_evidence_edges",
            "graph_predicates_used", "relation_fidelity_violations", "relation_conflict_detected",
            "relation_conflicts", "relation_semantics_version", "forbidden_fact_violation", "failure_summary",
        ]
        temporary = output_dir / "results.csv.tmp"
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for result in payload["results"]:
                row = dict(result)
                for name in ("sources", "model_call_counts", "deterministic_score", "deterministic_score_v2", "relation_fidelity", "graph_evidence_edges", "graph_predicates_used", "relation_fidelity_violations", "relation_conflicts"):
                    row[name] = json.dumps(row[name], ensure_ascii=False, sort_keys=True)
                writer.writerow(row)
        os.replace(temporary, output_dir / "results.csv")
        lines = ["# Offline RAG evaluation", "", payload["notice"], "", "| Mode | Questions | Success | v1 accuracy | v2 accuracy | Graph participation |", "|---|---:|---:|---:|---:|---:|"]
        for mode, values in payload["summary"].items():
            lines.append(f"| {mode} | {values['questions']} | {values['success_rate']:.3f} | {values['question_accuracy']:.3f} | {values['question_accuracy_v2']:.3f} | {values['graph_participation_rate'] if values['graph_participation_rate'] is not None else '-'} |")
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
        if case:
            return _OfflineMessage(case.offline_answer or case.expected_answer or "资料未提供，无法回答。")
        return _OfflineMessage("资料未提供，无法回答。")


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
        case_list = tuple(cases)
        self.entity_source = {
            entity: case.expected_sources[0]
            for case in case_list if case.expected_sources
            for entity in case.offline_entities
        }
        self.evidence_by_entity: dict[str, list[dict[str, Any]]] = {}
        for case in case_list:
            if not case.expected_sources:
                continue
            source = case.expected_sources[0]
            document_id = offline_document_id(source)
            for index, (subject, predicate, target, direction) in enumerate(case.expected_relation_path):
                edge = {
                    "subject": subject, "predicate": predicate, "object": target,
                    "direction": direction, "document_id": document_id,
                    "document_version": 1, "source": source,
                    "evidence_key": f"offline-{document_id[:8]}-{case.question_id}-{index}",
                }
                self.evidence_by_entity.setdefault(subject, []).append(edge)
                self.evidence_by_entity.setdefault(target, []).append(edge)
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
        edges = self.evidence_by_entity.get(entity_name, [])
        source = self.entity_source.get(entity_name)
        if not source and not edges:
            return []
        if edges:
            return [{
                "source": edge["subject"], "relations": [edge["predicate"]], "target": edge["object"],
                "target_type": "Concept", "target_desc": "offline fixture evidence",
                "evidence_edges": [edge],
            } for edge in edges] + [{
                "source": entity_name, "relations": ["RELATED_TO"], "target": "foreign",
                "target_type": "Concept", "target_desc": "foreign evidence",
                "evidence_edges": [{
                    "subject": entity_name, "predicate": "RELATED_TO", "object": "foreign",
                    "direction": "forward", "document_id": str(uuid5(NAMESPACE_URL, "foreign-graph")),
                    "document_version": 1, "source": "foreign.txt", "evidence_key": "foreign-evidence",
                }],
            }, {
                "source": entity_name, "relations": ["RELATED_TO"], "target": "legacy",
                "target_type": "Concept", "target_desc": "legacy evidence", "evidence_edges": [],
            }]
        document_id = offline_document_id(source)
        return [{
            "source": entity_name, "relations": ["RELATED_TO"], "target": "synthetic",
            "target_type": "Concept", "target_desc": "offline evidence",
            "evidence_edges": [{
                "subject": entity_name, "predicate": "RELATED_TO", "object": "synthetic",
                "direction": "forward", "document_id": document_id, "document_version": 1,
                "source": source, "evidence_key": f"offline-{document_id[:8]}-{entity_name}",
            }],
        }, {
            "source": entity_name, "relations": ["RELATED_TO"], "target": "foreign",
            "target_type": "Concept", "target_desc": "foreign evidence",
            "evidence_edges": [{
                "subject": entity_name, "predicate": "RELATED_TO", "object": "foreign",
                "direction": "forward", "document_id": str(uuid5(NAMESPACE_URL, "foreign-graph")),
                "document_version": 1, "source": "foreign.txt", "evidence_key": "foreign-evidence",
            }],
        }, {
            "source": entity_name, "relations": ["RELATED_TO"], "target": "legacy",
            "target_type": "Concept", "target_desc": "legacy evidence", "evidence_edges": [],
        }]


def build_offline_runner(
    run_id: str = "offline-fixture", *, fixture: EvaluationFixture | None = None,
) -> RAGEvaluationRunner:
    """Construct a fake-only runner without importing configuration or providers."""
    cases = fixture.cases if fixture else EVALUATION_CASES
    embeddings = OfflineEmbeddingProvider()
    scope = EvaluationScope.from_uploaded_document_ids(
        run_id,
        (offline_document_id(document.filename) for document in fixture.documents)
        if fixture else (offline_document_id(source) for source in _OFFLINE_SOURCES),
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


def run_offline_sync(
    run_id: str, modes: Iterable[RetrievalMode | str], output_root: str | Path,
    *, fixture: EvaluationFixture | None = None,
) -> dict[str, Any]:
    return asyncio.run(build_offline_runner(run_id, fixture=fixture).run(run_id=run_id, modes=modes, output_root=output_root))
