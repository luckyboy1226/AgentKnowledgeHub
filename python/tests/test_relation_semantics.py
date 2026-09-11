"""Fake-only S4.4 contracts for directed relation semantics."""

from __future__ import annotations

import pytest

from agents.knowledge_extract_agent import Entity, ExtractionResult, Relation
from agents.qa_agent import QAAgent
from services.document_processor import DocumentProcessorAdapter
from services.rag_evaluation import EvaluationCase, graph_evidence_diagnostics, score_relation_fidelity
from services.relation_semantics import (
    RELATION_SEMANTICS_VERSION,
    canonicalize_relation,
    is_canonical_predicate,
)


def _edge(predicate: str, *, raw: str | None = None, direction: str = "forward") -> dict:
    return {
        "subject": "北极星", "predicate": predicate, "object": "天枢", "direction": direction,
        "document_id": "doc-s4", "document_version": 1, "source": "s4-eval-collaboration.txt",
        "evidence_key": f"e-{predicate}", "raw_predicate": raw or predicate,
        "relation_semantics_version": RELATION_SEMANTICS_VERSION,
    }


def test_provides_index_is_canonical_and_not_dependency():
    relation = canonicalize_relation("为天枢提供检索索引")
    assert relation.predicate == "PROVIDES_INDEX"
    assert canonicalize_relation("depends on").predicate == "DEPENDS_ON"
    assert relation.predicate != canonicalize_relation("depends on").predicate


def test_unknown_or_dangerous_relation_falls_back_to_related_to():
    assert canonicalize_relation("drop; delete").predicate == "RELATED_TO"
    assert canonicalize_relation("任意未审核关系").predicate == "RELATED_TO"


def test_uses_and_responsibility_cannot_become_dependency_or_membership():
    assert canonicalize_relation("使用").predicate == "USES"
    assert canonicalize_relation("负责").predicate == "RESPONSIBLE_FOR"
    assert canonicalize_relation("使用").predicate != "DEPENDS_ON"
    assert canonicalize_relation("负责").predicate != "MEMBER_OF"


def test_provides_index_synonyms_share_one_semantic_class():
    assert canonicalize_relation("提供索引").predicate == "PROVIDES_INDEX"
    assert canonicalize_relation("提供检索索引").predicate == "PROVIDES_INDEX"


def test_raw_predicate_is_bounded_without_affecting_canonical_value():
    relation = canonicalize_relation("依赖", raw_predicate="x" * 400)
    assert relation.predicate == "DEPENDS_ON"
    assert len(relation.raw_predicate) == 160


def test_benchmark_predicates_are_explicit_reviewed_vocabulary():
    assert is_canonical_predicate("PROVIDES_INDEX")
    assert is_canonical_predicate("NOT_DEPENDS_ON")
    assert not is_canonical_predicate("INVENTED_RELATION")


def test_direction_is_not_reversed_by_canonicalization():
    relation = canonicalize_relation("提供检索索引")
    assert relation.predicate == "PROVIDES_INDEX"
    assert relation.raw_predicate == "提供检索索引"


def test_evidence_key_is_stable_for_same_canonical_relation():
    from services.knowledge_graph import KnowledgeGraphService
    relation_a = canonicalize_relation("提供索引")
    relation_b = canonicalize_relation("提供检索索引")
    assert relation_a.predicate == relation_b.predicate
    assert KnowledgeGraphService.evidence_key("doc", 1, "A", relation_a.predicate, "B") == (
        KnowledgeGraphService.evidence_key("doc", 1, "A", relation_b.predicate, "B")
    )


def test_extraction_parser_preserves_model_raw_predicate_without_provider_call():
    from agents.knowledge_extract_agent import KnowledgeExtractAgent
    parsed = KnowledgeExtractAgent(chat_provider=None)._parse_response(
        '{"entities":[{"name":"A","type":"Concept"},{"name":"B","type":"Concept"}],'
        '"relations":[{"head":"A","relation":"provides_index","raw_predicate":"提供检索索引",'
        '"tail":"B","confidence":0.9}],"events":[]}',
        "chunk-1",
    )
    assert parsed.relations[0].properties["raw_predicate"] == "提供检索索引"


def test_scoped_evidence_preserves_raw_predicate_and_directed_canonical_triple():
    text = QAAgent._format_scoped_graph_evidence([_edge("PROVIDES_INDEX", raw="提供检索索引")])
    assert "北极星 --PROVIDES_INDEX--> 天枢" in text
    assert "原始关系: 提供检索索引" in text
    assert "DEPENDS_ON" not in text


def test_scoped_evidence_never_conflates_provides_index_with_depends_on():
    provides = QAAgent._format_scoped_graph_evidence([_edge("PROVIDES_INDEX")])
    depends = QAAgent._format_scoped_graph_evidence([_edge("DEPENDS_ON")])
    assert "DEPENDS_ON" not in provides
    assert "PROVIDES_INDEX_TO" not in depends


def test_relation_fidelity_requires_exact_predicate_and_direction():
    case = EvaluationCase("R1", "q", "multi_hop", (), (), expected_relation_path=(
        ("北极星", "PROVIDES_INDEX", "天枢", "forward"),
    ))
    context = type("Context", (), {"retrieval_type": "graph", "metadata": {"graph_evidence": [_edge("DEPENDS_ON")]}})()
    score = score_relation_fidelity(case, [context])
    assert score["relation_fidelity_score"] == 0.0
    assert not score["relation_fidelity_pass"]


def test_relation_fidelity_rejects_reversed_direction():
    case = EvaluationCase("R2", "q", "multi_hop", (), (), expected_relation_path=(
        ("北极星", "PROVIDES_INDEX", "天枢", "forward"),
    ))
    context = type("Context", (), {"retrieval_type": "graph", "metadata": {"graph_evidence": [_edge("PROVIDES_INDEX", direction="reverse")]}})()
    assert score_relation_fidelity(case, [context])["relation_fidelity_score"] == 0.0


def test_future_run_diagnostics_capture_edges_and_missing_expected_relation():
    case = EvaluationCase("R3", "q", "multi_hop", (), (), expected_relation_path=(
        ("北极星", "PROVIDES_INDEX", "天枢", "forward"),
    ))
    context = type("Context", (), {"retrieval_type": "graph", "metadata": {"graph_evidence": [_edge("DEPENDS_ON")]}})()
    diagnostics = graph_evidence_diagnostics(case, [context])
    assert diagnostics["graph_predicates_used"] == ["DEPENDS_ON"]
    assert diagnostics["graph_evidence_edges"][0]["raw_predicate"] == "DEPENDS_ON"
    assert diagnostics["relation_fidelity_violations"][-1]["type"] == "expected_relation_not_in_prompt"


def test_conflicting_predicates_for_same_directed_pair_are_audited_not_inferred():
    case = EvaluationCase("R4", "q", "multi_hop", (), ())
    context = type("Context", (), {"retrieval_type": "graph", "metadata": {"graph_evidence": [
        _edge("PROVIDES_INDEX"), _edge("DEPENDS_ON"),
    ]}})()
    diagnostics = graph_evidence_diagnostics(case, [context])
    assert diagnostics["relation_conflict_detected"] is True
    assert diagnostics["relation_conflicts"][0]["predicates"] == ["DEPENDS_ON", "PROVIDES_INDEX"]


def test_diagnostic_evidence_scrubs_an_absolute_source_path():
    case = EvaluationCase("R5", "q", "multi_hop", (), ())
    edge = _edge("PROVIDES_INDEX")
    edge["source"] = r"C:\private\evidence.txt"
    context = type("Context", (), {"retrieval_type": "graph", "metadata": {"graph_evidence": [edge]}})()
    diagnostics = graph_evidence_diagnostics(case, [context])
    assert diagnostics["graph_evidence_edges"][0]["source"] == "evidence.txt"


class _Parser:
    async def parse(self, _path):
        from agents.doc_parser_agent import DocType, DocumentChunk
        return [DocumentChunk("x", "parser", 0, DocType.TEXT, {})]


class _Extractor:
    def __init__(self, relation: Relation):
        self.relation = relation

    async def extract(self, _chunks):
        return [ExtractionResult(
            [Entity("北极星", "Product"), Entity("天枢", "Product")], [self.relation], []
        )]


async def _prepared_relation(tmp_path, relation: Relation):
    processor = DocumentProcessorAdapter(_Parser(), _Extractor(relation), temp_root=tmp_path)
    artifact = await processor.prepare(
        content=b"x", filename="safe.txt", document_id="doc", version=1,
        content_hash="a" * 64, operation_id="op",
    )
    return artifact.relations[0]


@pytest.mark.asyncio
async def test_processor_preserves_raw_predicate_and_canonicalizes_provides_index(tmp_path):
    relation = await _prepared_relation(tmp_path, Relation("北极星", "提供检索索引", "天枢"))
    assert relation.relation == "PROVIDES_INDEX"
    assert relation.properties["raw_predicate"] == "提供检索索引"
    assert relation.properties["relation_semantics_version"] == RELATION_SEMANTICS_VERSION


@pytest.mark.asyncio
async def test_processor_does_not_reorient_relation_endpoints(tmp_path):
    relation = await _prepared_relation(tmp_path, Relation("北极星", "provides index to", "天枢"))
    assert (relation.head, relation.tail) == ("北极星", "天枢")


@pytest.mark.asyncio
async def test_processor_deduplicates_by_canonical_relation_without_erasing_raw_audit(tmp_path):
    relation = await _prepared_relation(tmp_path, Relation("北极星", "provides index", "天枢"))
    assert relation.relation == "PROVIDES_INDEX"
