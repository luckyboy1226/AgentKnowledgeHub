import json

from services.retrieval_benchmark_v2 import BENCHMARK_VERSION, build_retrieval_benchmark_v2


def test_ground_truth_is_stable_document_level_and_reviewable(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "benchmark_documents.json").write_text(json.dumps({"documents": [{"id": "D01", "filename": "one.txt"}]}), encoding="utf-8")
    (source / "benchmark_questions.json").write_text(json.dumps({"questions": [
        {"question_id": "Q1", "category": "single_hop", "question": "x", "expected_sources": ["one.txt"], "expected_relation_path": [["A", "USES", "B"]], "requires_abstention": False},
        {"question_id": "Q2", "category": "abstention", "question": "x", "expected_sources": [], "expected_relation_path": [], "requires_abstention": True},
    ]}), encoding="utf-8")
    target = tmp_path / "retrieval-v2"
    manifest = build_retrieval_benchmark_v2(source, target)
    payload = json.loads((target / "ground_truth.json").read_text(encoding="utf-8"))
    assert manifest["benchmark_version"] == BENCHMARK_VERSION
    assert payload["relevance_level"] == "document"
    assert payload["questions"][0]["relevant_documents"] == ["D01"]
    assert payload["questions"][0]["expected_edges"] == [["A", "USES", "B", "forward"]]
    assert payload["questions"][1]["answerable"] is False
    assert not any(row["needs_manual_review"] for row in payload["questions"])
    assert "Retrieval Ground Truth Review" in (target / "ground_truth_review.md").read_text(encoding="utf-8")
