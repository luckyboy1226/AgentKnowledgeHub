import json

from services.hybrid_retrieval_v2_evaluation import VARIANTS, run_fake_evaluation


def _rows(payload, question_id):
    return {row["variant"]: row for row in payload["results"] if row["question_id"] == question_id}


def test_fake_phase_f_variants_metrics_and_na_semantics(tmp_path):
    payload = run_fake_evaluation(tmp_path, "phase-f-unit")
    assert {row["variant"] for row in payload["results"]} == set(VARIANTS)
    exact = _rows(payload, "F01")
    assert exact["bm25_vector_rrf"]["retrieval_metrics"]["mrr"] > exact["vector_only"]["retrieval_metrics"]["mrr"]
    assert exact["vector_only"]["graph_metrics"]["applicable"] is False
    assert exact["vector_only"]["graph_metrics"]["complete_path_coverage"] is None


def test_graph_path_is_directed_and_full_only_for_graph_variants(tmp_path):
    payload = run_fake_evaluation(tmp_path, "phase-f-graph")
    rows = _rows(payload, "F03")
    assert rows["vector_only"]["graph_metrics"]["complete_path_coverage"] is None
    assert rows["bm25_vector_rrf"]["graph_metrics"]["complete_path_coverage"] is None
    assert rows["vector_graph_rrf"]["graph_metrics"]["complete_path_coverage"] is True
    assert rows["hybrid_v2_full"]["graph_metrics"]["complete_path_coverage"] is True


def test_recovery_is_idempotent_and_json_is_machine_readable(tmp_path):
    first = run_fake_evaluation(tmp_path, "phase-f-recovery", ("vector_only",))
    second = run_fake_evaluation(tmp_path, "phase-f-recovery", VARIANTS)
    assert len(first["results"]) == 8
    assert len(second["results"]) == 8 * len(VARIANTS)
    output = tmp_path / "phase-f-recovery"
    assert json.loads((output / "results.json").read_text(encoding="utf-8"))["fake_only"] is True
    assert (output / "summary.json").exists() and (output / "summary.md").exists()


def test_variant_order_does_not_change_core_rows(tmp_path):
    forward = run_fake_evaluation(tmp_path, "phase-f-forward", VARIANTS)
    reverse = run_fake_evaluation(tmp_path, "phase-f-reverse", tuple(reversed(VARIANTS)))
    def core(payload):
        return [{key: value for key, value in row.items() if key not in {"_elapsed_unused", "run_id", "request_id"}} for row in payload["results"]]
    assert core(forward) == core(reverse)


def test_optional_traces_are_observers_and_graph_trace_is_graph_only(tmp_path):
    plain = run_fake_evaluation(tmp_path, "phase-f-plain")
    traced = run_fake_evaluation(tmp_path, "phase-f-traced", retrieval_trace=True, graph_trace=True)
    assert [row["retrieval_metrics"] for row in plain["results"]] == [row["retrieval_metrics"] for row in traced["results"]]
    traces = json.loads((tmp_path / "phase-f-traced" / "graph-trace.json").read_text(encoding="utf-8"))
    assert traces and {row["variant"] for row in traces} == {"vector_graph_rrf", "hybrid_v2_full"}
