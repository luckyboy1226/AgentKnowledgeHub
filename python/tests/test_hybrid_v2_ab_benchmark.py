import asyncio
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from types import SimpleNamespace

from services.hybrid_v2_ab_benchmark import (ABCase, DeterministicOfflineABAdapter,
    FrozenABInputs, GRAPH_MODES, HybridV2ABRunner, MODES, graph_metrics,
    offline_inputs, percentile, retrieval_metrics, stable_hash, summarize)
from services.hybrid_v2_ab_benchmark import summary_markdown
from services.hybrid_retrieval_v2_runtime import build_hosted_real_runtime


ROOT=Path(__file__).resolve().parents[2]
BENCHMARK=ROOT/"benchmarks"/"enterprise_20docs_retrieval_v2"
SCRIPT=ROOT/"scripts"/"run-hybrid-v2-ab.py"


def inputs(): return offline_inputs(BENCHMARK,run_id="unit-frozen")


def test_enterprise_fixture_builds_sixty_plans_and_one_hundred_eighty_queries():
    frozen=inputs()
    assert len(frozen.cases)==60 and len(frozen.plans)==60
    assert sum(len(plan["queries"]) for plan in frozen.plans)==180
    assert len(frozen.allowed_document_ids)==20


def test_four_modes_share_exact_plan_object_snapshot_identity_and_allowlist(tmp_path):
    async def run():
        frozen=inputs(); adapter=DeterministicOfflineABAdapter(frozen.cases)
        result=await HybridV2ABRunner(inputs=frozen,adapter=adapter,output_root=tmp_path,ab_run_id="shared").run()
        assert len(result["payload"]["results"])==720
        groups={}
        for plan_id,ordinal,mode,vector_sha,scope in adapter.calls:
            groups.setdefault((plan_id,ordinal),(set(),set(),set())).__getitem__(0).add(mode)
            groups[(plan_id,ordinal)][1].add(vector_sha); groups[(plan_id,ordinal)][2].add(scope)
        assert all(modes==set(MODES) and len(vectors)==1 and scopes=={frozen.allowed_document_ids} for modes,vectors,scopes in groups.values())
    asyncio.run(run())


@pytest.mark.parametrize("field,replacement",[
    ("allowlist_hash","0"*64),("snapshot_root_hash","1"*64),("fixture_fingerprint","bad"),("query_plan_hash","2"*64),
])
def test_frozen_identity_mismatch_fails_closed(field,replacement,tmp_path):
    frozen=replace(inputs(),**{field:replacement})
    with pytest.raises(ValueError): HybridV2ABRunner(inputs=frozen,adapter=DeterministicOfflineABAdapter(frozen.cases),output_root=tmp_path,ab_run_id="bad")


def test_recall_mrr_and_ndcg_are_document_level_and_correct():
    metrics=retrieval_metrics(["noise","D2","D1"],["D1","D2"])
    assert metrics["recall_at_1"]==0 and metrics["recall_at_3"]==1
    assert metrics["mrr_at_10"]==0.5 and metrics["ndcg_at_10"]>0
    assert all(value is None for value in retrieval_metrics(["x"],[]).values())


def test_document_ndcg_deduplicates_multiple_chunks_from_same_document():
    metrics=retrieval_metrics(["D1","D1","D1","D2"],["D1","D2"])
    assert metrics["recall_at_3"]==1.0
    assert metrics["ndcg_at_10"]==1.0


def test_metrics_beyond_configured_top_k_are_na_not_fabricated():
    metrics=retrieval_metrics(["D1","D2"],["D1","D2"],available_k=8)
    assert metrics["recall_at_5"]==1.0
    assert metrics["recall_at_10"] is None
    assert metrics["mrr_at_10"] is None
    assert metrics["ndcg_at_10"] is None


def test_percentile_interpolates_p50_and_p95():
    assert percentile([1,2,3,4],.5)==2.5
    assert percentile([1,2,3,4],.95)==3.85


def test_graph_metrics_require_exact_predicate_direction_and_provenance():
    expected=(("A","PROVIDES_INDEX_TO","B","forward"),)
    wrong=graph_metrics(mode="graph_only",edges=(("A","DEPENDS_ON","B","forward"),),expected_edges=expected,expected_path=expected,graph_ids=("g",),final_ids=("g",),provenance=({"document_id":"d","source":"d.txt"},))
    assert wrong["expected_relation_edge_recall"]==0 and wrong["predicate_fidelity"]==0
    correct=graph_metrics(mode="graph_only",edges=expected,expected_edges=expected,expected_path=expected,graph_ids=("g",),final_ids=("g",),provenance=({"document_id":"d","source":"d.txt"},))
    assert correct["expected_relation_path_recall"]==1 and correct["direction_fidelity"]==1 and correct["provenance_coverage"]==1


@pytest.mark.parametrize("mode",["vector_only","bm25_only"])
def test_non_graph_modes_emit_na_for_every_graph_metric(mode):
    metrics=graph_metrics(mode=mode,edges=(),expected_edges=(),expected_path=(),graph_ids=(),final_ids=(),provenance=())
    assert metrics and all(value is None for value in metrics.values())


def test_zero_result_and_mode_failure_are_distinct(tmp_path):
    class Empty(DeterministicOfflineABAdapter):
        async def retrieve(self,**kwargs):
            if kwargs["mode"]=="graph_only": raise RuntimeError("broken")
            result=await super().retrieve(**kwargs); result["document_ranks"]=[]; result["candidate_ids"]=[]; return result
    async def run():
        frozen=inputs(); result=await HybridV2ABRunner(inputs=frozen,adapter=Empty(frozen.cases),output_root=tmp_path,ab_run_id="fail-v-zero",modes=("vector_only","graph_only")).run()
        vector=next(row for row in result["payload"]["results"] if row["retrieval_mode"]=="vector_only")
        graph=next(row for row in result["payload"]["results"] if row["retrieval_mode"]=="graph_only")
        assert vector["success"] and vector["zero_result"]
        assert not graph["success"] and not graph["zero_result"] and graph["failure_summary"]=="RuntimeError"
    asyncio.run(run())


def test_source_coverage_and_component_counts_are_aggregated(tmp_path):
    async def run():
        frozen=inputs(); result=await HybridV2ABRunner(inputs=frozen,adapter=DeterministicOfflineABAdapter(frozen.cases),output_root=tmp_path,ab_run_id="summary",modes=("vector_only",)).run()
        item=result["summary"]["modes"]["vector_only"]
        assert item["source_coverage"]==1 and item["component_calls"]["vector"]==180
        assert item["component_calls"]["neo4j"]==0 and item["answer_accuracy"] is None
    asyncio.run(run())


def test_stable_result_hash_and_candidate_id_tie_break(tmp_path):
    async def once(name):
        frozen=inputs(); result=await HybridV2ABRunner(inputs=frozen,adapter=DeterministicOfflineABAdapter(frozen.cases),output_root=tmp_path,ab_run_id=name).run()
        rows=result["payload"]["results"]
        assert all(row["document_ranks"]==sorted(row["document_ranks"],key=lambda value:(value not in next(case for case in frozen.cases if case.question_id==row["question_id"]).relevant_documents,value)) for row in rows)
        return [row["result_hash"] for row in rows]
    first=asyncio.run(once("stable-a")); second=asyncio.run(once("stable-b"))
    assert first!=second  # run identity is intentionally part of each result hash


def test_outputs_are_complete_parseable_and_contain_no_sensitive_or_absolute_values(tmp_path):
    async def run():
        frozen=inputs(); directory=tmp_path/"safe"
        await HybridV2ABRunner(inputs=frozen,adapter=DeterministicOfflineABAdapter(frozen.cases),output_root=tmp_path,ab_run_id="safe").run()
        assert {path.name for path in directory.iterdir()}=={"results.json","results.csv","summary.md","metric-definitions.json","safe-run-metadata.json","input-hashes.json"}
        assert len(json.loads((directory/"results.json").read_text())["results"])==720
        combined="\n".join(path.read_text(encoding="utf-8") for path in directory.iterdir())
        assert "D:\\" not in combined and "API_KEY" not in combined and "Authorization" not in combined
    asyncio.run(run())


def test_existing_output_is_never_overwritten(tmp_path):
    async def run():
        frozen=inputs(); runner=lambda:HybridV2ABRunner(inputs=frozen,adapter=DeterministicOfflineABAdapter(frozen.cases),output_root=tmp_path,ab_run_id="immutable",modes=("vector_only",))
        await runner().run()
        before=(tmp_path/"immutable"/"results.json").read_bytes()
        with pytest.raises(ValueError,match="ab_output_directory_exists"): await runner().run()
        assert (tmp_path/"immutable"/"results.json").read_bytes()==before
    asyncio.run(run())


def test_real_cli_requires_both_explicit_authorizations():
    missing_retrieval=subprocess.run([sys.executable,str(SCRIPT),"--runtime","real","--run-id","g2-real-ingestion-20260920-02","--ab-run-id","no-run"],capture_output=True,text=True)
    assert missing_retrieval.returncode==2 and "--allow-real-retrieval" in missing_retrieval.stderr
    missing_snapshot=subprocess.run([sys.executable,str(SCRIPT),"--runtime","real","--run-id","g2-real-ingestion-20260920-02","--ab-run-id","no-run","--allow-real-retrieval"],capture_output=True,text=True)
    assert missing_snapshot.returncode==2 and "--allow-frozen-snapshot" in missing_snapshot.stderr


def test_fake_runner_never_imports_provider_or_database_clients(tmp_path,monkeypatch):
    before=set(sys.modules)
    async def run():
        frozen=inputs(); await HybridV2ABRunner(inputs=frozen,adapter=DeterministicOfflineABAdapter(frozen.cases),output_root=tmp_path,ab_run_id="no-io",modes=("bm25_only",)).run()
    asyncio.run(run()); added=set(sys.modules)-before
    assert not any(name.startswith(("pymongo","neo4j","chromadb","openai")) for name in added)


def test_metric_summary_uses_na_not_zero_for_unavailable_answer_and_graph_values():
    row={"retrieval_mode":"vector_only","success":True,"zero_result":False,"retrieval_metrics":retrieval_metrics(["D"],["D"]),"source_coverage":1.0,"latency_ms":1.0,"call_audit":{},"graph_metrics":{key:None for key in ("graph_evidence_participation_rate","graph_evidence_final_top_k_rate","expected_relation_edge_recall","expected_relation_path_recall","direction_fidelity","predicate_fidelity","provenance_coverage")}}
    summary=summarize([row])["modes"]["vector_only"]
    assert summary["answer_accuracy"] is None and summary["expected_relation_edge_recall"] is None


def test_hosted_ab_runtime_uses_frozen_snapshot_source_run_not_report_run():
    snapshot=SimpleNamespace(payload={"run_id":"frozen-source-run"})
    settings=SimpleNamespace(
        embedding_config=SimpleNamespace(provider="qwen",model="embedding"),
        embedding_dimensions=1024,resolved_embedding_space_id="qwen:embedding:1024",
    )
    components={name:object() for name in (
        "DocumentUpdateCoordinator","HybridRetrieverV2","RRFFusion","ContextBuilderV2",
    )}
    runtime=build_hosted_real_runtime(SimpleNamespace(settings=settings,components=components),
                                      trace_run_id="new-ab-report-run",
                                      embedding_snapshot=snapshot,query_plans_hash="plans")
    assert runtime._run_id=="frozen-source-run"
    assert runtime._production_adapter_factory()._trace_run_id=="new-ab-report-run"


def test_real_summary_is_not_mislabeled_as_offline_or_fake_only():
    rendered=summary_markdown({"modes":{}},{"offline":False,"result_count":0})
    assert "real read-only" in rendered
    assert "Fake-only" not in rendered
