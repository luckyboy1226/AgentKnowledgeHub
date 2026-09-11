"""Fake-only contracts for the S4 Vector RAG versus GraphRAG evaluation."""

from __future__ import annotations

import asyncio
import csv
import importlib.util
import json
import socket
from pathlib import Path

import pytest

from agents.qa_agent import RetrievalMode
from services.rag_evaluation import (
    DeterministicScorer,
    EVALUATION_CASES,
    RAGEvaluationRunner,
    build_offline_runner,
    safe_source,
)


def _case(question_id: str):
    return next(case for case in EVALUATION_CASES if case.question_id == question_id)


@pytest.mark.asyncio
async def test_vector_only_skips_graph_and_keeps_graph_context_out_of_prompt():
    runner = build_offline_runner()
    case = _case("Q03")
    plan = await runner.agent.build_evaluation_query_plan(
        case.question, run_id="unit", question_id=case.question_id
    )

    result = await runner.agent.answer_with_evaluation_plan(
        plan, RetrievalMode.VECTOR_ONLY, scope=runner.scope
    )

    assert result.contexts and all(context.retrieval_type != "graph" for context in result.contexts)
    assert runner.agent.knowledge_graph.calls == 0
    assert runner.agent.vector_store.embeddings.query_calls == 1
    assert "类型: graph" not in runner.agent.llm.final_prompts[-1]


@pytest.mark.asyncio
async def test_graph_mode_calls_graph_and_records_graph_context():
    runner = build_offline_runner()
    case = _case("Q03")
    plan = await runner.agent.build_evaluation_query_plan(
        case.question, run_id="unit", question_id=case.question_id
    )

    result = await runner.agent.answer_with_evaluation_plan(
        plan, RetrievalMode.GRAPH_RAG, scope=runner.scope
    )

    assert runner.agent.knowledge_graph.calls > 0
    assert any(context.retrieval_type == "graph" for context in result.contexts)
    assert "类型: graph" in runner.agent.llm.final_prompts[-1]


@pytest.mark.asyncio
async def test_graph_mode_result_includes_safe_graph_sources(tmp_path):
    payload = await build_offline_runner().run(
        run_id="graph-sources", modes=("graph_rag",), output_root=tmp_path
    )

    graph_sources = [
        source
        for row in payload["results"]
        for source in row["sources"]
        if source["retrieval_type"] == "graph"
    ]
    assert graph_sources
    assert all(source["document_id"] in build_offline_runner().scope.allowed_document_ids for source in graph_sources)
    assert all("/" not in source["source"] and "\\" not in source["source"] for source in graph_sources)


@pytest.mark.asyncio
async def test_runner_builds_one_shared_plan_per_question_for_both_modes(tmp_path):
    runner = build_offline_runner()

    payload = await runner.run(
        run_id="shared-plan", modes=(RetrievalMode.VECTOR_ONLY, RetrievalMode.GRAPH_RAG), output_root=tmp_path
    )

    assert len(payload["results"]) == len(EVALUATION_CASES) * 2
    # One intent + one rewrite per case, then one final answer per mode.
    assert runner.agent.llm.call_count == len(EVALUATION_CASES) * 4
    assert all(row["model_call_counts"]["shared_preprocess_chat"] == 2 for row in payload["results"])
    assert all(row["model_call_counts"]["embedding_query"] == 1 for row in payload["results"])


@pytest.mark.asyncio
async def test_both_modes_use_same_memoryless_final_answer_configuration(tmp_path):
    runner = build_offline_runner()
    await runner.run(run_id="same-config", modes=("vector_only", "graph_rag"), output_root=tmp_path)

    prompts = runner.agent.llm.final_prompts
    assert len(prompts) == len(EVALUATION_CASES) * 2
    assert all("专业的企业知识问答助手" in prompt for prompt in prompts)
    assert all("近期对话（短期记忆）" not in prompt for prompt in prompts)


@pytest.mark.asyncio
async def test_evaluation_rejects_memory_or_checkpoint_participation():
    runner = build_offline_runner()
    runner.agent.memory_service = object()
    with pytest.raises(ValueError, match="memoryless"):
        await runner.agent.answer_with_evaluation_plan(
            await runner.agent.build_evaluation_query_plan(
                _case("Q01").question, run_id="unit", question_id="Q01"
            ),
            "vector_only",
        )
    with pytest.raises(ValueError, match="memory_service"):
        RAGEvaluationRunner(runner.agent)


def test_deterministic_keyword_and_source_scoring():
    case = _case("Q01")
    score = DeterministicScorer.score(
        case,
        answer="陈航是数据平台部的数据工程师。",
        sources=[{"source": "s4-eval-org.txt"}],
        mode=RetrievalMode.VECTOR_ONLY,
        graph_context_count=0,
    )
    assert score["question_correct"] is True
    assert score["source_hit"] is True


def test_deterministic_abstention_and_graph_participation_scoring():
    unanswered = DeterministicScorer.score(
        _case("Q11"), answer="资料未提供，无法回答。", sources=[],
        mode=RetrievalMode.GRAPH_RAG, graph_context_count=0,
    )
    missing_graph = DeterministicScorer.score(
        _case("Q03"), answer="Atlas 服务由赵启负责。", sources=[{"source": "s4-eval-projects"}],
        mode=RetrievalMode.GRAPH_RAG, graph_context_count=0,
    )
    assert unanswered["question_correct"] is True
    assert missing_graph["question_correct"] is False


@pytest.mark.asyncio
async def test_runner_writes_parseable_json_csv_and_markdown_reports(tmp_path):
    payload = await build_offline_runner().run(
        run_id="reports", modes=("vector_only", "graph_rag"), output_root=tmp_path
    )
    output = tmp_path / "reports"

    assert json.loads((output / "results.json").read_text(encoding="utf-8"))["run_id"] == "reports"
    assert len(list(csv.DictReader((output / "results.csv").open(encoding="utf-8")))) == len(payload["results"])
    assert "Fake-only" in (output / "summary.md").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_offline_runner_cannot_access_network_or_real_services(tmp_path, monkeypatch):
    def network_forbidden(*_args, **_kwargs):
        raise AssertionError("offline runner attempted network access")

    monkeypatch.setattr(socket, "create_connection", network_forbidden)
    payload = await build_offline_runner().run(
        run_id="no-network", modes=("vector_only",), output_root=tmp_path
    )
    assert payload["offline"] is True
    assert payload["summary"]["vector_only"]["questions"] == 12


def test_result_output_and_source_sanitization_exclude_paths_and_credentials(tmp_path):
    unsafe_path = f"{chr(68)}:{chr(92)}private{chr(92)}secrets{chr(92)}source.txt"
    assert safe_source(unsafe_path) == "source.txt"
    payload = asyncio.run(build_offline_runner().run(
        run_id="safe-output", modes=("graph_rag",), output_root=tmp_path
    ))
    serialized = json.dumps(payload, ensure_ascii=False)
    assert f"{chr(68)}:{chr(92)}" not in serialized
    assert "Authorization" not in serialized
    assert "api_key" not in serialized.casefold()


def _load_cli_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "run-rag-eval.py"
    spec = importlib.util.spec_from_file_location("run_rag_eval", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_cli_mode_is_explicitly_refused(capsys):
    module = _load_cli_module()
    with pytest.raises(SystemExit) as exit_info:
        module.main(["--mode", "both"])
    assert exit_info.value.code == 2
    assert "real evaluation is refused" in capsys.readouterr().err


def test_offline_cli_uses_only_safe_run_identifier(tmp_path, monkeypatch):
    module = _load_cli_module()
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    assert module.main(["--offline", "--mode", "vector_only", "--run-id", "cli-safe"]) == 0
    assert (tmp_path / ".runtime" / "evaluation" / "cli-safe" / "results.json").exists()
    assert module.main(["--offline", "--run-id", "../unsafe"]) == 1


def test_fixture_contains_exactly_twelve_questions_with_required_categories():
    assert len(EVALUATION_CASES) == 12
    categories = {case.category for case in EVALUATION_CASES}
    assert {"single_hop", "multi_hop", "constraint", "unanswerable"}.issubset(categories)
