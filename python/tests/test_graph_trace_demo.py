"""The trace demo remains fake-only and writes the required safe label."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def test_fake_trace_demo_writes_all_five_first_loss_scenarios(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[2] / "scripts" / "run-graph-trace-demo.py"
    spec = importlib.util.spec_from_file_location("graph_trace_demo", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    assert module.main() == 0
    payload = json.loads((tmp_path / ".runtime" / "evaluation" / "s4-6a-trace-offline" / "trace-demo.json").read_text(encoding="utf-8"))
    assert payload["notice"] == "fake-only trace verification; not real GraphRAG quality evidence"
    assert set(payload["scenarios"]) == {"extraction_missing", "persistence_missing", "retrieval_missing", "top_k_truncated", "prompt_present"}
