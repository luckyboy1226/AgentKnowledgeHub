"""Write fake-only S4.7b ingestion/QA trace scenarios; never uses services."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = PROJECT_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from services.graph_evidence_trace import EvaluationTraceJournal, GraphEvidenceTrace  # noqa: E402
from services.graph_trace_diagnosis import diagnose_first_loss  # noqa: E402


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _edge(document_id: str) -> dict[str, object]:
    return {
        "subject": "A", "predicate": "DEPENDS_ON", "raw_predicate": "依赖", "object": "B",
        "direction": "forward", "document_id": document_id, "document_version": 1,
        "source": "fake-fixture.txt", "evidence_key": "fake-evidence", "status": "ready",
        "is_current": True, "relation_semantics_version": "relation-semantics-v1",
    }


def _diagnose(root: Path, run_id: str, document_id: str, *, missing: str | None = None) -> str:
    operation_id = str(uuid4())
    journal = EvaluationTraceJournal(root=root, run_id=run_id, operation_id=operation_id, fixture_id="fake")
    edge = _edge(document_id)
    for stage in ("extracted", "normalized", "persisted"):
        journal.record_stage(
            stage, document_id=document_id, document_version=1,
            source="fake-fixture.txt", status="processing", is_current=False,
        )
        if stage != missing:
            journal.record_edges(stage, [edge])
    trace = GraphEvidenceTrace(run_id=run_id, question_id=f"Q-{missing or 'complete'}", scope_verified=True, allowed_document_ids_count=1)
    EvaluationTraceJournal.merge_into_question_trace(trace, root=root, run_id=run_id, allowed_document_ids=frozenset({document_id}))
    for stage in ("retrieved_raw", "scope_accepted", "relevance_scored", "ranked", "entered_final_top_k", "entered_prompt"):
        if stage != missing:
            trace.record_edges(stage, [edge])
        else:
            # QA-stage snapshots are in-memory.  An explicit empty list mirrors
            # the ingestion journal marker and proves the stage was observed.
            trace.question.stages.setdefault(stage, [])
    return diagnose_first_loss([("A", "DEPENDS_ON", "B", "forward")], trace.to_dict())["edge_diagnoses"][0]["first_loss_stage"]


def main() -> int:
    output = PROJECT_ROOT / ".runtime" / "evaluation" / "s4-7b-trace-journal-offline"
    document_id = str(uuid4())
    scenarios = {
        # v2 keeps a previous fake journal immutable if this demo is rerun after
        # its schema changes.  Real run IDs are likewise never reused.
        "lost_at_extraction": _diagnose(output, "s4-7b-demo-extraction-v2", document_id, missing="extracted"),
        "lost_at_normalization": _diagnose(output, "s4-7b-demo-normalized-v2", document_id, missing="normalized"),
        "lost_at_persistence": _diagnose(output, "s4-7b-demo-persisted-v2", document_id, missing="persisted"),
        "lost_at_retrieval": _diagnose(output, "s4-7b-demo-retrieval-v2", document_id, missing="retrieved_raw"),
        "entered_prompt": _diagnose(output, "s4-7b-demo-complete-v2", document_id),
    }
    _atomic_json(output / "trace-journal-demo.json", {
        "notice": "fake-only trace journal verification; not real GraphRAG quality evidence",
        "scenarios": scenarios,
    })
    print(f"fake-only trace journal demo completed: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
