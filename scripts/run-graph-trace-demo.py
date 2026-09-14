"""Write five fake-only S4.6a first-loss scenarios without services or models."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = PROJECT_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from services.graph_trace_diagnosis import diagnose_first_loss  # noqa: E402


EXPECTED = [("北极星", "PROVIDES_INDEX", "天枢", "forward")]
EDGE = {"subject": "北极星", "predicate": "PROVIDES_INDEX", "object": "天枢", "direction": "forward"}
STAGES = ("extracted", "normalized", "persisted", "retrieved_raw", "scope_accepted", "relevance_scored", "entered_final_top_k", "entered_prompt")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _snapshots(*, empty_stage: str | None = None) -> dict[str, list[dict[str, str]]]:
    missing = empty_stage is not None
    snapshots: dict[str, list[dict[str, str]]] = {}
    for stage in STAGES:
        if stage == empty_stage:
            missing = True
        snapshots[stage] = [] if missing else [dict(EDGE)]
    return snapshots


def main() -> int:
    output = PROJECT_ROOT / ".runtime" / "evaluation" / "s4-6a-trace-offline" / "trace-demo.json"
    scenarios = {
        "extraction_missing": {"stages": {"extracted": []}},
        "persistence_missing": {"stages": _snapshots(empty_stage="persisted")},
        "retrieval_missing": {"stages": _snapshots(empty_stage="retrieved_raw")},
        "top_k_truncated": {"stages": _snapshots(empty_stage="entered_final_top_k")},
        "prompt_present": {"stages": _snapshots()},
    }
    payload = {
        "notice": "fake-only trace verification; not real GraphRAG quality evidence",
        "scenarios": {name: diagnose_first_loss(EXPECTED, trace) for name, trace in scenarios.items()},
    }
    _atomic_json(output, payload)
    print(f"fake-only graph trace demo completed: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
