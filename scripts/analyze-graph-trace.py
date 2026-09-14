"""Analyze an existing GraphRAG trace without providers, APIs, or databases."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = PROJECT_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from services.graph_trace_analysis import GraphTraceAnalysisError, analyze_trace_run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline first-loss analysis for a persisted GraphRAG trace")
    parser.add_argument("--trace-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--benchmark-dir", type=Path, default=PROJECT_ROOT / "benchmarks" / "enterprise_20docs_60q_expanded")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    try:
        result = analyze_trace_run(args.trace_dir, args.output_dir, args.benchmark_dir, strict=args.strict)
    except GraphTraceAnalysisError as exc:
        print(f"graph trace analysis refused: {exc}", file=sys.stderr)
        return 2
    print(f"graph trace analysis completed: {result['run_id']} ({len(result['expected_edges'])} expected edges)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
