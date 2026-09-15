"""Run the S4.8a offline-only relation gap analyser."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from services.graph_trace_gap_analysis import GraphTraceGapAnalysisError, analyze_trace_gaps  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline, immutable S4.8a relation-gap analysis")
    parser.add_argument("--trace-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--benchmark-dir", type=Path, default=ROOT / "benchmarks" / "enterprise_20docs_60q_expanded")
    args = parser.parse_args()
    try:
        result = analyze_trace_gaps(args.trace_dir, args.output_dir, args.benchmark_dir)
    except GraphTraceGapAnalysisError as exc:
        print(f"graph trace gap analysis refused: {exc}", file=sys.stderr)
        return 2
    print(f"graph trace gap analysis completed: {result['run_id']} ({len(result['matrix'])} extraction targets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
