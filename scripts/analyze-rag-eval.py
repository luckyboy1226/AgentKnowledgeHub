"""Create an immutable, offline-only diagnosis of a persisted RAG evaluation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = PROJECT_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from services.rag_evaluation_diagnosis import DIAGNOSIS_VERSION, DiagnosisError, run_diagnosis


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline deterministic RAG evaluation diagnosis")
    parser.add_argument("--results", required=True, help="Persisted results.json to read")
    parser.add_argument("--benchmark-dir", required=True, help="Immutable reviewed fixture directory")
    parser.add_argument("--output-dir", required=True, help="New directory for derived reports")
    parser.add_argument("--diagnosis-version", default=DIAGNOSIS_VERSION)
    parser.add_argument("--strict", action="store_true", help="Require exactly one result for each fixture question/mode")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output = run_diagnosis(
            args.results,
            args.benchmark_dir,
            args.output_dir,
            diagnosis_version=args.diagnosis_version,
            strict=args.strict,
        )
    except DiagnosisError as exc:
        print(f"diagnosis refused: {exc}", file=sys.stderr)
        return 1
    # Deliberately omit model answers, prompt text, paths, and service configuration.
    print(f"diagnosis complete: run_id={output['run_id']} rows={len(output['results'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
