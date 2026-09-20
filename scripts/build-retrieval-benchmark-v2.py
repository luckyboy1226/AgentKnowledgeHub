"""Materialize the independent, reviewable Phase G0 retrieval benchmark."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from services.retrieval_benchmark_v2 import build_retrieval_benchmark_v2  # noqa: E402


if __name__ == "__main__":
    manifest = build_retrieval_benchmark_v2(
        ROOT / "benchmarks" / "enterprise_20docs_60q_expanded",
        ROOT / "benchmarks" / "enterprise_20docs_retrieval_v2",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
