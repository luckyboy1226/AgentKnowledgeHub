"""Read-only acceptance evidence for the original PC-1R D04 recovery."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


EXPECTED_D04 = {
    "logical_key": "D04",
    "document_id": "90eec68b-7560-4181-99d4-c74a4390f4a0",
    "document_version": 1,
    "original_operation_id": "addb1b99-c332-4698-ad1f-2b9eb35bcd11",
}


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _exit_code(path: Path) -> int | None:
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    try:
        return int(value)
    except ValueError:
        return None


def inspect_original_recovery(
    *,
    run_id: str,
    command: str,
    stdout_path: Path,
    stderr_path: Path,
    state_path: Path,
    progress_path: Path,
    exit_code_path: Path,
    active_processes: Iterable[dict[str, Any]],
    process_scan_errors: int = 0,
) -> dict[str, Any]:
    """Classify existing evidence without contacting any production service."""
    active = [dict(row) for row in active_processes]
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.is_file() else ""
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.is_file() else ""
    state = _json(state_path, {})
    progress = _json(progress_path, {"documents": []})
    documents = progress.get("documents", []) if isinstance(progress, dict) else []
    d04 = [row for row in documents if isinstance(row, dict) and row.get("logical_key") == "D04"]
    exit_code = _exit_code(exit_code_path)
    traceback = "Traceback (most recent call last):" in stderr
    terminal_error = next(
        (line.strip() for line in reversed(stderr.splitlines()) if line.strip()),
        None,
    )
    reasons: list[str] = []
    if active:
        outcome = "WAITING"
        reasons.append("original_recovery_process_still_running")
    else:
        if exit_code is None:
            reasons.append("exit_code_unrecoverable")
        elif exit_code != 0:
            reasons.append("nonzero_exit_code")
        if traceback:
            reasons.append("stderr_traceback")
        if state.get("status") != "COMPLETED":
            reasons.append("runner_state_not_completed")
        if len(d04) != 1:
            reasons.append("d04_progress_missing_or_duplicate")
        elif (
            str(d04[0].get("document_id")) != EXPECTED_D04["document_id"]
            or int(d04[0].get("document_version") or 0) != EXPECTED_D04["document_version"]
            or d04[0].get("status") != "ready"
        ):
            reasons.append("d04_progress_identity_invalid")
        # A successful command would still require the separately authorized
        # storage acceptance checklist. This file-only gate may never infer it.
        outcome = "BLOCKED" if reasons else "NEEDS_READ_ONLY_STORAGE_ACCEPTANCE"
    return {
        "schema_version": "pc1r-original-acceptance-v1",
        "run_id": str(run_id),
        "outcome": outcome,
        "original_command": str(command),
        "active_processes": active,
        "process_scan_errors": int(process_scan_errors),
        "exit_code": exit_code,
        "exit_code_status": "recorded" if exit_code is not None else "unrecoverable",
        "stdout_empty": stdout == "",
        "stderr_has_traceback": traceback,
        "terminal_error": terminal_error,
        "runner_state": state.get("status"),
        "progress_logical_keys": [str(row.get("logical_key") or "") for row in documents if isinstance(row, dict)],
        "d04_progress_records": len(d04),
        "storage_acceptance_performed": False,
        "reasons": reasons,
        "evidence": {
            "stdout": {"path": str(stdout_path.resolve()), "sha256": _sha256(stdout_path)},
            "stderr": {"path": str(stderr_path.resolve()), "sha256": _sha256(stderr_path)},
            "state": {"path": str(state_path.resolve()), "sha256": _sha256(state_path)},
            "progress": {"path": str(progress_path.resolve()), "sha256": _sha256(progress_path)},
            "exit_code_sentinel": {"path": str(exit_code_path.resolve()), "sha256": _sha256(exit_code_path)},
        },
        "safety": {
            "production_clients_constructed": False,
            "recovery_started": False,
            "storage_write_performed": False,
            "d05_d20_processed": False,
            "cleanup_executed": False,
        },
    }
