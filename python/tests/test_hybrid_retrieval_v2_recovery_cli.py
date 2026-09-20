import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run-hybrid-v2-real-eval.py"
RUN_ID = "g2-real-ingestion-20260920-02"


def test_recovery_cli_requires_real_runtime_and_separate_authorization():
    fake = subprocess.run(
        [sys.executable, str(SCRIPT), "--runtime", "fake", "--run-id", RUN_ID,
         "--step", "recover-ingestion-document"],
        capture_output=True, text=True,
    )
    assert fake.returncode == 2
    assert "requires --runtime real" in fake.stderr

    missing_authorization = subprocess.run(
        [sys.executable, str(SCRIPT), "--runtime", "real", "--run-id", RUN_ID,
         "--step", "recover-ingestion-document"],
        capture_output=True, text=True,
    )
    assert missing_authorization.returncode == 2
    assert "requires --allow-recovery-record" in missing_authorization.stderr


def test_g3_cli_requires_real_runtime_and_separate_llm_authorization():
    fake = subprocess.run(
        [sys.executable, str(SCRIPT), "--runtime", "fake", "--run-id", RUN_ID,
         "--step", "build-query-plans"], capture_output=True, text=True,
    )
    assert fake.returncode == 2
    assert "requires --runtime real" in fake.stderr

    missing_authorization = subprocess.run(
        [sys.executable, str(SCRIPT), "--runtime", "real", "--run-id", RUN_ID,
         "--step", "build-query-plans"], capture_output=True, text=True,
    )
    assert missing_authorization.returncode == 2
    assert "requires --allow-query-plan-llm" in missing_authorization.stderr


def test_g4_cli_requires_real_runtime_and_separate_retrieval_authorization():
    fake = subprocess.run(
        [sys.executable, str(SCRIPT), "--runtime", "fake", "--run-id", RUN_ID,
         "--step", "trace-equivalence"], capture_output=True, text=True,
    )
    assert fake.returncode == 2
    assert "requires --runtime real" in fake.stderr

    missing_authorization = subprocess.run(
        [sys.executable, str(SCRIPT), "--runtime", "real", "--run-id", RUN_ID,
         "--step", "trace-equivalence"], capture_output=True, text=True,
    )
    assert missing_authorization.returncode == 2
    assert "requires --allow-real-retrieval" in missing_authorization.stderr


def test_g4_resume_cli_requires_real_runtime_and_separate_authorization():
    fake = subprocess.run(
        [sys.executable, str(SCRIPT), "--runtime", "fake", "--run-id", RUN_ID,
         "--step", "resume-trace-equivalence"], capture_output=True, text=True,
    )
    assert fake.returncode == 2
    assert "requires --runtime real" in fake.stderr

    missing_authorization = subprocess.run(
        [sys.executable, str(SCRIPT), "--runtime", "real", "--run-id", RUN_ID,
         "--step", "resume-trace-equivalence"], capture_output=True, text=True,
    )
    assert missing_authorization.returncode == 2
    assert "requires --allow-g4-resume" in missing_authorization.stderr

    missing_recovery_id = subprocess.run(
        [sys.executable, str(SCRIPT), "--runtime", "real", "--run-id", RUN_ID,
         "--step", "resume-trace-equivalence", "--allow-g4-resume"],
        capture_output=True, text=True,
    )
    assert missing_recovery_id.returncode == 2
    assert "requires --recovery-id" in missing_recovery_id.stderr


def test_g4_recovery_execution_requires_both_authorizations():
    missing_resume_authorization = subprocess.run(
        [sys.executable, str(SCRIPT), "--runtime", "real", "--run-id", RUN_ID,
         "--recovery-id", "g4-recovery-test", "--step", "trace-equivalence",
         "--allow-real-retrieval"], capture_output=True, text=True,
    )
    assert missing_resume_authorization.returncode == 2
    assert "requires --allow-g4-resume" in missing_resume_authorization.stderr
