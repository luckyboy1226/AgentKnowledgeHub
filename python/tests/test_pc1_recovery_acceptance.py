import json
import subprocess
import sys
from pathlib import Path

from services.pc1_recovery_acceptance import inspect_original_recovery


COMMAND = (
    r"D:\Anaconda3\envs\kghub\python.exe scripts\run-parent-child-pc1.py "
    "--run-id pc1-flood-longdoc-v3-20260921-01 --step recover-d04 "
    "--runtime real --allow-real-io --allow-d04-recovery"
)
SCRIPT = Path(__file__).resolve().parents[2] / 'scripts' / 'run-parent-child-pc1.py'


def inspect(tmp_path: Path, *, stderr="", exit_code=None, progress=None, active=()):
    stdout_path=tmp_path/'stdout.log'; stderr_path=tmp_path/'stderr.log'
    state_path=tmp_path/'state.json'; progress_path=tmp_path/'progress.json'; exit_path=tmp_path/'exit.txt'
    stdout_path.write_text('',encoding='utf-8'); stderr_path.write_text(stderr,encoding='utf-8')
    state_path.write_text(json.dumps({'status':'FAILED'}),encoding='utf-8')
    progress_path.write_text(json.dumps(progress or {'documents':[]}),encoding='utf-8')
    if exit_code is not None: exit_path.write_text(str(exit_code),encoding='utf-8')
    return inspect_original_recovery(
        run_id='pc1-flood-longdoc-v3-20260921-01',command=COMMAND,
        stdout_path=stdout_path,stderr_path=stderr_path,state_path=state_path,
        progress_path=progress_path,exit_code_path=exit_path,active_processes=active,
    )


def test_missing_exit_code_is_unrecoverable_and_never_guessed(tmp_path):
    result=inspect(tmp_path,stderr='Traceback (most recent call last):\nRuntimeError: graph mismatch\n')
    assert result['outcome']=='BLOCKED'
    assert result['exit_code'] is None and result['exit_code_status']=='unrecoverable'
    assert 'stderr_traceback' in result['reasons']
    assert result['safety']['production_clients_constructed'] is False


def test_active_original_process_is_waiting_not_blocked(tmp_path):
    result=inspect(tmp_path,active=({'pid':42,'name':'python.exe','command':'exact'},))
    assert result['outcome']=='WAITING'
    assert result['reasons']==['original_recovery_process_still_running']


def test_failed_state_or_missing_d04_cannot_pass(tmp_path):
    progress={'documents':[{'logical_key':key,'status':'ready'} for key in ('D01','D02','D03')]}
    result=inspect(tmp_path,exit_code=0,progress=progress)
    assert result['outcome']=='BLOCKED'
    assert 'runner_state_not_completed' in result['reasons']
    assert 'd04_progress_missing_or_duplicate' in result['reasons']


def test_file_only_success_still_requires_storage_acceptance(tmp_path):
    stdout_path=tmp_path/'stdout.log'; stderr_path=tmp_path/'stderr.log'
    state_path=tmp_path/'state.json'; progress_path=tmp_path/'progress.json'; exit_path=tmp_path/'exit.txt'
    stdout_path.write_text('',encoding='utf-8'); stderr_path.write_text('',encoding='utf-8')
    state_path.write_text(json.dumps({'status':'COMPLETED'}),encoding='utf-8')
    progress_path.write_text(json.dumps({'documents':[{
        'logical_key':'D04','document_id':'90eec68b-7560-4181-99d4-c74a4390f4a0',
        'document_version':1,'status':'ready'}]}),encoding='utf-8')
    exit_path.write_text('0',encoding='utf-8')
    result=inspect_original_recovery(
        run_id='pc1-flood-longdoc-v3-20260921-01',command=COMMAND,
        stdout_path=stdout_path,stderr_path=stderr_path,state_path=state_path,
        progress_path=progress_path,exit_code_path=exit_path,active_processes=(),
    )
    assert result['outcome']=='NEEDS_READ_ONLY_STORAGE_ACCEPTANCE'
    assert result['storage_acceptance_performed'] is False


def test_inspection_cli_requires_real_runtime_and_explicit_authorization():
    fake=subprocess.run([sys.executable,str(SCRIPT),'--run-id','pc1-flood-longdoc-v3-20260921-01',
                         '--step','inspect-pc1r','--runtime','fake'],capture_output=True,text=True)
    assert fake.returncode==2 and 'requires --runtime real' in fake.stderr
    denied=subprocess.run([sys.executable,str(SCRIPT),'--run-id','pc1-flood-longdoc-v3-20260921-01',
                           '--step','inspect-pc1r','--runtime','real'],capture_output=True,text=True)
    assert denied.returncode==2 and 'requires --allow-pc1r-inspection' in denied.stderr


def test_graph_reconciliation_cli_has_independent_guard_and_preflight_binding():
    denied=subprocess.run([
        sys.executable,str(SCRIPT),'--run-id','pc1-flood-longdoc-v3-20260921-01',
        '--step','reconcile-d04-graph-stage','--runtime','real','--allow-real-io',
    ],capture_output=True,text=True)
    assert denied.returncode==2
    assert 'requires --allow-real-io and --allow-d04-graph-reconciliation' in denied.stderr
    wrong_run=subprocess.run([
        sys.executable,str(SCRIPT),'--run-id','wrong-run',
        '--step','reconcile-d04-graph-stage','--runtime','real','--allow-real-io',
        '--allow-d04-graph-reconciliation','--reconciliation-preflight-sha256','0'*64,
    ],capture_output=True,text=True)
    assert wrong_run.returncode==2 and 'exact authorized run-id' in wrong_run.stderr
