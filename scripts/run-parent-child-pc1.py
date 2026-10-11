"""PC-1 controlled ingestion for the frozen flood long-document benchmark."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BENCHMARK=ROOT/'benchmarks'/'flood_emergency_20docs_60q_longdoc_v3_FINAL'/'flood_emergency_20docs_60q_longdoc_v3_final'
ARTIFACT_ROOT=ROOT/'.runtime'/'parent-child-pc1'
sys.path.insert(0,str(ROOT/'python')); os.chdir(ROOT/'python')
from services.hybrid_retrieval_v2_real_evaluation import ControlledRealEvaluationRunner, atomic_json
from services.hybrid_retrieval_v2_runtime import build_hosted_real_runtime
from services.pc1_recovery_acceptance import inspect_original_recovery


def sha(path:Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()
def canonical_hash(value:object)->str: return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(name:str)->dict: return json.loads((BENCHMARK/name).read_text(encoding='utf-8'))

def inventory()->dict:
    manifest,documents,questions,ground,pc_ground=(read(name) for name in ('benchmark_manifest.json','benchmark_documents.json','benchmark_questions.json','ground_truth.json','parent_child_ground_truth.json'))
    document_rows=documents.get('documents') or []; question_rows=questions.get('questions') or []
    files=[]
    for row in document_rows:
        key,filename=str(row.get('id') or ''),Path(str(row.get('filename') or '')).name
        path=BENCHMARK/'documents'/filename
        if not key or not filename or not path.is_file(): raise ValueError('pc1_document_inventory_invalid')
        files.append({'logical_key':key,'source':filename,'sha256':sha(path),'bytes':path.stat().st_size})
    evidence=sum(len(row.get('gold_evidence') or []) for row in (pc_ground.get('questions') or []))
    bundle=[]
    for path in sorted(item for item in BENCHMARK.rglob('*') if item.is_file()):
        bundle.append({'path':path.relative_to(BENCHMARK).as_posix(),'sha256':sha(path)})
    checks={'document_count_20':len(files)==20==int(manifest.get('document_count') or 0),'question_count_60':len(question_rows)==60==int(manifest.get('question_count') or 0),'gold_evidence_83':evidence==83==int(manifest.get('gold_evidence_count') or 0),'unique_document_keys':len({row['logical_key'] for row in files})==20,'unique_question_ids':len({str(row.get('question_id') or '') for row in question_rows})==60}
    if not all(checks.values()): raise ValueError('pc1_frozen_benchmark_inventory_failed')
    return {'benchmark_root':str(BENCHMARK),'benchmark_id':manifest.get('benchmark_id'),'benchmark_version':manifest.get('benchmark_version'),'document_files':files,'document_count':20,'question_count':60,'gold_evidence_count':83,'hashes':{'benchmark_manifest_sha256':sha(BENCHMARK/'benchmark_manifest.json'),'questions_sha256':sha(BENCHMARK/'benchmark_questions.json'),'ground_truth_sha256':sha(BENCHMARK/'ground_truth.json'),'parent_child_gold_evidence_sha256':sha(BENCHMARK/'parent_child_ground_truth.json'),'source_bundle_sha256':canonical_hash(bundle)},'checks':checks}

def configuration(settings)->dict:
    values={'parent_child_chunk_enabled':settings.parent_child_chunk_enabled,'parent_target_tokens':settings.parent_target_tokens,'parent_max_tokens':settings.parent_max_tokens,'child_target_tokens':settings.child_target_tokens,'child_overlap_tokens':settings.child_overlap_tokens,'token_estimator':'UnicodeTokenEstimator','reranker_enabled':False,'parent_child_expected':{'parent_count':3,'child_min':17,'child_max':18}}
    if values != {'parent_child_chunk_enabled':True,'parent_target_tokens':1000,'parent_max_tokens':1400,'child_target_tokens':250,'child_overlap_tokens':50,'token_estimator':'UnicodeTokenEstimator','reranker_enabled':False,'parent_child_expected':{'parent_count':3,'child_min':17,'child_max':18}}: raise ValueError('pc1_parent_child_configuration_mismatch')
    return values

def embedding(settings)->dict:
    value={'provider':settings.embedding_config.provider,'model':settings.embedding_config.model,'dimension':settings.embedding_dimensions,'embedding_space_id':settings.resolved_embedding_space_id}
    if value != {'provider':'qwen','model':'qwen3.7-text-embedding-flash','dimension':1024,'embedding_space_id':'qwen:qwen3.7-text-embedding-flash:1024'}: raise ValueError('pc1_embedding_configuration_mismatch')
    return value

def documents()->list[dict]:
    entries=[]
    for row in read('benchmark_documents.json').get('documents') or []:
        filename=Path(str(row.get('filename') or '')).name; path=BENCHMARK/'documents'/filename
        entries.append({'id':str(row.get('id') or ''),'filename':filename,'content':path.read_bytes()})
    if len(entries)!=20 or len({row['id'] for row in entries})!=20: raise ValueError('pc1_document_loader_invalid')
    return entries

def runner(run_id:str,settings): return ControlledRealEvaluationRunner(ARTIFACT_ROOT,run_id,BENCHMARK,configuration(settings))

def active_recovery_processes(run_id:str)->tuple[list[dict],int]:
    import psutil
    matches=[]; errors=0
    for process in psutil.process_iter(('pid','name','cmdline')):
        try:
            command=[str(value) for value in (process.info.get('cmdline') or [])]
            rendered=' '.join(command)
            if ('run-parent-child-pc1.py' in rendered and '--step recover-d04' in rendered
                    and f'--run-id {run_id}' in rendered):
                matches.append({'pid':int(process.info['pid']),'name':str(process.info.get('name') or ''),'command':rendered})
        except (psutil.AccessDenied,psutil.NoSuchProcess,psutil.ZombieProcess):
            errors+=1
    return matches,errors

def append_gate_history(path:Path,row:dict)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n')
        stream.flush(); os.fsync(stream.fileno())

def inspect_pc1r(run_id:str)->dict:
    if run_id!='pc1-flood-longdoc-v3-20260921-01': raise ValueError('pc1r_inspection_run_id_mismatch')
    run_dir=ARTIFACT_ROOT/run_id; log_dir=ROOT/'.runtime'/'pc1'
    command=(r'D:\Anaconda3\envs\kghub\python.exe scripts\run-parent-child-pc1.py '
             f'--run-id {run_id} --step recover-d04 --runtime real --allow-real-io --allow-d04-recovery')
    active,errors=active_recovery_processes(run_id)
    artifact=inspect_original_recovery(
        run_id=run_id,command=command,active_processes=active,process_scan_errors=errors,
        stdout_path=log_dir/'pc1r-recover.stdout.log',stderr_path=log_dir/'pc1r-recover.stderr.log',
        state_path=run_dir/'state.json',progress_path=run_dir/'ingestion-progress.json',
        exit_code_path=log_dir/'pc1r-recover.exitcode.txt',
    )
    atomic_json(run_dir/'pc1r-original-acceptance.json',artifact)
    append_gate_history(run_dir/'gate-history.jsonl',{'gate':'PC-1R-A','outcome':artifact['outcome'],'artifact':'pc1r-original-acceptance.json','artifact_sha256':sha(run_dir/'pc1r-original-acceptance.json')})
    return artifact

PC1R_RUN_ID='pc1-flood-longdoc-v3-20260921-01'
D04_ID='90eec68b-7560-4181-99d4-c74a4390f4a0'
D04_FAILED_OPERATION='addb1b99-c332-4698-ad1f-2b9eb35bcd11'

async def d01_d03_invariants(runtime,current_runner)->dict:
    progress=current_runner._read('ingestion-progress.json',{'documents':[]})
    rows=[row for row in progress.get('documents',[]) if row.get('logical_key') in {'D01','D02','D03'}]
    if len(rows)!=3 or {row.get('logical_key') for row in rows}!={'D01','D02','D03'}:
        raise ValueError('pc1r_existing_progress_guard_failed')
    required=('ready','current','parents_ready','children_ready','vectors_current',
              'graph_provenance_current','graph_evidence_current','bm25_current')
    snapshots=[]
    for row in sorted(rows,key=lambda value:value['logical_key']):
        verified=await runtime.verify_document(str(row['document_id']))
        if (not all(verified.get(key) is True for key in required)
                or verified.get('parent_child_stats')!=row.get('parent_child_stats')):
            raise ValueError('pc1r_existing_documents_guard_failed')
        snapshots.append({
            'logical_key':row['logical_key'],'document_id':row['document_id'],
            'document_version':row['document_version'],'content_hash':row['content_hash'],
            'parent_child_stats':row['parent_child_stats'],
            'graph_evidence_keys':verified.get('graph_evidence_keys') or [],
        })
    return {'documents':snapshots,'sha256':canonical_hash(snapshots)}

def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument('--run-id',required=True); parser.add_argument('--step',required=True,choices=('inventory','validate-runtime-io','prepare','ingest','verify-scope','recover-d04','inspect-pc1r','inspect-d04-graph-reconciliation','reconcile-d04-graph-stage')); parser.add_argument('--runtime',choices=('fake','real'),default='fake'); parser.add_argument('--allow-real-io',action='store_true'); parser.add_argument('--allow-d04-recovery',action='store_true'); parser.add_argument('--allow-pc1r-inspection',action='store_true'); parser.add_argument('--allow-d04-graph-reconciliation',action='store_true'); parser.add_argument('--reconciliation-preflight-sha256'); args=parser.parse_args(argv)
    if args.runtime!='real': parser.error('PC-1 requires --runtime real')
    if args.step=='inspect-pc1r':
        if not args.allow_pc1r_inspection: parser.error('inspect-pc1r requires --allow-pc1r-inspection')
        print(json.dumps(inspect_pc1r(args.run_id),ensure_ascii=False,sort_keys=True)); return 0
    if args.step in {'inspect-d04-graph-reconciliation','reconcile-d04-graph-stage'} and args.run_id!=PC1R_RUN_ID:
        parser.error('D04 graph reconciliation requires the exact authorized run-id')
    from config import settings
    current_inventory=inventory(); current_embedding=embedding(settings); current_runner=runner(args.run_id,settings)
    if args.step=='inventory':
        atomic_json(current_runner.path('inventory.json'),current_inventory); print({'run_id':args.run_id,'inventory':current_inventory['checks']}); return 0
    stored=current_runner._read('inventory.json',None)
    if stored != current_inventory: raise ValueError('pc1_inventory_changed')
    if args.step=='validate-runtime-io':
        from services.hybrid_retrieval_v2_host import build_real_runtime_host
        with build_real_runtime_host(settings) as host:
            print(host.validate_readiness()); return 0
    if args.step=='prepare':
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(); print(current_runner.prepare(git_commit=commit,embedding=current_embedding)); return 0
    if args.step in {'inspect-d04-graph-reconciliation','reconcile-d04-graph-stage'}:
        if args.run_id!=PC1R_RUN_ID: parser.error('D04 graph reconciliation requires the exact authorized run-id')
        if args.step=='inspect-d04-graph-reconciliation' and not args.allow_pc1r_inspection:
            parser.error('inspect-d04-graph-reconciliation requires --allow-pc1r-inspection')
        if args.step=='reconcile-d04-graph-stage' and (not args.allow_real_io or not args.allow_d04_graph_reconciliation):
            parser.error('reconcile-d04-graph-stage requires --allow-real-io and --allow-d04-graph-reconciliation')
        async def graph_reconciliation():
            from services.hybrid_retrieval_v2_host import build_real_runtime_host
            with build_real_runtime_host(settings) as host:
                try:
                    await host.open_ingestion_dependencies()
                    runtime=build_hosted_real_runtime(host,trace_run_id=args.run_id,ingestion_namespace=f'pc1-{args.run_id}')
                    invariants=await d01_d03_invariants(runtime,current_runner)
                    inspection=await runtime.inspect_failed_graph_reconciliation(
                        document_id=D04_ID,version=1,failed_operation_id=D04_FAILED_OPERATION,
                    )
                    inspection.update({
                        'schema_version':'pc1r-graph-reconciliation-preflight-v1',
                        'run_id':args.run_id,'logical_key':'D04',
                        'benchmark_hashes':current_inventory['hashes'],
                        'configuration':configuration(settings),'embedding':current_embedding,
                        'd01_d03_invariants':invariants,'write_operations':0,
                    })
                    preflight_path=current_runner.path('pc1r-graph-reconciliation-preflight.json')
                    if args.step=='inspect-d04-graph-reconciliation':
                        atomic_json(preflight_path,inspection)
                        return {**inspection,'artifact_sha256':sha(preflight_path)}

                    if not preflight_path.is_file(): raise ValueError('pc1r_graph_reconciliation_preflight_missing')
                    if not args.reconciliation_preflight_sha256 or sha(preflight_path)!=args.reconciliation_preflight_sha256:
                        raise ValueError('pc1r_graph_reconciliation_preflight_hash_mismatch')
                    frozen=json.loads(preflight_path.read_text(encoding='utf-8'))
                    operation_id=str(uuid.uuid5(uuid.NAMESPACE_URL,
                        f'{args.run_id}:D04:graph-reconciliation:{args.reconciliation_preflight_sha256}'))
                    result_path=current_runner.path('pc1r-graph-reconciliation-result.json')
                    prior=current_runner._read('pc1r-graph-reconciliation-result.json',None)
                    already_succeeded=bool(
                        prior and prior.get('outcome')=='PASS'
                        and prior.get('reconciliation_operation_id')==operation_id
                        and prior.get('preflight_sha256')==args.reconciliation_preflight_sha256
                    )
                    frozen_identity_valid=bool(
                        frozen.get('passed') is True
                        and frozen.get('document_id')==D04_ID and int(frozen.get('version') or 0)==1
                        and frozen.get('failed_operation_id')==D04_FAILED_OPERATION
                        and frozen.get('d01_d03_invariants',{}).get('sha256')==invariants.get('sha256')
                    )
                    live_identity_valid=bool(
                        inspection.get('passed') is True
                        and frozen.get('graph_evidence_keys')==inspection.get('graph_evidence_keys')
                        and frozen.get('graph_evidence_identity_hash')==inspection.get('graph_evidence_identity_hash')
                    )
                    if not frozen_identity_valid or (not already_succeeded and not live_identity_valid):
                        raise ValueError('pc1r_graph_reconciliation_identity_changed')
                    result=await runtime.reconcile_failed_graph_stage(
                        document_id=D04_ID,version=1,failed_operation_id=D04_FAILED_OPERATION,
                        expected_evidence_keys=frozen['graph_evidence_keys'],
                        expected_evidence_hash=frozen['graph_evidence_identity_hash'],
                        reconciliation_operation_id=operation_id,
                    )
                    after=await runtime.verify_failed_retry_preconditions(
                        document_id=D04_ID,version=1,failed_operation_id=D04_FAILED_OPERATION,
                    )
                    invariants_after=await d01_d03_invariants(runtime,current_runner)
                    if (result.get('status')!='succeeded' or after.get('passed') is not True
                            or invariants_after.get('sha256')!=invariants.get('sha256')):
                        raise ValueError('pc1r_graph_reconciliation_postcondition_failed')
                    artifact={
                        'schema_version':'pc1r-graph-reconciliation-result-v1','gate':'PC-1R-GRAPH-RECONCILIATION',
                        'outcome':'PASS','run_id':args.run_id,'logical_key':'D04','document_id':D04_ID,
                        'version':1,'original_operation_id':D04_FAILED_OPERATION,
                        'reconciliation_operation_id':operation_id,
                        'preflight_sha256':args.reconciliation_preflight_sha256,
                        'graph_evidence_identity_hash':frozen['graph_evidence_identity_hash'],
                        'before_counts':frozen['counts'],'after_counts':after['counts'],
                        'd01_d03_invariants_before_sha256':invariants['sha256'],
                        'd01_d03_invariants_after_sha256':invariants_after['sha256'],
                        'registry_unchanged_failed_version_one':True,'d04_retry_executed':False,
                        'd05_d20_processed':False,'cleanup_executed':False,
                        'idempotent_replay':already_succeeded,
                    }
                    atomic_json(result_path,artifact)
                    if not already_succeeded:
                        append_gate_history(current_runner.path('gate-history.jsonl'),{
                            'gate':'PC-1R-GRAPH-RECONCILIATION','outcome':'PASS',
                            'artifact':'pc1r-graph-reconciliation-result.json','artifact_sha256':sha(result_path),
                        })
                    return artifact
                finally:
                    await host.close_ingestion_dependencies()
        print(json.dumps(asyncio.run(graph_reconciliation()),ensure_ascii=False,sort_keys=True)); return 0
    if args.step=='recover-d04':
        if args.run_id!='pc1-flood-longdoc-v3-20260921-01' or not args.allow_d04_recovery: parser.error('recover-d04 requires the exact authorized run-id and --allow-d04-recovery')
    if not args.allow_real_io: parser.error('ingest requires --allow-real-io')
    async def ingest():
        from services.hybrid_retrieval_v2_host import build_real_runtime_host
        with build_real_runtime_host(settings) as host:
            try:
                await host.open_ingestion_dependencies()
                runtime=build_hosted_real_runtime(host,trace_run_id=args.run_id,ingestion_namespace=f'pc1-{args.run_id}')
                if args.step=='verify-scope':
                    await current_runner.verify_scope(runtime)
                    return {'run_id':args.run_id,'status':'PC1_SCOPE_VERIFIED','scope':current_runner._state().get('scope')}
                if args.step=='recover-d04':
                    original={'logical_key':'D04','document_id':'90eec68b-7560-4181-99d4-c74a4390f4a0','version':1,'operation_id':'addb1b99-c332-4698-ad1f-2b9eb35bcd11'}
                    progress=current_runner._read('ingestion-progress.json',{'documents':[]})
                    retry_result_path=current_runner.path('pc1r-recovery-result.json')
                    recovery_operation_id=None
                    try:
                        if {row.get('logical_key') for row in progress.get('documents',[])} != {'D01','D02','D03'}: raise ValueError('pc1r_progress_guard_failed')
                        required=('ready','current','parents_ready','children_ready','vectors_current','graph_provenance_current','graph_evidence_current','bm25_current')
                        stable=await runtime.verify_document('6328824e-6fda-465f-91e6-1ecdcbd4c7e3'); stable2=await runtime.verify_document('b7a4c3b7-f34f-4bc4-84a9-df93f767ee3b'); stable3=await runtime.verify_document('682c450b-d177-4307-8716-0c6ef556f9a7')
                        if not all(all(row.get(key) is True for key in required) for row in (stable,stable2,stable3)): raise ValueError('pc1r_existing_documents_guard_failed')
                        preflight=await runtime.verify_failed_retry_preconditions(document_id=original['document_id'],version=1,failed_operation_id=original['operation_id'])
                        preflight.update({'run_id':args.run_id,'benchmark_hashes':current_inventory['hashes'],'original_manifest_git_commit':current_runner._manifest().get('git_commit'),'retry_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
                        atomic_json(current_runner.path('pc1r-retry-preflight.json'),preflight)
                        if preflight.get('passed') is not True: raise ValueError('pc1r_retry_preflight_failed')
                        doc=next(row for row in documents() if row['id']=='D04')
                        result=await runtime.retry_failed_create_version(document_id=original['document_id'],version=1,filename=doc['filename'],content=doc['content'],failed_operation_id=original['operation_id'])
                        recovery_operation_id=result.get('operation_id')
                        verified=await runtime.verify_document(original['document_id'])
                        expected_keys=sorted(str(value) for value in (result.get('graph_evidence_keys') or []))
                        observed_keys=list(verified.get('graph_evidence_keys') or [])
                        if (result['document_id']!=original['document_id'] or int(result['version'])!=1
                                or not all(verified.get(key) is True for key in required)
                                or not expected_keys or observed_keys!=expected_keys): raise ValueError('pc1r_d04_verification_failed')
                        stats=verified.get('parent_child_stats') or {}
                        if int(stats.get('parent_count') or 0)!=3 or not 17<=int(stats.get('child_count') or 0)<=18: raise ValueError('pc1r_d04_shape_failed')
                        progress['documents'].append({'logical_key':'D04','source':doc['filename'],'document_id':original['document_id'],'document_version':1,'content_hash':result['content_hash'],'status':'ready','operation_id':result['operation_id'],'recovery_of_operation_id':original['operation_id'],'parent_child_stats':stats})
                        atomic_json(current_runner.path('ingestion-progress.json'),progress)
                        artifact={'gate':'PC-1R-B','outcome':'PASS','run_id':args.run_id,'original_operation_id':original['operation_id'],'recovery_operation_id':result['operation_id'],'document_id':result['document_id'],'version':result['version'],'graph_evidence_keys':expected_keys,'parent_child_stats':stats,'d01_d03_invariants':True,'cleanup_executed':False}
                        atomic_json(retry_result_path,artifact)
                        append_gate_history(current_runner.path('gate-history.jsonl'),{'gate':'PC-1R-B','outcome':'PASS','artifact':'pc1r-recovery-result.json','artifact_sha256':sha(retry_result_path)})
                        atomic_json(current_runner.path('state.json'),{'run_id':args.run_id,'status':'PC1R_VERIFIED','completed':{'PC-1R-A':'BLOCKED','PC-1R-B':'PASS'},'cleanup_executed':False})
                        return artifact
                    except Exception as exc:
                        artifact={'gate':'PC-1R-B','outcome':'BLOCKED','run_id':args.run_id,'error':type(exc).__name__,'recovery_operation_id':recovery_operation_id or (runtime.last_ingestion_operation or {}).get('operation_id'),'original_operation_id':original['operation_id'],'document_id':original['document_id'],'version':1,'cleanup_executed':False}
                        atomic_json(retry_result_path,artifact)
                        append_gate_history(current_runner.path('gate-history.jsonl'),{'gate':'PC-1R-B','outcome':'BLOCKED','artifact':'pc1r-recovery-result.json','artifact_sha256':sha(retry_result_path)})
                        raise
                return await current_runner.ingest(runtime,documents())
            finally:
                await host.close_ingestion_dependencies()
    print(asyncio.run(ingest())); return 0

if __name__=='__main__': raise SystemExit(main())
