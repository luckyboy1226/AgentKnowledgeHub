"""PC-1 controlled ingestion for the frozen flood long-document benchmark."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BENCHMARK=ROOT/'benchmarks'/'flood_emergency_20docs_60q_longdoc_v3_FINAL'/'flood_emergency_20docs_60q_longdoc_v3_final'
ARTIFACT_ROOT=ROOT/'.runtime'/'parent-child-pc1'
sys.path.insert(0,str(ROOT/'python')); os.chdir(ROOT/'python')
from services.hybrid_retrieval_v2_real_evaluation import ControlledRealEvaluationRunner, atomic_json
from services.hybrid_retrieval_v2_runtime import build_hosted_real_runtime


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

def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument('--run-id',required=True); parser.add_argument('--step',required=True,choices=('inventory','validate-runtime-io','prepare','ingest','recover-d04')); parser.add_argument('--runtime',choices=('fake','real'),default='fake'); parser.add_argument('--allow-real-io',action='store_true'); parser.add_argument('--allow-d04-recovery',action='store_true'); args=parser.parse_args(argv)
    if args.runtime!='real': parser.error('PC-1 requires --runtime real')
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
    if args.step=='recover-d04':
        if args.run_id!='pc1-flood-longdoc-v3-20260921-01' or not args.allow_d04_recovery: parser.error('recover-d04 requires the exact authorized run-id and --allow-d04-recovery')
    if not args.allow_real_io: parser.error('ingest requires --allow-real-io')
    async def ingest():
        from services.hybrid_retrieval_v2_host import build_real_runtime_host
        with build_real_runtime_host(settings) as host:
            try:
                await host.open_ingestion_dependencies()
                runtime=build_hosted_real_runtime(host,trace_run_id=args.run_id,ingestion_namespace=f'pc1-{args.run_id}')
                if args.step=='recover-d04':
                    original={'logical_key':'D04','document_id':'90eec68b-7560-4181-99d4-c74a4390f4a0','version':1,'operation_id':'addb1b99-c332-4698-ad1f-2b9eb35bcd11'}
                    progress=current_runner._read('ingestion-progress.json',{'documents':[]})
                    if {row.get('logical_key') for row in progress.get('documents',[])} != {'D01','D02','D03'}: raise ValueError('pc1r_progress_guard_failed')
                    stable=await runtime.verify_document('6328824e-6fda-465f-91e6-1ecdcbd4c7e3'); stable2=await runtime.verify_document('b7a4c3b7-f34f-4bc4-84a9-df93f767ee3b'); stable3=await runtime.verify_document('682c450b-d177-4307-8716-0c6ef556f9a7')
                    if not all(all(row.get(key) is True for key in ('ready','current','parents_ready','children_ready','vectors_current','graph_provenance_current','bm25_current')) for row in (stable,stable2,stable3)): raise ValueError('pc1r_existing_documents_guard_failed')
                    doc=next(row for row in documents() if row['id']=='D04')
                    result=await runtime.retry_failed_create_version(document_id=original['document_id'],version=1,filename=doc['filename'],content=doc['content'],failed_operation_id=original['operation_id'])
                    verified=await runtime.verify_document(original['document_id'])
                    if (result['document_id']!=original['document_id'] or int(result['version'])!=1 or not all(verified.get(key) is True for key in ('ready','current','parents_ready','children_ready','vectors_current','graph_provenance_current','bm25_current'))): raise ValueError('pc1r_d04_verification_failed')
                    stats=verified.get('parent_child_stats') or {}
                    if int(stats.get('parent_count') or 0)!=3 or not 17<=int(stats.get('child_count') or 0)<=18: raise ValueError('pc1r_d04_shape_failed')
                    progress['documents'].append({'logical_key':'D04','source':doc['filename'],'document_id':original['document_id'],'document_version':1,'content_hash':result['content_hash'],'status':'ready','operation_id':result['operation_id'],'recovery_of_operation_id':original['operation_id'],'parent_child_stats':stats})
                    atomic_json(current_runner.path('ingestion-progress.json'),progress)
                    return {'original_operation_id':original['operation_id'],'recovery_operation_id':result['operation_id'],'document_id':result['document_id'],'version':result['version'],'parent_child_stats':stats}
                return await current_runner.ingest(runtime,documents())
            finally:
                await host.close_ingestion_dependencies()
    print(asyncio.run(ingest())); return 0

if __name__=='__main__': raise SystemExit(main())
