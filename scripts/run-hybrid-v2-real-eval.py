"""Phase G command boundary. Only prepare is intentionally self-contained."""
from __future__ import annotations
import argparse, subprocess, asyncio, json, os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'python'))
# Evaluation settings use a relative .env path; set it before importing any
# service that may import config.settings.
os.chdir(ROOT/'python')
from services.hybrid_retrieval_v2_real_evaluation import ControlledRealEvaluationRunner
from services.hybrid_retrieval_v2_runtime import FakePhaseGRuntime, build_real_runtime, build_hosted_real_runtime, build_query_planning_runtime

RECOVERY_RECORDS = {
 'g2-real-ingestion-20260920-02': {
  'logical_key':'D01', 'source':'D01_group_organization.txt',
  'document_id':'7ce6d859-9a26-4844-b017-245a4256b69a', 'document_version':1,
  'operation_id':'48d26deb-9ea0-42ea-84cf-7793ea2a81c4',
 }
}
G3_RESUME_ALLOWLIST_HASHES = {
 'g2-real-ingestion-20260920-02': '4778224a999f6f3ec91d0555c313a8c58c97fa0e34567a532c23621b92376fa0',
}
G4_RESUME_GUARDS = {
 'g2-real-ingestion-20260920-02': {
  'query_plans_hash':'99eb2478bb57b59e1847ad7220af03925a5562369032757861eabef64f382b8a',
  'allowlist_hash':'4778224a999f6f3ec91d0555c313a8c58c97fa0e34567a532c23621b92376fa0',
  'failure_artifact_hash':'4ed080f0d068d58cc3e1cf0023db73467b9979d85289ff8c9f55dc9f54187e46',
  'retry_artifact_hash':'02c4edeb9e8adde645576b1e1ef3ceb30297890d82335d5d9cfb7b62d4772724',
  'snapshot_hash':'320f345b9a2c9ac259356f9fad11d60d9cbe03babd012c540abd5d9a487afe98',
 },
}

def _configuration(settings):
 return {'final_top_k':settings.final_context_top_k,'rrf_k':settings.rrf_k,'rrf_fusion_top_k':settings.rrf_fusion_top_k,'context_token_budget':settings.final_context_token_budget,'parent_child_chunk_enabled':settings.parent_child_chunk_enabled,'reranker_enabled':False}

def _embedding(settings):
 return {'provider':settings.embedding_config.provider,'model':settings.embedding_config.model,'dimension':settings.embedding_dimensions,'embedding_space_id':settings.resolved_embedding_space_id}

def _documents():
 return json.loads((ROOT/'benchmarks'/'enterprise_20docs_60q_expanded'/'benchmark_documents.json').read_text(encoding='utf-8'))['documents']

def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument('--run-id'); p.add_argument('--recovery-id'); p.add_argument('--step',required=True,choices=('validate-runtime','validate-runtime-io','prepare','ingest','recover-ingestion-document','verify-scope','resume-query-plans','build-query-plans','freeze-query-embeddings','resume-trace-equivalence','trace-equivalence','evaluate','report','show-cleanup-plan')); p.add_argument('--runtime',choices=('fake','real'),default='fake'); p.add_argument('--allow-real-io',action='store_true'); p.add_argument('--allow-recovery-record',action='store_true'); p.add_argument('--allow-query-plan-llm',action='store_true'); p.add_argument('--allow-query-embedding-freeze',action='store_true'); p.add_argument('--allow-g3-resume',action='store_true'); p.add_argument('--allow-g4-resume',action='store_true'); p.add_argument('--allow-real-retrieval',action='store_true'); a=p.parse_args(argv)
 if a.step not in {'validate-runtime','validate-runtime-io'} and not a.run_id: p.error('--run-id is required for run steps')
 if a.step=='validate-runtime':
  if a.runtime!='real': p.error('validate-runtime requires --runtime real')
  # Settings intentionally use a relative .env path; align this read-only
  # validation with the backend process without constructing any clients.
  from config import settings
  print(build_real_runtime(settings).construction_metadata); return 0
 if a.step=='validate-runtime-io':
  if a.runtime!='real': p.error('validate-runtime-io requires --runtime real')
  from config import settings
  from services.hybrid_retrieval_v2_host import build_real_runtime_host
  with build_real_runtime_host(settings) as host: print(host.validate_readiness()); return 0
 from config import settings
 r=ControlledRealEvaluationRunner(ROOT/'.runtime'/'evaluation',a.run_id,ROOT/'benchmarks'/'enterprise_20docs_retrieval_v2',_configuration(settings))
 if a.step=='prepare':
  commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(); print(r.prepare(git_commit=commit,embedding=_embedding(settings))); return 0
 if a.step=='show-cleanup-plan': print(r.cleanup_plan()); return 0
 if a.step=='recover-ingestion-document':
  if a.runtime!='real': p.error('recover-ingestion-document requires --runtime real')
  if not a.allow_recovery_record: p.error('recover-ingestion-document requires --allow-recovery-record')
  if a.run_id not in RECOVERY_RECORDS: p.error('recovery record is not authorized for this run-id')
 if a.step=='build-query-plans':
  if a.runtime!='real': p.error('build-query-plans requires --runtime real')
  if not a.allow_query_plan_llm: p.error('build-query-plans requires --allow-query-plan-llm')
 if a.step=='freeze-query-embeddings':
  if a.runtime!='real': p.error('freeze-query-embeddings requires --runtime real')
  if not a.allow_query_embedding_freeze: p.error('freeze-query-embeddings requires --allow-query-embedding-freeze')
 if a.step=='resume-query-plans':
  if a.runtime!='real': p.error('resume-query-plans requires --runtime real')
  if not a.allow_g3_resume: p.error('resume-query-plans requires --allow-g3-resume')
  if a.run_id not in G3_RESUME_ALLOWLIST_HASHES: p.error('G3 resume is not authorized for this run-id')
 if a.step=='resume-trace-equivalence':
  if a.runtime!='real': p.error('resume-trace-equivalence requires --runtime real')
  if not a.allow_g4_resume: p.error('resume-trace-equivalence requires --allow-g4-resume')
  if a.run_id not in G4_RESUME_GUARDS: p.error('G4 resume is not authorized for this run-id')
  if not a.recovery_id: p.error('resume-trace-equivalence requires --recovery-id')
 if a.step=='trace-equivalence':
  if a.runtime!='real': p.error('trace-equivalence requires --runtime real')
  if not a.allow_real_retrieval: p.error('trace-equivalence requires --allow-real-retrieval')
  if a.recovery_id and not a.allow_g4_resume: p.error('recovery trace-equivalence requires --allow-g4-resume')
  if a.recovery_id and a.run_id not in G4_RESUME_GUARDS: p.error('G4 recovery is not authorized for this run-id')
 if a.runtime=='real' and a.step not in {'prepare','show-cleanup-plan','recover-ingestion-document','verify-scope','resume-query-plans','build-query-plans','freeze-query-embeddings','resume-trace-equivalence','trace-equivalence'} and not a.allow_real_io: p.error('--runtime real requires --allow-real-io')
 if a.runtime=='real' and a.step not in {'ingest','recover-ingestion-document','verify-scope','resume-query-plans','build-query-plans','freeze-query-embeddings','resume-trace-equivalence','trace-equivalence'}: p.error('real runtime is authorized only for controlled G2 ingest, recovery, verify-scope, G3 planning, embedding freeze, and G4 trace equivalence')
 if a.step=='resume-query-plans':
  r.resume_g3_pre_llm_failure(allowlist_hash=G3_RESUME_ALLOWLIST_HASHES[a.run_id]); print({'status':'SCOPE_VERIFIED','llm_calls':0}); return 0
 if a.step=='resume-trace-equivalence':
  manifest=r.prepare_g4_recovery(recovery_id=a.recovery_id,**G4_RESUME_GUARDS[a.run_id]); print({'status':'RECOVERY_PREPARED','recovery_id':a.recovery_id,'real_io':False,'manifest':manifest}); return 0
 fake=FakePhaseGRuntime()
 async def step():
  if a.runtime=='real' and a.step=='build-query-plans':
   return await r.build_query_plans(build_query_planning_runtime(settings))
  if a.runtime=='real' and a.step=='freeze-query-embeddings':
   from providers.factory import create_embedding_provider
   return await r.freeze_query_embeddings(create_embedding_provider(settings),_embedding(settings))
  if a.runtime=='real':
    from services.hybrid_retrieval_v2_host import build_real_runtime_host
    with build_real_runtime_host(settings) as host:
     try:
      await host.open_ingestion_dependencies()
      # Every authorized real-retrieval phase must resolve a run-scoped
      # snapshot before the adapter is constructed.  There is deliberately no
      # provider fallback on this path.
      snapshot=r.frozen_query_embeddings() if a.step in {'trace-equivalence','evaluate'} else None
      runtime=build_hosted_real_runtime(host,trace_run_id=a.run_id,embedding_snapshot=snapshot,query_plans_hash=r._manifest().get('query_plans_hash'))
      if a.step=='recover-ingestion-document': return await r.recover_ready_document(runtime,dict(RECOVERY_RECORDS[a.run_id]))
      if a.step=='ingest':
       docs=_documents()
       if len(docs)!=r._manifest()['document_count']: raise ValueError('frozen_document_count_mismatch')
       return await r.ingest(runtime,docs)
      if a.step=='verify-scope': return await r.verify_scope(runtime)
      if a.step=='trace-equivalence':
       if a.recovery_id:
        return await r.trace_equivalence_recovery(runtime,recovery_id=a.recovery_id,**G4_RESUME_GUARDS[a.run_id])
       return await r.trace_equivalence(runtime)
     finally:
      await host.close_ingestion_dependencies()
  if a.step=='ingest':
   return await r.ingest(fake,_documents())
  if a.step=='verify-scope': return await r.verify_scope(fake)
  if a.step=='build-query-plans': return await r.build_query_plans(fake)
  if a.step=='trace-equivalence': return await r.trace_equivalence(fake)
  if a.step=='evaluate': return await r.evaluate(fake)
  if a.step=='report': return r.report()
 print(asyncio.run(step())); return 0
if __name__=='__main__': raise SystemExit(main())
