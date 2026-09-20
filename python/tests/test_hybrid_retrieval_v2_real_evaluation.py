import asyncio, hashlib, json, uuid
from pathlib import Path
import pytest
from services.hybrid_retrieval_v2_real_evaluation import ControlledRealEvaluationRunner, RunState, VARIANTS, _hash_file_tree

class Fake:
    def __init__(self): self.ingests=0; self.plans=0; self.calls=[]
    async def ingest(self,d,r): self.ingests+=1; return {'document_id':str(uuid.uuid5(uuid.NAMESPACE_URL,d['id'])),'version':1,'content_hash':'x'}
    async def verify_document(self,id): return {k:True for k in ('ready','current','parents_ready','children_ready','vectors_current','graph_provenance_current')}
    async def verify_scope(self,a): return {k:True for k in ('mongo','chroma','neo4j','bm25')}
    async def build_query_plan(self,q,r): self.plans+=1; return {'question_id':q['question_id'],'queries':[q['question_id']],'entities':[],'keywords':[],'intent':'factoid'}
    async def evaluate(self,p,v,a,**kw):
        self.calls.append((p['question_id'],v,kw)); graph=v in {'vector_graph_rrf','hybrid_v2_no_rerank'}; bm25=v in {'bm25_vector_rrf','hybrid_v2_no_rerank'}
        return {'final_context_ids':['x'],'candidate_ranks':[1],'document_ranks':['doc-a'],'call_audit':{'vector_searches':1,'bm25_searches':int(bm25),'graph_queries':int(graph),'reranker_calls':0}}

def source_bundle(tmp_path, questions):
 source=tmp_path/'source'; source.mkdir(); (source/'benchmark_manifest.json').write_text(json.dumps({'files':{'questions_json':'benchmark_questions.json'}}),encoding='utf-8'); (source/'benchmark_questions.json').write_text(json.dumps({'questions':questions}),encoding='utf-8'); return source.name,_hash_file_tree(source)

def make(tmp_path, question_count=2):
 questions=[{'question_id':f'Q{i:02d}','question':f'question {i}'} for i in range(1,question_count+1)]
 source_name,source_hash=source_bundle(tmp_path,questions)
 b=tmp_path/'benchmark'; b.mkdir(); (b/'benchmark_manifest.json').write_text(json.dumps({'benchmark_id':'b','benchmark_version':'1','document_count':2,'question_count':question_count,'source_benchmark_sha256':source_hash}),encoding='utf-8'); (b/'ground_truth.json').write_text(json.dumps({'source_benchmark':{'path':source_name,'sha256':source_hash},'questions':[{'question_id':q['question_id']} for q in questions]}),encoding='utf-8'); return ControlledRealEvaluationRunner(tmp_path/'out','r1',b,{'x':1})

def test_prepare_is_immutable_and_has_no_cleanup(tmp_path):
 r=make(tmp_path); m=r.prepare(git_commit='abc',embedding={'dimension':1024}); assert m['status']=='PREPARED'; assert r.prepare(git_commit='abc',embedding={'dimension':1024})['created_at']==m['created_at']; assert not hasattr(r,'cleanup')
 with pytest.raises(ValueError): r.prepare(git_commit='different',embedding={'dimension':1024})

def test_recovery_scope_plan_trace_and_exact_cleanup(tmp_path):
 async def run():
  r=make(tmp_path,question_count=5); f=Fake(); r.prepare(git_commit='abc',embedding={'dimension':1024}); docs=[{'id':'D1','filename':'a.txt'},{'id':'D2','filename':'b.txt'}]
  await r.ingest(f,docs); assert f.ingests==2; await r.ingest(f,docs); assert f.ingests==2
  await r.verify_scope(f); await r.build_query_plans(f); assert f.plans==5; await r.build_query_plans(f); assert f.plans==5
  await r.trace_equivalence(f); assert r._state()['status']==RunState.TRACE_EQUIVALENCE_VERIFIED.value
  artifact=json.loads(r.path('trace-equivalence.json').read_text())
  assert artifact['sample_question_ids']==['Q01','Q02','Q03','Q04','Q05'] and len(artifact['rows'])==20
  assert all(set(row['compared_outputs']['off'])=={'final_context_ids','candidate_ranks','document_ranks'} for row in artifact['rows'])
  assert all(row['call_audit']['off']==row['call_audit']['expected']==row['call_audit']['on'] for row in artifact['rows'])
  assert len(f.calls)==40 and all(call[2]['graph_trace']==(call[1] in {'vector_graph_rrf','hybrid_v2_no_rerank'}) for call in f.calls if call[2]['trace'])
  plan=r.cleanup_plan(); assert plan['document_count']==2 and plan['requires_explicit_authorization'] and not plan['cleanup_executed']
 asyncio.run(run())

def test_scope_is_fail_closed_and_graph_trace_is_only_graph_variants(tmp_path):
 r=make(tmp_path)
 with pytest.raises(ValueError): r._allowlist()
 assert set(VARIANTS)=={'vector_only','bm25_vector_rrf','vector_graph_rrf','hybrid_v2_no_rerank'}

def test_ingestion_failure_never_emits_allowlist_or_final_results(tmp_path):
 class Incomplete(Fake):
  async def verify_document(self, _):
   return {key: False for key in ('ready','current','parents_ready','children_ready','vectors_current','graph_provenance_current')}
 async def run():
  r=make(tmp_path); r.prepare(git_commit='abc',embedding={'dimension':1024})
  with pytest.raises(ValueError, match='ingestion_incomplete'):
   await r.ingest(Incomplete(), [{'id':'D1','filename':'a.txt'}])
  assert r._state()['status']==RunState.FAILED.value
  assert not r.path('ingestion-results.json').exists()
  assert not r.path('document-map.json').exists()
  assert not r.path('document-allowlist.json').exists()
 asyncio.run(run())

def test_exact_recovery_records_ready_document_without_reingest(tmp_path):
 class RecoveryFake(Fake):
  def __init__(self): super().__init__(); self.recovery_calls=[]; self.ingested=[]
  async def verify_recovery_identity(self, **identity):
   self.recovery_calls.append(identity)
   return {**identity, 'document_version':1, 'content_hash':'hash-d1', 'status':'ready'}
  async def ingest(self, document, run_id):
   self.ingested.append(document['id'])
   return {'document_id':'00000000-0000-4000-8000-000000000003','version':1,'content_hash':'hash-d2'}
 async def run():
  r=make(tmp_path); fake=RecoveryFake(); r.prepare(git_commit='abc',embedding={'dimension':1024})
  record={'logical_key':'D1','source':'a.txt','document_id':'00000000-0000-4000-8000-000000000001','document_version':1,'content_hash':'hash-d1','operation_id':'00000000-0000-4000-8000-000000000002'}
  await r.recover_ready_document(fake,record)
  assert fake.recovery_calls==[{'logical_key':'D1','source':'a.txt','document_id':record['document_id'],'operation_id':record['operation_id']}]
  assert fake.ingested==[]
  await r.ingest(fake,[{'id':'D1','filename':'a.txt'},{'id':'D2','filename':'b.txt'}])
  assert fake.ingested==['D2']
  assert r.path('document-allowlist.json').exists()
 asyncio.run(run())

def test_recovery_identity_mismatch_leaves_progress_absent(tmp_path):
 class Mismatch(Fake):
  async def verify_recovery_identity(self, **identity):
   return {**identity, 'document_version':2, 'content_hash':'wrong', 'status':'ready'}
 async def run():
  r=make(tmp_path); r.prepare(git_commit='abc',embedding={'dimension':1024})
  record={'logical_key':'D1','source':'a.txt','document_id':'00000000-0000-4000-8000-000000000001','document_version':1,'operation_id':'00000000-0000-4000-8000-000000000002'}
  with pytest.raises(ValueError, match='recovery_identity_mismatch'):
   await r.recover_ready_document(Mismatch(),record)
  assert not r.path('ingestion-progress.json').exists()
 asyncio.run(run())

def test_sixty_query_plans_are_unique_frozen_and_resume_only_missing(tmp_path):
 async def run():
  questions=[{'question_id':f'Q{i:02d}','question':f'question {i}'} for i in range(1,61)]
  source_name,source_hash=source_bundle(tmp_path,questions)
  b=tmp_path/'benchmark60'; b.mkdir()
  (b/'benchmark_manifest.json').write_text(json.dumps({'benchmark_id':'b','benchmark_version':'1','document_count':1,'question_count':60,'source_benchmark_sha256':source_hash}),encoding='utf-8')
  (b/'ground_truth.json').write_text(json.dumps({'source_benchmark':{'path':source_name,'sha256':source_hash},'questions':[{'question_id':q['question_id']} for q in questions]}),encoding='utf-8')
  r=ControlledRealEvaluationRunner(tmp_path/'out','r60',b,{})
  f=Fake(); r.prepare(git_commit='abc',embedding={'dimension':1024}); r._write_state(RunState.SCOPE_VERIFIED)
  await r.build_query_plans(f)
  assert f.plans==60 and r._state()['status']==RunState.QUERY_PLANS_FROZEN.value
  plans=json.loads(r.path('query-plans.json').read_text())['plans']
  assert len(plans)==60 and len({p['question_id'] for p in plans})==60
  await r.build_query_plans(f)
  assert f.plans==60
  r2=ControlledRealEvaluationRunner(tmp_path/'out','r60-resume',b,{})
  f2=Fake(); r2.prepare(git_commit='abc',embedding={'dimension':1024}); r2._write_state(RunState.SCOPE_VERIFIED)
  atomic = {'question_id':'Q01','queries':['Q01'],'entities':[],'keywords':[],'intent':'factoid'}
  atomic['plan_hash']=__import__('services.hybrid_retrieval_v2_real_evaluation',fromlist=['_hash'])._hash(atomic)
  (r2.path('query-plans.json')).write_text(json.dumps({'plans':[atomic]}),encoding='utf-8')
  await r2.build_query_plans(f2)
  assert f2.plans==59
 asyncio.run(run())

def test_joined_question_source_fails_closed_before_any_planner_call(tmp_path):
 async def run():
  r=make(tmp_path); f=Fake(); r.prepare(git_commit='abc',embedding={'dimension':1024}); r._write_state(RunState.SCOPE_VERIFIED)
  ground=json.loads((r.benchmark_root/'ground_truth.json').read_text()); ground['source_benchmark']['sha256']='0'*64
  (r.benchmark_root/'ground_truth.json').write_text(json.dumps(ground),encoding='utf-8')
  with pytest.raises(ValueError, match='canonical_question_source_hash_mismatch'):
   await r.build_query_plans(f)
  assert f.plans==0 and not r.path('query-plans.json').exists()
 asyncio.run(run())

def test_g3_pre_llm_failure_resume_requires_intact_g2_artifacts(tmp_path):
 r=make(tmp_path); r.prepare(git_commit='abc',embedding={'dimension':1024})
 ids=[str(uuid.uuid5(uuid.NAMESPACE_URL,f'd{i}')) for i in range(20)]
 (r.path('document-allowlist.json')).parent.mkdir(parents=True,exist_ok=True)
 (r.path('document-allowlist.json')).write_text(json.dumps({'document_ids':ids,'sha256':'allow'}),encoding='utf-8')
 (r.path('document-map.json')).write_text(json.dumps({'mapping':{f'D{i}':value for i,value in enumerate(ids)}}),encoding='utf-8')
 r._write_state(RunState.FAILED,error='KeyError',failed_question_id='Q01',scope={key:True for key in ('mongo','chroma','neo4j','bm25')})
 r.resume_g3_pre_llm_failure(allowlist_hash='allow')
 assert r._state()['status']==RunState.SCOPE_VERIFIED.value and not r.path('query-plans.json').exists()

def test_g4_resume_guard_preserves_exact_failure_evidence_and_frozen_inputs(tmp_path):
 async def run():
  from services.query_embedding_snapshot import FrozenQueryEmbeddingSnapshot
  r=make(tmp_path,question_count=5); f=Fake(); r.prepare(git_commit='abc',embedding={'dimension':1024})
  await r.ingest(f,[{'id':'D1','filename':'a.txt'},{'id':'D2','filename':'b.txt'}]); await r.verify_scope(f); await r.build_query_plans(f)
  plans=json.loads(r.path('query-plans.json').read_text())['plans']; allow=json.loads(r.path('document-allowlist.json').read_text())
  artifact={'run_id':'r1','sample_question_ids':['Q01','Q02','Q03','Q04','Q05'],'query_plans_hash':__import__('services.hybrid_retrieval_v2_real_evaluation',fromlist=['_hash'])._hash(plans),'allowlist_hash':allow['sha256'],'passed':False,'rows':[{'question_id':'Q01','variant':'vector_only','equivalent':True},{'question_id':'Q01','variant':'bm25_vector_rrf','equivalent':False,'differences':{'document_ranks':{'off':['a'],'on':['b']}}}]}
  r.path('trace-equivalence.json').write_text(json.dumps(artifact),encoding='utf-8'); raw=r.path('trace-equivalence.json').read_bytes()
  retry={**artifact,'rows':[{'question_id':'Q01','variant':'bm25_vector_rrf','equivalent':False}]}
  r.path('trace-equivalence-retry-1.json').write_text(json.dumps(retry),encoding='utf-8')
  class Embedding:
   async def aembed_query(self,_): return [0.0]*1024
  snapshot=FrozenQueryEmbeddingSnapshot.empty(run_id='r1',query_plans_hash=artifact['query_plans_hash'],embedding={'dimension':1024})
  await snapshot.freeze_missing(plans=plans,run_id='r1',query_plans_hash=artifact['query_plans_hash'],embedding={'dimension':1024},provider=Embedding(),persist_partial=lambda _:None)
  r.path('query-embeddings.json').write_text(json.dumps(snapshot.payload),encoding='utf-8')
  r._write_state(RunState.FAILED,error='trace_equivalence_failed')
  guards={'query_plans_hash':artifact['query_plans_hash'],'allowlist_hash':allow['sha256'],
          'failure_artifact_hash':hashlib.sha256(raw).hexdigest(),
          'retry_artifact_hash':hashlib.sha256(r.path('trace-equivalence-retry-1.json').read_bytes()).hexdigest(),
          'snapshot_hash':hashlib.sha256(r.path('query-embeddings.json').read_bytes()).hexdigest()}
  protected={name:r.path(name).read_bytes() for name in ('state.json','manifest.json','trace-equivalence.json','trace-equivalence-retry-1.json','query-plans.json','document-allowlist.json','query-embeddings.json')}
  with pytest.raises(ValueError,match='g4_in_place_resume_forbidden'):
   r.resume_g4_trace_equivalence_failure(query_plans_hash=artifact['query_plans_hash'],allowlist_hash=allow['sha256'],failure_artifact_hash=guards['failure_artifact_hash'])
  prepared=r.prepare_g4_recovery(recovery_id='r1-g4-recovery',**guards)
  assert prepared['source_run_id']=='r1'
  result=await r.trace_equivalence_recovery(f,recovery_id='r1-g4-recovery',**guards)
  assert result['passed'] is True
  recovery=r.root/'r1-g4-recovery'
  assert json.loads((recovery/'state.json').read_text())['status']=='TRACE_EQUIVALENCE_VERIFIED'
  assert all(r.path(name).read_bytes()==value for name,value in protected.items())
 asyncio.run(run())

def test_g4_recovery_fails_closed_on_mutated_retry_or_snapshot(tmp_path):
 r=make(tmp_path)
 with pytest.raises(ValueError,match='g4_resume_guard_state_failed'):
  r.prepare_g4_recovery(recovery_id='safe-recovery',query_plans_hash='x',allowlist_hash='y',
                        failure_artifact_hash='z',retry_artifact_hash='r',snapshot_hash='s')
 with pytest.raises(ValueError,match='g4_recovery_id_invalid'):
  r._recovery_directory('../escape')
