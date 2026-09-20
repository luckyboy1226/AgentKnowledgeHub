import asyncio, json, uuid
from pathlib import Path
import pytest
from services.hybrid_retrieval_v2_real_evaluation import ControlledRealEvaluationRunner, RunState, VARIANTS

class Fake:
    def __init__(self): self.ingests=0; self.plans=0; self.calls=[]
    async def ingest(self,d,r): self.ingests+=1; return {'document_id':str(uuid.uuid5(uuid.NAMESPACE_URL,d['id'])),'version':1,'content_hash':'x'}
    async def verify_document(self,id): return {k:True for k in ('ready','current','parents_ready','children_ready','vectors_current','graph_provenance_current')}
    async def verify_scope(self,a): return {k:True for k in ('mongo','chroma','neo4j','bm25')}
    async def build_query_plan(self,q,r): self.plans+=1; return {'queries':[q['question_id']],'entities':[],'keywords':[],'intent':'factoid'}
    async def evaluate(self,p,v,a,**kw): self.calls.append((p['question_id'],v,kw)); return {'final_context_ids':['x'],'candidate_ranks':[1]}

def make(tmp_path):
 b=tmp_path/'benchmark'; b.mkdir(); (b/'benchmark_manifest.json').write_text(json.dumps({'benchmark_id':'b','benchmark_version':'1','document_count':2,'question_count':2}),encoding='utf-8'); (b/'ground_truth.json').write_text(json.dumps({'questions':[{'question_id':'Q1'},{'question_id':'Q2'}]}),encoding='utf-8'); return ControlledRealEvaluationRunner(tmp_path/'out','r1',b,{'x':1})

def test_prepare_is_immutable_and_has_no_cleanup(tmp_path):
 r=make(tmp_path); m=r.prepare(git_commit='abc',embedding={'dimension':1024}); assert m['status']=='PREPARED'; assert r.prepare(git_commit='abc',embedding={'dimension':1024})['created_at']==m['created_at']; assert not hasattr(r,'cleanup')
 with pytest.raises(ValueError): r.prepare(git_commit='different',embedding={'dimension':1024})

def test_recovery_scope_plan_trace_and_exact_cleanup(tmp_path):
 async def run():
  r=make(tmp_path); f=Fake(); r.prepare(git_commit='abc',embedding={'dimension':1024}); docs=[{'id':'D1','filename':'a.txt'},{'id':'D2','filename':'b.txt'}]
  await r.ingest(f,docs); assert f.ingests==2; await r.ingest(f,docs); assert f.ingests==2
  await r.verify_scope(f); await r.build_query_plans(f); assert f.plans==2; await r.build_query_plans(f); assert f.plans==2
  await r.trace_equivalence(f); assert r._state()['status']==RunState.TRACE_EQUIVALENCE_VERIFIED.value
  plan=r.cleanup_plan(); assert plan['document_count']==2 and plan['requires_explicit_authorization'] and not plan['cleanup_executed']
 asyncio.run(run())

def test_scope_is_fail_closed_and_graph_trace_is_only_graph_variants(tmp_path):
 r=make(tmp_path)
 with pytest.raises(ValueError): r._allowlist()
 assert set(VARIANTS)=={'vector_only','bm25_vector_rrf','vector_graph_rrf','hybrid_v2_no_rerank'}
