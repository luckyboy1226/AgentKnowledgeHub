import asyncio, copy
import pytest
from services.query_embedding_snapshot import FrozenQueryEmbeddingSnapshot
from services.vector_store import VectorStoreService
from services.hybrid_retrieval_v2_runtime import HybridRetrievalV2RealRuntime

class Provider:
 provider_name='fake'; model_name='fake'; dimensions=3
 def __init__(self): self.calls=0; self.values=[[1,0,0],[0,1,0]]
 async def aembed_query(self,_): value=self.values[self.calls % len(self.values)]; self.calls+=1; return value

def plan(question_id='Q01',query='query',plan_hash='plan'):
 return {'question_id':question_id,'queries':[query],'plan_hash':plan_hash}

IDENTITY={'provider':'fake','model':'fake','dimension':3,'embedding_space_id':'space'}

async def freeze(provider, plans):
 snapshot=FrozenQueryEmbeddingSnapshot.empty(run_id='r',query_plans_hash='plans',embedding=IDENTITY)
 await snapshot.freeze_missing(plans=plans,run_id='r',query_plans_hash='plans',embedding=IDENTITY,provider=provider,persist_partial=lambda _:None)
 return snapshot

def test_snapshot_binds_query_plan_identity_and_rejects_tampering():
 async def run():
  provider=Provider(); snapshot=await freeze(provider,[plan()])
  assert provider.calls==1 and snapshot.vector_for(run_id='r',plan=plan(),query_plans_hash='plans',embedding=IDENTITY)==(1.0,0.0,0.0)
  with pytest.raises(ValueError): snapshot.vector_for(run_id='r',plan=plan(query='changed'),query_plans_hash='plans',embedding=IDENTITY)
  with pytest.raises(ValueError): snapshot.vector_for(run_id='r',plan=plan(),query_plans_hash='plans',embedding={**IDENTITY,'embedding_space_id':'other'})
  with pytest.raises(ValueError): snapshot.vector_for(run_id='r',plan=plan(),query_plans_hash='plans',embedding={**IDENTITY,'dimension':4})
  altered=copy.deepcopy(snapshot.payload); altered['records'][0]['vector'][0]=9
  with pytest.raises(ValueError): FrozenQueryEmbeddingSnapshot(altered).vector_for(run_id='r',plan=plan(),query_plans_hash='plans',embedding=IDENTITY)
 asyncio.run(run())

def test_precomputed_vector_search_never_calls_provider():
 class Store:
  def query(self,**_): return {'documents':[['x']],'metadatas':[[{'document_id':'d','document_version':1,'chunk_id':'d:c','status':'ready','is_current':True,'source':'x.txt'}]],'distances':[[0.1]]}
 async def run():
  provider=Provider(); service=VectorStoreService(provider); service._backend='chroma'; service._store=Store()
  await service.search('q',top_k=1,query_embedding=[1,0,0]); assert provider.calls==0
  with pytest.raises(Exception): await service.search('q',top_k=1,query_embedding=[1,0])
 asyncio.run(run())

def test_trace_and_variants_share_one_snapshot_without_provider_calls():
 class Production:
  def __init__(self): self.vectors=[]
  async def run_variant(self,*args,**kwargs): self.vectors.append(tuple(kwargs['query_embedding'])); return {'final_context_ids':[],'document_ranks':[],'candidate_ranks':[],'call_audit':{}}
 async def run():
  provider=Provider(); snapshot=await freeze(provider,[plan()]); frozen_calls=provider.calls
  production=Production(); runtime=HybridRetrievalV2RealRuntime(coordinator_factory=lambda:None,services_factory=lambda:None,plan_factory=lambda:None,production_adapter_factory=lambda:production,embedding_snapshot=snapshot,run_id='r',embedding_identity=IDENTITY,query_plans_hash='plans')
  for variant in ('vector_only','bm25_vector_rrf','vector_graph_rrf','hybrid_v2_no_rerank'):
   for trace in (False,True):
    result=await runtime.evaluate(plan(),variant,frozenset({'d'}),trace=trace,graph_trace=trace and variant in {'vector_graph_rrf','hybrid_v2_no_rerank'})
    if trace:
     trace_embedding=result['trace_diagnostics']['query_embedding']; assert set(trace_embedding)=={'vector_sha256'} and trace_embedding['vector_sha256']==snapshot.vector_hash_for(run_id='r',plan=plan(),query_plans_hash='plans',embedding=IDENTITY)
  assert provider.calls==frozen_calls and production.vectors==[(1.0,0.0,0.0)]*8
 asyncio.run(run())

def test_partial_snapshot_resumes_only_missing_query_ordinals():
 async def run():
  provider=Provider(); plans=[{'question_id':'Q01','queries':['one','two'],'plan_hash':'plan'}]
  snapshot=FrozenQueryEmbeddingSnapshot.empty(run_id='r',query_plans_hash='plans',embedding=IDENTITY)
  persisted=[]
  original=provider.aembed_query
  async def fail_second(query):
   if provider.calls==1: raise RuntimeError('stop')
   return await original(query)
  provider.aembed_query=fail_second
  with pytest.raises(RuntimeError): await snapshot.freeze_missing(plans=plans,run_id='r',query_plans_hash='plans',embedding=IDENTITY,provider=provider,persist_partial=lambda value:persisted.append(copy.deepcopy(value)))
  assert provider.calls==1 and len(snapshot.payload['records'])==1 and 'snapshot_hash' not in snapshot.payload
  provider.aembed_query=original
  stats=await snapshot.freeze_missing(plans=plans,run_id='r',query_plans_hash='plans',embedding=IDENTITY,provider=provider,persist_partial=lambda value:persisted.append(copy.deepcopy(value)))
  assert stats=={'planned_query_count':2,'provider_calls':1,'skipped_records':1,'new_records':1} and provider.calls==2 and len(snapshot.payload['records'])==2 and snapshot.payload['snapshot_hash']
 asyncio.run(run())
