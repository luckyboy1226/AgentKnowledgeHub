import asyncio
from services.hybrid_retrieval_v2_runtime import HybridRetrievalV2RealRuntime, RealRuntimeStorageVerifier, build_real_runtime
import pytest

class Coordinator:
    def __init__(self): self.calls=[]
    async def create_document_version(self,**kw): self.calls.append(kw); return {'document_id':'00000000-0000-4000-8000-000000000001','version':1,'status':'ready','content_hash':'h'}
class Services:
    async def verify_document(self,_): return {'ready':True}
    async def verify_scope(self,_): return {'mongo':True,'chroma':True,'neo4j':True,'bm25':True}
    async def run_variant(self,plan,variant,allow,**kw):
        audit=kw['audit']; audit.vector_searches+=1; audit.bm25_searches+=int(kw['bm25_enabled']); audit.graph_queries+=int(bool(kw['entities'])); return {'final_context_ids':[variant],'document_ranks':sorted(allow),'candidate_ranks':[1],'latency':{},'graph_metrics':{}}
class Plan:
    async def build_evaluation_query_plan(self,q,**kw):
        return type('P',(),{'question_id':kw['question_id'],'normalized_query':q,'queries':(q,),'entities':('E',),'keywords':('K',),'intent':type('I',(),{'value':'factoid'})()})()

def test_construction_zero_io_and_bridge_routes():
 async def run():
  c=Coordinator(); r=HybridRetrievalV2RealRuntime(coordinator_factory=lambda:c,services_factory=Services,plan_factory=Plan)
  assert c.calls==[] and r.audit.vector_searches==0
  await r.ingest({'id':'D1','filename':'a.txt','content':'x'},'r'); assert c.calls[0]['logical_key']=='D1'
  plan=await r.build_query_plan({'question_id':'Q1','question':'question'},'r')
  for v, bm, graph in [('vector_only',0,0),('bm25_vector_rrf',1,0),('vector_graph_rrf',0,1),('hybrid_v2_no_rerank',1,1)]:
   out=await r.evaluate(plan,v,frozenset({'00000000-0000-4000-8000-000000000001'}),graph_trace=v in {'vector_graph_rrf','hybrid_v2_no_rerank'}); assert out['call_audit']['bm25_searches']==bm and out['call_audit']['graph_queries']==graph and out['call_audit']['reranker_calls']==0
 asyncio.run(run())

def test_real_runtime_preflight_construction_never_calls_factories():
 class Settings:
  mongodb_uri='mongodb://127.0.0.1:27017'; chroma_host='127.0.0.1'; chroma_port=8000; neo4j_uri='bolt://127.0.0.1:7687'; embedding_dimensions=1024; resolved_embedding_space_id='space'
  embedding_config=type('E',(),{'provider':'qwen','model':'embedding'})()
 runtime=build_real_runtime(Settings())
 assert runtime.construction_metadata['io_performed'] is False
 assert runtime.construction_metadata['production_adapter_type']=='HybridRetrievalV2ProductionAdapter'

def test_verifier_requires_explicit_registry_and_verifies_complete_components():
 with pytest.raises(RuntimeError, match='DocumentRegistry'):
  RealRuntimeStorageVerifier({})
 class Collection:
  def find(self,*_): return [
   {'kind':'parent','status':'ready','is_current':True},
   {'kind':'child','status':'ready','is_current':True},
  ]
 class Registry:
  def __init__(self): self.operations=type('Ops',(),{'find_one':lambda *_:{'operation_id':'00000000-0000-4000-8000-000000000002','operation_type':'create','status':'succeeded','document_id':'00000000-0000-4000-8000-000000000001','logical_key':'D1','source':'a.txt','version':1}})()
  def find(self,_): return {'document_id':'00000000-0000-4000-8000-000000000001','logical_key':'D1','filename':'a.txt','current_version':1,'status':'ready'}
  def versions_for(self,_): return [{'version':1,'status':'ready','is_current':True,'content_hash':'hash'}]
 class Chunks:
  collection=Collection()
  def list_current_children(self,_): return [{'kind':'child','document_id':'00000000-0000-4000-8000-000000000001','document_version':1,'status':'ready','is_current':True,'chunk_id':'c','content':'text'}]
 class Vectors:
  def _version_records(self,document_id,version): return [('v',{'document_id':document_id,'document_version':version,'status':'ready','is_current':True})]
 class Graph:
  async def execute_cypher(self,*_): return [{'status':'ready','is_current':True,'content_hash':'hash'}]
 class BM25:
  @staticmethod
  def _eligible(_): return True
 async def run():
  verifier=RealRuntimeStorageVerifier({'DocumentRegistry':Registry(),'ChunkRepository':Chunks(),'VectorStoreService':Vectors(),'KnowledgeGraphService':Graph(),'BM25Retriever':BM25()})
  result=await verifier.verify_document('00000000-0000-4000-8000-000000000001')
  assert all(result.values())
  recovered=await verifier.verify_recovery_identity(logical_key='D1',source='a.txt',document_id='00000000-0000-4000-8000-000000000001',operation_id='00000000-0000-4000-8000-000000000002')
  assert recovered['status']=='ready' and recovered['document_version']==1
 asyncio.run(run())

def test_query_plan_only_uses_injected_planner_and_counts_two_llm_calls():
 class Planner:
  async def build_evaluation_query_plan(self, question, **kwargs):
   return type('P',(),{'question_id':kwargs['question_id'],'normalized_query':question,'queries':(question,), 'entities':(), 'keywords':(), 'intent':type('I',(),{'value':'factoid'})()})()
 async def run():
  runtime=HybridRetrievalV2RealRuntime(coordinator_factory=lambda: (_ for _ in ()).throw(AssertionError('coordinator')),services_factory=lambda: (_ for _ in ()).throw(AssertionError('storage')),plan_factory=Planner)
  plan=await runtime.build_query_plan({'question_id':'Q1','question':'only planning'},'run')
  assert plan['queries']==['only planning'] and runtime.audit.query_plan_llm_calls==2
 asyncio.run(run())
