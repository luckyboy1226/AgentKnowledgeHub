import asyncio
from services.hybrid_retrieval_v2_runtime import HybridRetrievalV2RealRuntime

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
        return type('P',(),{'question_id':kw['question_id'],'normalized_query':q,'entities':('E',),'intent':type('I',(),{'value':'factoid'})()})()

def test_construction_zero_io_and_bridge_routes():
 async def run():
  c=Coordinator(); r=HybridRetrievalV2RealRuntime(coordinator_factory=lambda:c,services_factory=Services,plan_factory=Plan)
  assert c.calls==[] and r.audit.vector_searches==0
  await r.ingest({'id':'D1','filename':'a.txt','content':'x'},'r'); assert c.calls[0]['logical_key']=='D1'
  plan=await r.build_query_plan({'question_id':'Q1'},'r')
  for v, bm, graph in [('vector_only',0,0),('bm25_vector_rrf',1,0),('vector_graph_rrf',0,1),('hybrid_v2_no_rerank',1,1)]:
   out=await r.evaluate(plan,v,frozenset({'00000000-0000-4000-8000-000000000001'}),graph_trace=v in {'vector_graph_rrf','hybrid_v2_no_rerank'}); assert out['call_audit']['bm25_searches']==bm and out['call_audit']['graph_queries']==graph and out['call_audit']['reranker_calls']==0
 asyncio.run(run())
