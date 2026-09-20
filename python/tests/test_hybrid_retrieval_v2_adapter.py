import asyncio
from types import SimpleNamespace
from services.hybrid_retrieval_v2_adapter import HybridRetrievalV2ProductionAdapter

class Hybrid:
 def __init__(self): self.calls=[]
 async def retrieve(self,q,**kw):
  self.calls.append(kw); c=lambda n:SimpleNamespace(candidate_id=n,document_id='doc-a',rank=1)
  return {'vector':[c('v')],'bm25':[c('b')],'graph':[c('g')]}
class Fusion:
 def __init__(self):self.calls=[]
 def fuse(self,lists,**kw): self.calls.append(lists); return SimpleNamespace(candidates=[SimpleNamespace(candidate_id='f',document_id='doc-a')])
class Builder:
 reranker=__import__('retrieval.reranker',fromlist=['DisabledReranker']).DisabledReranker()
 def __init__(self):self.calls=[]
 async def build(self,q,candidates,**kw):self.calls.append((q,candidates,kw));return SimpleNamespace(contexts=[SimpleNamespace(context_id='p',document_id='doc-a',final_rank=1)])
class Audit: vector_searches=bm25_searches=graph_queries=reranker_calls=0

def test_explicit_variant_component_paths_and_scope():
 async def run():
  h=Hybrid();f=Fusion();b=Builder();a=HybridRetrievalV2ProductionAdapter(hybrid_factory=lambda:h,fusion_factory=lambda:f,context_builder_factory=lambda:b); plan={'queries':['q'],'entities':['E']}; scope=frozenset({'doc-a'})
  for v,sel,fuse,build in [('vector_only',(False,False),0,0),('bm25_vector_rrf',(True,False),1,0),('vector_graph_rrf',(False,True),2,0),('hybrid_v2_no_rerank',(True,True),3,1)]:
   out=await a.run_variant(plan,v,scope,audit=Audit()); call=h.calls[-1]; assert call['selection']['bm25']==sel[0] and call['selection']['graph']==sel[1]; assert len(f.calls)==fuse and len(b.calls)==build
   if build: assert b.calls[-1][2]['allowed_document_ids']==scope
 asyncio.run(run())
