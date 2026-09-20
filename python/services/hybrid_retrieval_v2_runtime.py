"""Lazy runtime adapters for Phase G; construction performs zero I/O."""
from __future__ import annotations
import uuid
from typing import Any
from services.hybrid_retrieval_v2_real_evaluation import GRAPH_VARIANTS

class FakePhaseGRuntime:
    """Pure fake adapter for CLI and lifecycle tests; no network dependencies."""
    async def ingest(self,d:dict[str,Any],run_id:str)->dict[str,Any]: return {"document_id":str(uuid.uuid5(uuid.NAMESPACE_URL,run_id+str(d['id']))),"version":1,"content_hash":"fake"}
    async def verify_document(self,_:str)->dict[str,Any]: return {k:True for k in ('ready','current','parents_ready','children_ready','vectors_current','graph_provenance_current')}
    async def verify_scope(self,_)->dict[str,Any]: return {k:True for k in ('mongo','chroma','neo4j','bm25')}
    async def build_query_plan(self,q:dict[str,Any],_:str)->dict[str,Any]: return {'queries':[q['question_id']],'entities':[],'keywords':[],'intent':'factoid'}
    async def evaluate(self,p:dict[str,Any],v:str,allow,*,trace:bool,graph_trace:bool)->dict[str,Any]:
        docs=sorted(allow); return {'final_context_ids':[f'{v}:{p["question_id"]}'],'document_ranks':docs[:8],'candidate_ranks':[1], 'latency':{'retrieval_total_ms':1.0},'graph_metrics':{'applicable':v in GRAPH_VARIANTS},'call_audit':{'vector_searches':1,'bm25_searches':int(v in {'bm25_vector_rrf','hybrid_v2_no_rerank'}),'graph_queries':int(v in GRAPH_VARIANTS),'reranker_calls':0}}

class RuntimeCallAudit:
    def __init__(self): self.query_plan_llm_calls=self.embedding_queries=self.vector_searches=self.bm25_searches=self.graph_queries=self.reranker_calls=0
    def snapshot(self): return self.__dict__.copy()
    def delta(self,before): return {k:getattr(self,k)-before[k] for k in before}

class HybridRetrievalV2RealRuntime:
    """Lazy composition point for Coordinator, BM25, HybridRetrieverV2, RRF and ContextBuilderV2.
    Factories are injected so constructor cannot allocate clients or invoke providers.
    """
    def __init__(self, *, coordinator_factory:Any, services_factory:Any, plan_factory:Any, production_adapter_factory:Any|None=None):
        self._coordinator_factory=coordinator_factory; self._services_factory=services_factory; self._plan_factory=plan_factory; self._production_adapter_factory=production_adapter_factory; self.audit=RuntimeCallAudit()
    async def ingest(self, document, run_id):
        coordinator=self._coordinator_factory(); result=await coordinator.create_document_version(filename=document['filename'],content=document['content'].encode() if isinstance(document['content'],str) else document['content'],logical_key=document['id'],namespace='evaluation',operation_id=str(uuid.uuid4()))
        return {'document_id':result['document_id'],'version':result.get('version'),'status':result.get('status'),'content_hash':result.get('content_hash'),'source':document['filename']}
    async def verify_document(self,document_id):
        s=self._services_factory(); return await s.verify_document(document_id)
    async def verify_scope(self,allow):
        if not allow: raise ValueError('empty_scope')
        s=self._services_factory(); return await s.verify_scope(allow)
    async def build_query_plan(self,question,run_id):
        builder=self._plan_factory(); self.audit.query_plan_llm_calls+=1; plan=await builder.build_evaluation_query_plan(question['question_id'],run_id=run_id,question_id=question['question_id'])
        return {'question_id':plan.question_id,'queries':[plan.normalized_query],'entities':list(plan.entities),'keywords':[],'intent':getattr(plan.intent,'value',plan.intent)}
    async def evaluate(self,plan,variant,allow,*,trace=False,graph_trace=False):
        if variant not in {'vector_only','bm25_vector_rrf','vector_graph_rrf','hybrid_v2_no_rerank'}: raise ValueError('invalid_variant')
        if graph_trace and variant not in GRAPH_VARIANTS: raise ValueError('graph_trace_not_applicable')
        before=self.audit.snapshot()
        if self._production_adapter_factory is not None:
            result=await self._production_adapter_factory().run_variant(plan,variant,frozenset(allow),trace=trace,graph_trace=graph_trace,audit=self.audit)
        else:
            s=self._services_factory(); entities=plan.get('entities',[]) if variant in GRAPH_VARIANTS else []; bm25=variant in {'bm25_vector_rrf','hybrid_v2_no_rerank'}
            result=await s.run_variant(plan,variant,frozenset(allow),entities=entities,bm25_enabled=bm25,trace=trace,graph_trace=graph_trace,audit=self.audit)
        return {**result,'call_audit':self.audit.delta(before)}
