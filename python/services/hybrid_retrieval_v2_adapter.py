"""Explicit, zero-I/O production composition for the Phase G runtime."""
from __future__ import annotations
from typing import Any
from retrieval.reranker import DisabledReranker

class HybridRetrievalV2ProductionAdapter:
    def __init__(self, *, hybrid_factory:Any, fusion_factory:Any, context_builder_factory:Any|None=None):
        self._hybrid_factory=hybrid_factory; self._fusion_factory=fusion_factory; self._context_builder_factory=context_builder_factory
    async def run_variant(self, plan:dict[str,Any], variant:str, scope:frozenset[str], *, trace=None, graph_trace=None, audit=None)->dict[str,Any]:
        if not scope: raise ValueError('empty_scope')
        graph=variant in {'vector_graph_rrf','hybrid_v2_no_rerank'}; bm25=variant in {'bm25_vector_rrf','hybrid_v2_no_rerank'}
        if graph_trace is not None and not graph: raise ValueError('graph_trace_not_applicable')
        hybrid=self._hybrid_factory(); selection={'vector':True,'bm25':bm25,'graph':graph}; query=(plan.get('queries') or [''])[0]
        raw=await hybrid.retrieve(query,entities=plan.get('entities',[]) if graph else [],allowed_document_ids=scope,bm25_enabled=bm25,trace=trace,selection=selection)
        if audit is not None:
            audit.vector_searches+=int(bool(selection['vector'])); audit.bm25_searches+=int(bm25); audit.graph_queries+=int(graph and bool(plan.get('entities')))
        if variant=='vector_only':
            items=raw['vector']; return {'final_context_ids':[item.candidate_id for item in items], 'document_ranks':[item.document_id for item in items if item.document_id], 'candidate_ranks':[item.rank for item in items], 'latency':{},'graph_metrics':{'applicable':False}}
        lists={'bm25':raw['bm25'] if bm25 else [],'vector':raw['vector'],'graph':raw['graph'] if graph else []}; fused=self._fusion_factory().fuse(lists,trace=trace).candidates
        if variant=='hybrid_v2_no_rerank':
            builder=self._context_builder_factory(); assert isinstance(getattr(builder,'reranker',DisabledReranker()),DisabledReranker)
            built=await builder.build(query,fused,allowed_document_ids=scope,trace=trace); contexts=built.contexts
            return {'final_context_ids':[x.context_id for x in contexts],'document_ranks':[x.document_id for x in contexts if x.document_id],'candidate_ranks':[x.final_rank for x in contexts],'latency':{},'graph_metrics':{'applicable':True},'context_builder_used':True}
        return {'final_context_ids':[item.candidate_id for item in fused],'document_ranks':[item.document_id for item in fused if item.document_id],'candidate_ranks':list(range(1,len(fused)+1)),'latency':{},'graph_metrics':{'applicable':graph}}
