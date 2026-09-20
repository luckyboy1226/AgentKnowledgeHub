"""Explicit, zero-I/O production composition for the Phase G runtime."""
from __future__ import annotations
from typing import Any
from observability.retrieval_trace import RetrievalTrace
from retrieval.reranker import DisabledReranker
from services.graph_evidence_trace import GraphEvidenceTrace

class HybridRetrievalV2ProductionAdapter:
    def __init__(self, *, hybrid_factory:Any, fusion_factory:Any, context_builder_factory:Any|None=None, trace_run_id:str='phase-g'):
        self._hybrid_factory=hybrid_factory; self._fusion_factory=fusion_factory; self._context_builder_factory=context_builder_factory; self._trace_run_id=str(trace_run_id)

    @staticmethod
    def _retrieval_diagnostics(trace: RetrievalTrace) -> dict[str, Any]:
        """Return only bounded observer metadata, never query text or content."""
        payload=trace.to_dict()
        stages=payload.get('stages', [])
        return {'enabled':True, 'stage_names':[str(row.get('stage')) for row in stages],
                'candidate_counts':[int((row.get('candidate_summary') or {}).get('original_count') or 0) for row in stages]}

    @staticmethod
    def _graph_diagnostics(trace: GraphEvidenceTrace) -> dict[str, Any]:
        """Keep graph observer output to counts and stage names only."""
        payload=trace.to_dict(); stages=payload.get('stages') or {}
        return {'enabled':True, 'stage_counts':{str(stage):len(rows) for stage,rows in sorted(stages.items())},
                'rejection_count':len(payload.get('rejections') or []),
                'ranked_edge_count':len(payload.get('rank_positions') or {})}

    async def run_variant(self, plan:dict[str,Any], variant:str, scope:frozenset[str], *, trace=None, graph_trace=None, audit=None, query_embedding=None)->dict[str,Any]:
        if not scope: raise ValueError('empty_scope')
        graph=variant in {'vector_graph_rrf','hybrid_v2_no_rerank'}; bm25=variant in {'bm25_vector_rrf','hybrid_v2_no_rerank'}
        if graph_trace and not graph: raise ValueError('graph_trace_not_applicable')
        hybrid=self._hybrid_factory(); selection={'vector':True,'bm25':bm25,'graph':graph}; query=(plan.get('queries') or [''])[0]
        question_id=str(plan.get('question_id') or 'unknown')
        retrieval_trace=RetrievalTrace(f'{self._trace_run_id}:{question_id}:{variant}') if trace is True else (trace if trace not in (False, None) else None)
        evidence_trace=GraphEvidenceTrace(run_id=self._trace_run_id,question_id=question_id,scope_verified=True,allowed_document_ids_count=len(scope)) if graph_trace is True else (graph_trace if graph_trace not in (False, None) else None)
        raw=await hybrid.retrieve(query,entities=plan.get('entities',[]) if graph else [],allowed_document_ids=scope,bm25_enabled=bm25,trace=retrieval_trace,selection=selection,graph_trace=evidence_trace,query_embedding=query_embedding)
        if audit is not None:
            audit.vector_searches+=int(bool(selection['vector'])); audit.bm25_searches+=int(bm25); audit.graph_queries+=int(graph and bool(plan.get('entities')))
        if variant=='vector_only':
            items=raw['vector']; result={'final_context_ids':[item.candidate_id for item in items], 'document_ranks':[item.document_id for item in items if item.document_id], 'candidate_ranks':[item.rank for item in items], 'latency':{},'graph_metrics':{'applicable':False}}
        else:
            lists={'bm25':raw['bm25'] if bm25 else [],'vector':raw['vector'],'graph':raw['graph'] if graph else []}; fused=self._fusion_factory().fuse(lists,trace=retrieval_trace).candidates
            if variant=='hybrid_v2_no_rerank':
                builder=self._context_builder_factory(); assert isinstance(getattr(builder,'reranker',DisabledReranker()),DisabledReranker)
                built=await builder.build(query,fused,allowed_document_ids=scope,trace=retrieval_trace); contexts=built.contexts
                result={'final_context_ids':[x.context_id for x in contexts],'document_ranks':[x.document_id for x in contexts if x.document_id],'candidate_ranks':[x.final_rank for x in contexts],'latency':{},'graph_metrics':{'applicable':True},'context_builder_used':True}
            else:
                result={'final_context_ids':[item.candidate_id for item in fused],'document_ranks':[item.document_id for item in fused if item.document_id],'candidate_ranks':list(range(1,len(fused)+1)),'latency':{},'graph_metrics':{'applicable':graph}}
        if isinstance(retrieval_trace, RetrievalTrace): result['trace_diagnostics']={'retrieval':self._retrieval_diagnostics(retrieval_trace)}
        if isinstance(evidence_trace, GraphEvidenceTrace): result.setdefault('trace_diagnostics',{})['graph']=self._graph_diagnostics(evidence_trace)
        return result
