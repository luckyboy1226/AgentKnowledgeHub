"""Internal V2 three-way recall orchestrator; deliberately no fusion yet."""

from __future__ import annotations

import time
from typing import Any

from retrieval.candidates import RetrievalCandidate, from_graph_record, from_vector_result
from retrieval.trace_support import emit


class HybridRetrieverV2:
    """Return independently ranked candidate lists for the later RRF phase."""

    def __init__(self, *, bm25: Any | None, vector_store: Any, knowledge_graph: Any) -> None:
        self.bm25 = bm25
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph

    async def retrieve(
        self, query: str, *, entities: list[str] | None = None,
        bm25_top_k: int = 20, vector_top_k: int = 20, graph_top_k: int = 20,
        allowed_document_ids: frozenset[str] | None = None, bm25_enabled: bool = False,
        trace: Any | None = None, selection: dict[str, bool] | None = None,
        graph_trace: Any | None = None,
        query_embedding: list[float] | tuple[float, ...] | None = None,
    ) -> dict[str, list[RetrievalCandidate]]:
        emit(trace, "query_received", query)
        emit(trace, "query_rewritten", (query,), entity_count=len(entities or []), keyword_count=0)
        # A caller that deliberately supplies an empty allowlist must never
        # degrade into an unscoped vector/graph query.
        if allowed_document_ids is not None and not allowed_document_ids:
            for stage in ("bm25_retrieved", "vector_retrieved", "graph_retrieved"):
                emit(trace, "record_stage", stage, details={"input_count": 0, "output_count": 0}, latency_ms=0.0)
            return {"bm25": [], "vector": [], "graph": []}
        selected = {"bm25": bool(bm25_enabled), "vector": True, "graph": True} | (selection or {})
        bm25_started = time.monotonic()
        bm25_candidates = (
            await self.bm25.search(query, bm25_top_k, allowed_document_ids=allowed_document_ids)
            if selected["bm25"] and bm25_enabled and self.bm25 is not None else []
        )
        emit(trace, "record_stage", "bm25_retrieved", candidates=bm25_candidates,
             details={"input_count": 1 if bm25_enabled and self.bm25 is not None else 0, "output_count": len(bm25_candidates)},
             latency_ms=(time.monotonic() - bm25_started) * 1000)
        vector_started = time.monotonic()
        vector_kwargs={"top_k":vector_top_k,"allowed_document_ids":allowed_document_ids}
        if query_embedding is not None: vector_kwargs["query_embedding"]=query_embedding
        vector_rows = await self.vector_store.search(query, **vector_kwargs) if selected["vector"] else []
        vector_candidates = [from_vector_result(record, score, rank) for rank, (record, score) in enumerate(vector_rows, start=1)]
        emit(trace, "record_stage", "vector_retrieved", candidates=vector_candidates,
             details={"input_count": 1, "output_count": len(vector_candidates)},
             latency_ms=(time.monotonic() - vector_started) * 1000)
        graph_started = time.monotonic()
        graph_candidates: list[RetrievalCandidate] = []
        for entity in (entities or []) if selected["graph"] else []:
            records = await self.knowledge_graph.get_neighbors(
                entity, hops=2, allowed_document_ids=allowed_document_ids,
                graph_trace=graph_trace,
            )
            for record in records:
                graph_candidates.append(from_graph_record(record, len(graph_candidates) + 1, raw_score=0.8))
                if len(graph_candidates) >= graph_top_k:
                    break
            if len(graph_candidates) >= graph_top_k:
                break
        emit(trace, "record_stage", "graph_retrieved", candidates=graph_candidates,
             details={"input_count": len(entities or []), "output_count": len(graph_candidates)},
             latency_ms=(time.monotonic() - graph_started) * 1000)
        return {"bm25": bm25_candidates, "vector": vector_candidates, "graph": graph_candidates}
