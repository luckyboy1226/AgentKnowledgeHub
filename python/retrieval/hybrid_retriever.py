"""Internal V2 three-way recall orchestrator; deliberately no fusion yet."""

from __future__ import annotations

from typing import Any

from retrieval.candidates import RetrievalCandidate, from_graph_record, from_vector_result


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
    ) -> dict[str, list[RetrievalCandidate]]:
        # A caller that deliberately supplies an empty allowlist must never
        # degrade into an unscoped vector/graph query.
        if allowed_document_ids is not None and not allowed_document_ids:
            return {"bm25": [], "vector": [], "graph": []}
        bm25_candidates = (
            await self.bm25.search(query, bm25_top_k, allowed_document_ids=allowed_document_ids)
            if bm25_enabled and self.bm25 is not None else []
        )
        vector_rows = await self.vector_store.search(
            query, top_k=vector_top_k, allowed_document_ids=allowed_document_ids
        )
        vector_candidates = [from_vector_result(record, score, rank) for rank, (record, score) in enumerate(vector_rows, start=1)]
        graph_candidates: list[RetrievalCandidate] = []
        for entity in entities or []:
            records = await self.knowledge_graph.get_neighbors(
                entity, hops=2, limit=graph_top_k, allowed_document_ids=allowed_document_ids
            )
            for record in records:
                graph_candidates.append(from_graph_record(record, len(graph_candidates) + 1, raw_score=0.8))
                if len(graph_candidates) >= graph_top_k:
                    break
            if len(graph_candidates) >= graph_top_k:
                break
        return {"bm25": bm25_candidates, "vector": vector_candidates, "graph": graph_candidates}
