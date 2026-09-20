"""In-memory contracts for stable same-score vector ordering in G4."""

from __future__ import annotations

import copy

import pytest

from observability.retrieval_trace import RetrievalTrace
from retrieval.candidates import RetrievalCandidate, from_vector_result
from retrieval.fusion import RRFFusion, rrf_sort_key
from services.vector_store import VectorStoreService


class Embeddings:
    provider_name = "fake"
    model_name = "fake"
    dimensions = 3
    async def aembed_query(self, _query): return [1.0, 0.0, 0.0]


class AlternatingEqualDistanceCollection:
    def __init__(self): self.calls = 0
    def query(self, **_kwargs):
        order = (("doc-b", "doc-a", "doc-c"), ("doc-a", "doc-b", "doc-c"))[self.calls % 2]
        self.calls += 1
        return {"documents":[[f"content-{value}" for value in order]],
                "metadatas":[[{"document_id":value,"document_version":1,"chunk_id":f"{value}:v1:c0",
                                "parent_chunk_id":f"{value}:v1:p0","source":f"{value}.txt",
                                "status":"ready","is_current":True} for value in order]],
                "distances":[[0.1,0.1,0.2]]}


def _bm25_candidate():
    return RetrievalCandidate(candidate_id="doc-c:v1:c0",content="content-doc-c",source="doc-c.txt",
                              document_id="doc-c",document_version=1,chunk_id="doc-c:v1:c0",
                              parent_chunk_id="doc-c:v1:p0",retrieval_type="bm25",raw_score=1.0,rank=1,
                              metadata={"status":"ready","is_current":True})


async def _run(service, trace):
    rows=await service.search("q",top_k=3,allowed_document_ids=frozenset({"doc-a","doc-b","doc-c"}))
    vectors=[from_vector_result(record,score,rank) for rank,(record,score) in enumerate(rows,start=1)]
    supplied={"bm25":[_bm25_candidate()],"vector":vectors,"graph":[]}
    before=copy.deepcopy(supplied)
    result=RRFFusion().fuse(supplied,trace=trace)
    assert supplied==before  # Trace and fusion do not mutate supplied candidates or metadata.
    return [{"candidate_id":item.candidate_id,"document_id":item.document_id,"source_ranks":item.source_ranks,
             "raw_scores":item.raw_scores,"rrf_score":item.rrf_score,"sort_key":rrf_sort_key(item)} for item in result.candidates]


@pytest.mark.asyncio
async def test_equal_distance_chroma_order_is_stable_for_off_off_on_on_and_off_on():
    service=VectorStoreService(Embeddings()); service._backend="chroma"; service._store=AlternatingEqualDistanceCollection()
    off_one=await _run(service,None); off_two=await _run(service,None)
    on_one=await _run(service,RetrievalTrace("on-one")); on_two=await _run(service,RetrievalTrace("on-two"))
    assert off_one==off_two==on_one==on_two
    by_id={row["candidate_id"]:row for row in off_one}
    assert by_id["doc-a:v1:c0"]["source_ranks"]=={"vector":1}
    assert by_id["doc-b:v1:c0"]["source_ranks"]=={"vector":2}


def test_equal_rrf_scores_use_candidate_id_as_the_last_stable_key():
    def candidate(identifier, retrieval_type):
        return RetrievalCandidate(candidate_id=identifier,content=identifier,source=f"{identifier}.txt",
                                  document_id=identifier,document_version=1,chunk_id=identifier,
                                  parent_chunk_id=None,retrieval_type=retrieval_type,raw_score=1.0,rank=1,
                                  metadata={"status":"ready","is_current":True})
    result=RRFFusion().fuse({"bm25":[candidate("b","bm25")],"vector":[candidate("a","vector")],"graph":[]})
    assert [item.candidate_id for item in result.candidates]==["a","b"]
    assert result.candidates[0].rrf_score==result.candidates[1].rrf_score
    assert rrf_sort_key(result.candidates[0]) < rrf_sort_key(result.candidates[1])
