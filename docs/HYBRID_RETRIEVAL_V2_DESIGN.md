# Hybrid Retrieval V2 — Phase B

## Scope

Phase B introduces a deterministic BM25 derived index and a common
`RetrievalCandidate` contract. It does **not** change ordinary
`/api/qa/ask`: the default path remains V1 vector retrieval + graph retrieval
+ current heuristic rerank + Top-K + LLM.

`HYBRID_RETRIEVAL_V2_ENABLED=false` and `BM25_ENABLED=false` are the defaults.
The V2 services are internal building blocks; they are not an API query switch,
reranker, Parent Expansion, retrieval trace, or A/B evaluation implementation.

## Responsibilities

| Retriever | Purpose | Source of truth |
|---|---|---|
| BM25 | exact identifiers, error codes, API names, versions, names and phrases | ready/current Child catalog rows |
| Vector | semantic similarity | existing Chroma child vectors |
| Graph | entity relations and bounded traversal | existing Neo4j evidence |

BM25 complements rather than replaces vector retrieval. Parent chunks never
enter BM25: they are retained solely for a future Parent Context Expansion.

## Candidate contract

`RetrievalCandidate` always has a stable `candidate_id`, safe source,
document/version provenance, child/parent identifiers where applicable,
`retrieval_type` (`bm25`, `vector`, `graph`), raw score, rank, and structured
metadata. It intentionally has no RRF or rerank score yet.

- Child candidates use durable `chunk_id`; missing legacy IDs receive an
  explicit SHA-256 `legacy-child:` fallback.
- Graph candidates use `evidence_key`; multi-edge candidates use SHA-256 of
  document/version plus ordered evidence keys.
- Text prefixes and Python `hash()` are never used as durable identities.

## Phase C: reciprocal-rank fusion

`RRFFusion` consumes only the three independently ranked lists already returned
by `HybridRetrieverV2`; it never calls a retriever, relaxes scope/lifecycle
filters, expands parents, or calls an LLM. Raw BM25, vector, and graph scores
are intentionally **not** summed or compared because they use incomparable
score spaces.

The standard, unweighted calculation is
`RRF(candidate) = Σ 1 / (RRF_K + rank_i)`, with one-based ranks and default
`RRF_K=60`. It uses no source boost, graph boost, query-type weight, or other
heuristic. The internal output pool defaults to `RRF_FUSION_TOP_K=30`; this is
a future reranker input pool, not the final LLM Top-8.

Child candidates with the same stable `candidate_id` can merge across BM25 and
vector. Graph evidence remains distinct because its identity is an
`evidence_key` or ordered evidence-path digest, not its rendered text. A
`FusedCandidate` retains `retrieval_types`, one-based `source_ranks`,
per-source `raw_scores`, and per-source metadata under
`metadata.retrieval_metadata`. Shared durable provenance (document/version,
child/parent, safe source) must agree exactly; a conflict rejects the whole
identity instead of silently selecting one source.

Malformed ranks (missing, non-positive, or duplicated in a source list),
invalid source types, inactive/non-ready candidates, or explicit scope
conflicts are rejected. `RRFFusionResult.diagnostics` exposes only safe
counts: inputs, unique/multi-source candidates, invalid/conflicting candidates
and outputs. Sorting is deterministic: RRF descending, source count
descending, best source rank ascending, then lexical candidate ID.

## Derived BM25 index

`BM25Retriever` is an in-process, deterministic index rebuilt from Mongo
`document_chunks` rows that are exactly `kind=child`, `status=ready`, and
`is_current=true`. Mongo remains the source of truth.

The tokenizer NFKC-normalizes and case-folds input, preserves complete latin,
numeric, path, and identifier tokens such as `E1007`, `Redis-7.2`, and
`/api/qa/ask`, and emits a complete CJK phrase plus deterministic CJK bigrams.
It uses no external Chinese NLP service.

The Coordinator sends only a best-effort `mark_stale()` notification after a
successful Parent–Child activation or catalog cleanup. The next BM25 search
rebuilds from catalog. A rebuild failure or `BM25_MAX_INDEXED_CHILDREN` limit
marks the index unavailable/stale and returns no BM25 results; it never rolls
back or invalidates the completed document Saga.

## Version and scope safety

Eligibility is checked during rebuild and again before Candidate generation:
processing, failed, deleted, old-current and parent rows cannot be returned.
When an `allowed_document_ids` scope is present, only exact IDs in that set are
returned. An empty scope returns three empty lists; it never falls back to an
unscoped vector or graph query.

## Configuration

| Setting | Default |
|---|---:|
| `HYBRID_RETRIEVAL_V2_ENABLED` | `false` |
| `BM25_ENABLED` | `false` |
| `BM25_TOP_K` | `20` |
| `VECTOR_V2_TOP_K` | `20` |
| `GRAPH_V2_TOP_K` | `20` |
| `BM25_MAX_INDEXED_CHILDREN` | `50000` |
| `RRF_ENABLED` | `true` (internal only) |
| `RRF_K` | `60` |
| `RRF_FUSION_TOP_K` | `30` |

This design targets a small-to-medium single-instance knowledge base. It does
not claim million-document search capacity; future scale work can evaluate a
dedicated search service without changing the catalog truth model.

## Deliberate next steps

Phase C implements RRF only through the internal service, without changing V1
defaults. Later phases may feed its Top-30 into reranking, bounded Parent
Expansion, safe Retrieval Trace, and controlled offline A/B evaluation.
