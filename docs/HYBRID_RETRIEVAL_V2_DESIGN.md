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
| `RERANK_ENABLED` | `false` |
| `RERANK_INPUT_TOP_K` | `30` |
| `RERANK_OUTPUT_TOP_K` | `12` |
| `RERANK_TIMEOUT_SECONDS` | `10` |
| `RERANK_MAX_ATTEMPTS` | `2` |
| `PARENT_EXPANSION_ENABLED` | `false` |
| `FINAL_CONTEXT_TOP_K` | `8` |
| `FINAL_CONTEXT_TOKEN_BUDGET` | `6000` estimated tokens |

This design targets a small-to-medium single-instance knowledge base. It does
not claim million-document search capacity; future scale work can evaluate a
dedicated search service without changing the catalog truth model.

## Deliberate next steps

Phase C implements RRF only through the internal service, without changing V1
defaults. Later phases may feed its Top-30 into reranking, bounded Parent
Expansion, safe Retrieval Trace, and controlled offline A/B evaluation.

## Phase D: rerank, parent expansion, and final context budget

Phase D completes an internal-only context builder:

`RRF Top-30 -> rerank Top-12 -> Child-to-Parent expansion -> complete-unit budget -> FinalContext Top-8`.

Recall remains responsible for finding candidates; reranking only reorders the
already fused RRF pool. `Reranker` is a provider-neutral protocol. The
default `DisabledReranker` preserves RRF order, and `FakeReranker` is used only
by deterministic tests. `ConfigurableModelReranker` accepts an injected
provider protocol, uses `RERANK_TIMEOUT_SECONDS` and bounded
`RERANK_MAX_ATTEMPTS`, and strictly validates that the returned candidate IDs
are exactly the supplied set with finite scores. Timeout, provider failure, or
malformed responses safely fall back to RRF order with a reason code; they do
not make retrieval unavailable. No BGE model, endpoint, package, or GPU runtime
is bundled or claimed to be in use.

Reranking uses the normalized query already produced by a shared plan. It does
not rewrite the query. Child content is sent once per fused ID. Graph evidence
is rendered as stable directed triples with `subject`, exact `predicate`,
`object`, and `direction`; it is never a Python dictionary string and cannot
silently turn `PROVIDES_INDEX` into `DEPENDS_ON`. Successful rerank sorting is
`rerank_score desc`, `rrf_score desc`, `pre_rerank_rank asc`, then candidate ID.
RRF and rerank scores are retained separately and are never manually blended.

Parent expansion happens **after** reranking and only for a child with an exact
`(document_id, document_version, parent_chunk_id)`. The catalog parent must be
`kind=parent`, `status=ready`, and `is_current=true`; historical versions are
never used. Multiple children that hit the same parent produce one parent
context retaining all supporting child/candidate IDs and the best rerank rank.
Missing parents and legacy children without `parent_chunk_id` become a child
fallback; no parent is guessed from filename, index, text proximity, or source.
Graph candidates never expand to a parent and remain independent graph evidence
contexts.

`FinalContext` is the common V2 output, with safe source/document/version
provenance, supporting IDs, retrieval types, RRF/rerank scores, deterministic
rank, and an `estimated_token_count`. Parent contexts dedupe by exact
document/version/parent ID, child fallbacks by chunk ID, and graph evidence by
its evidence identity. A restricted allowlist remains fail-closed throughout;
an empty scope returns no contexts and an out-of-scope parent or graph candidate
cannot become an unscoped fallback.

The complete-unit budget uses estimates rather than claiming model tokenizer
tokens. It selects ranked contexts up to `FINAL_CONTEXT_TOP_K=8` and
`FINAL_CONTEXT_TOKEN_BUDGET=6000`. It never slices parent text. If a parent
does not fit but its best supporting child fits, that child is substituted;
otherwise the candidate is dropped. Graph evidence is budgeted like every other
context. Safe diagnostics record rerank, parent expansion, graph, and budget
counts only—never queries, prompts, full content, credentials, or paths.

`ContextBuilderV2` is not imported by `QAAgent` or the public API. Defaults
remain `HYBRID_RETRIEVAL_V2_ENABLED=false`, `RERANK_ENABLED=false`, and
`PARENT_EXPANSION_ENABLED=false`; ordinary `/api/qa/ask` remains the V1
vector-plus-graph heuristic path.
