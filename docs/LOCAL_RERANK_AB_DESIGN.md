# Local BGE paired rerank A/B

R2 compares two retrieval arms built from one immutable RRF Top-20 pool:

- `hybrid_v2_rrf_top20_no_rerank`: preserve RRF order;
- `hybrid_v2_rrf_top20_local_bge_rerank`: reorder the same 20 candidate IDs
  with the local BGE provider.

Both arms use the same frozen Query Plan, query embedding snapshot, UUID
allowlist, Vector/BM25/Graph recall, RRF settings, parent expansion and final
Top-8 budget. The adapter is invoked once per planned query and returns both
arms, so retrieval cannot drift between arms. Candidate IDs and pre-rerank
ranks must match exactly.

## Failure semantics

Normal QA retains the availability-oriented ContextBuilder fallback to RRF.
The benchmark is stricter: missing/duplicate IDs, non-finite scores, timeout,
provider failure or any fallback marks the rerank arm `rerank_failed`. It may
not be reported as a successful rerank result. Scope violations and any pool
size other than 20 fail closed.

## Frozen and read-only gates

A future real run must validate the fixture, Query Plan, allowlist and query
embedding snapshot hashes, plus the local model identity. Device is fixed to
CPU, input/output sizes are 20/8, and planned writes must be zero. It may only
read the existing Chroma, MongoDB, Neo4j and derived BM25 data. Chat and
Embedding calls, ingestion and cleanup are forbidden.

The historical no-rerank report remains an immutable reference. Since it used
RRF Top-30 rather than this paired Top-20 protocol, only its compatible
retrieval fields may be cited descriptively. Rerank effect claims must use the
new paired run.

## Metrics

The report contains Recall@1/@3/@5, MRR@8, nDCG@8, source coverage,
zero-result rate, changed-Top-1/Top-8 rates, selected candidate rank deltas,
rerank success, rerank/retrieval P50 and P95, and component call counts. Model
cold-load latency is recorded separately. No answer accuracy or LLM quality
metric is produced.

## Offline contract check

```powershell
conda run -n kghub python scripts/run-hybrid-v2-rerank-ab.py `
  --offline --run-id g2-real-ingestion-20260920-02 `
  --ab-run-id hybrid-v2-rerank-ab-offline
```

This command is fake-only: it does not load the local model or open a database.
Real mode remains blocked pending a separate authorization.
