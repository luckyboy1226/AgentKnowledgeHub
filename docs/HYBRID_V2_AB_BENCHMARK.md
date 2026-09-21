# Hybrid V2 Retrieval A/B Benchmark

`hybrid-v2-ab-v1` defines a frozen-input, retrieval-only comparison across four modes:

- `vector_only`: dense vector retrieval only.
- `bm25_only`: lexical BM25 retrieval only.
- `graph_only`: provenance-scoped graph retrieval only.
- `full_hybrid_v2`: vector, BM25, and graph candidates combined by the existing Hybrid V2 fusion path. The current frozen configuration has reranking disabled.

This protocol measures retrieval behavior. It does not call an answer model and does not claim improvements in answer quality.

## Fairness and frozen inputs

Every mode receives the same frozen query plan, query ordinal, document UUID allowlist, fixture fingerprint, Top-K, and candidate budget. Query vectors must come from the run-scoped frozen snapshot; retrieval must not generate or replace embeddings. Candidate ordering uses score descending and candidate ID ascending as the deterministic final tie-break.

Trace collection is disabled for the A/B quality run. A failed mode produces an explicit failed row rather than being skipped. A real run fails closed unless the source run, query-plan hash, allowlist hash, fixture fingerprint, snapshot root, collection, embedding model, dimension, and embedding-space identity match the frozen inputs. Its output directory must be new and must not overwrite G4 or recovery artifacts.

## Metrics

The benchmark reports document-level `Recall@1/3/5/10`, `MRR@10`, and `nDCG@10`, plus source coverage, latency percentiles, and logical channel call counts. Graph-capable modes additionally report graph participation, graph evidence entering final Top-K, expected edge/path recall, direction fidelity, predicate fidelity, and provenance coverage.

Metrics whose cutoff exceeds the frozen final Top-K are `not available`; the runner does not extrapolate unseen ranks. Repeated chunks or graph evidence from one document are collapsed to the document's first occurrence before document-level DCG is calculated. Source coverage is computed from final Top-K sources only.

Graph metrics are `not available` for non-graph modes. Chunk-level recall is `not available` because the frozen ground truth does not provide relevant chunk IDs. Answer accuracy, answer quality, and LLM-as-a-judge scores are also `not available`; they must never be inferred from retrieval metrics.

## Outputs

Each run writes a new directory under `.runtime/evaluation/<ab_run_id>/` containing:

- `results.json`
- `results.csv`
- `summary.md`
- `metric-definitions.json`
- `safe-run-metadata.json`
- `input-hashes.json`

Artifacts contain safe identities, counts, metrics, hashes, and bounded failure summaries. They do not contain question text, document or chunk bodies, embedding vectors, prompts, credentials, connection strings, or absolute paths.

## Relationship to G4 and cleanup

G4 establishes trace ON/OFF equivalence and remains an immutable audit artifact. This A/B protocol measures retrieval quality and does not modify G4 state. It performs no document writes and no cleanup. Cleanup remains a separately authorized operation based on exact document UUIDs.

## Offline validation

```powershell
conda run -n kghub python scripts/run-hybrid-v2-ab.py `
  --offline `
  --ab-run-id hybrid-v2-ab-offline
```

Offline mode uses deterministic fake channels and produces all four mode reports without network or database access. Its scores validate framework behavior only and are not evidence that one production retrieval mode is better than another.
