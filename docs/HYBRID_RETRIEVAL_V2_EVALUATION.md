# Hybrid Retrieval V2 — Phase F Evaluation

Phase F is an offline A/B framework. It does not select the production retrieval
path, invoke a provider, access a database, or alter `/api/qa/ask`.

## Variants and fairness

`vector_only`, `bm25_vector_rrf`, `vector_graph_rrf`, and `hybrid_v2_full`
receive the same fixture-owned corpus, question plan, allowlisted scope, embedding
configuration, and final K. A real adapter must generate an `EvaluationQueryPlan`
once per question and pass that identical plan to every variant. Empty scopes fail
closed; they must never become whole-corpus retrieval.

The first implementation is explicitly **fake-only**. It exercises exact
identifier, semantic, multi-hop, version conflict, unanswerable, single-hop,
cross-document, and distractor fixture rows. Its results validate contracts only;
they are not evidence of production quality.

## Schema and metrics

`results.json` contains one immutable `question_id + variant` row with
`retrieval_metrics`, `graph_metrics`, `latency`, `final_context_ids`,
`source_ids`, `answer_metrics`, and `diagnostics`. Relevance is currently
document-level: Recall@5/10 is retrieved-relevant documents divided by all relevant
documents; MRR uses the first relevant rank; binary nDCG@10 uses standard DCG/IDCG.
There is no passage-level claim without chunk ground truth.

Graph metrics are `null` with `applicable=false` when graph was not used, never
zero. Expected edges compare subject, canonical predicate, object, **and direction**.
A multi-hop path is complete only when every expected directed edge is present.

Stage latency fields use `null` for a disabled stage. Summary reports P50/P95 and
marks samples below 30 as small. No dollar costs are inferred.

Failure labels are evidence-bound: `retrieval_miss`, `graph_edge_missing`,
`budget_drop`, `unsupported_answer`, or `insufficient_evidence`. The harness does
not attribute an unsupported result to a reranker without rank evidence.

## Run protocol

Run only the deterministic fixture:

```powershell
python scripts/run-rag-eval.py --v2-fake --run-id phase-f-local
python scripts/run-rag-eval.py --v2-fake --variants vector_only,bm25_vector_rrf --final-top-k 8
```

Outputs are atomically written under `.runtime/evaluation/<run_id>/` as
`results.json`, `summary.json`, and `summary.md`. Existing completed pairs are
preserved on recovery. A future real run needs an immutable reviewed benchmark with
document/chunk relevance, a verified uploaded-document scope, a fixed query plan,
and explicit `--authorized-real-run`-style authorization. That adapter is not part
of Phase F fake validation.
