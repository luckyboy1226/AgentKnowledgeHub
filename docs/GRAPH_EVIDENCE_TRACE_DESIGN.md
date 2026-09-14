# S4.6a Graph evidence trace design

## Purpose and boundary

This is an evaluation-only, in-memory trace for locating the first observed
loss of a directed GraphRAG edge. It is not enabled by `/api/qa/ask`, does not
create a database/provider client, and retains no document text, prompt,
credential, URI, or full record dump. A trace belongs to exactly one scoped
`graph_rag` question and is passed explicitly; `vector_only` neither calls the
graph nor creates a graph trace.

## Safe edge contract

Each trace edge has a stable fingerprint plus only `subject`, canonical and
raw predicate, `object`, traversal direction, document UUID/version, safe
source basename, evidence key, and relation-semantics version. The collector
records the stages `extracted`, `normalized`, `persisted`, `retrieved_raw`,
`scope_accepted`, `scope_rejected`, `relevance_scored`, `ranked`,
`entered_final_top_k`, and `entered_prompt`. Rejection values are a finite
enum; no exception body is persisted.

S4.6a adds optional sink propagation through `DocumentProcessorAdapter`,
`DocumentUpdateCoordinator`, and `KnowledgeGraphService.stage_document_version`
for extraction, normalization, dangling-endpoint rejection, and persisted
evidence. The public API never supplies this sink. A future authorized
evaluation factory can inject one shared run/question-scoped collector; absent
snapshots remain missing evidence, never a guessed root cause.

## Scope and read boundary

The future snapshot helper `list_scoped_evaluation_evidence()` accepts only a
nonempty set of UUID document IDs, uses parameterized Cypher, and returns only
ready/current provenance-complete evidence. Legacy relations and full-graph
fallbacks cannot satisfy it. It is internal, read-only, and not called by
ordinary QA. Scoped evaluation performs the existing second validation before
formatting context; only accepted evidence can be ranked or enter the prompt.

## First-loss policy

`graph_trace_diagnosis` scores expected paths edge by edge, preserving
subject/predicate/object/direction separately. It can classify
`extraction_missing`, `normalization_changed`, `persistence_missing`,
`retrieval_missing`, `scope_filtered`, `relevance_filtered`,
`top_k_truncated`, or `prompt_present` **only** when adjacent snapshots prove
the transition. Missing snapshots produce `insufficient_evidence`. A
multi-hop path is complete only when every expected edge entered the final
prompt; no path-level inference fills a missing edge.

## Safe outputs

Future runner invocations write a separate atomic `graph-evidence-trace.json`,
`graph-evidence-trace-summary.json`, and Markdown summary beside results under
`.runtime/evaluation/<run_id>/`. They never rewrite an existing result file.
Offline output is explicitly labelled `fake-only trace verification; not real
GraphRAG quality evidence`.

## Future small authorized diagnosis run

Before any real run, request a separate authorization for 6–10 fixed fixture
questions selected from the failure matrix (recommended: Q03, Q05, Q08 plus
three relation-fidelity failures and up to four control cases). The request
must enumerate the frozen synthetic document texts, expected query-plan/model
call budget, exact uploaded UUID allowlist, and exact cleanup UUIDs. Upload
only those fixture documents needed by the selected questions; use a fresh
run ID, no history/memory, and clean only their Chroma/Neo4j provenance via
the versioned delete saga while retaining Mongo tombstones/audit. Do not merge
those results with the immutable enterprise baseline.
# Subset execution

The evaluation runner accepts optional `--document-ids` and `--question-ids`
comma-separated selections. It always validates the entire reviewed fixture
first, then preserves fixture order for the selected corpus and questions.
Every selected question must have all declared source files in the selected
document set; otherwise the run fails before any real upload or model call.

Subset runs persist a deterministic selection fingerprint, selected IDs,
fixture/upload dual hashes, exact uploaded UUIDs, scope, and completed pairs.
`--recover-run` reloads that selection rather than widening to the full
fixture. A subset run is a diagnostic artifact and must not be compared as an
overall replacement for a complete benchmark.
