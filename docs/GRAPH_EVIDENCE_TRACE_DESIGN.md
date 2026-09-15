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

## S4.7a: immutable first-loss attribution

`scripts/analyze-graph-trace.py` is an offline-only companion for an existing
trace directory. It hashes `results.json`, `graph-trace.json`,
`safe-run-metadata.json`, and `ingestion-state.json` before and after analysis,
then writes a new derived directory only. It validates the run ID, selected
fixture IDs, selection fingerprint, graph trace question coverage, and the
expected vector/graph result pairs before output.

For every expected relation edge, the derived report records independent
subject/object endpoint, canonical-predicate, direction, provenance and exact
edge match counts at `extracted → normalized → persisted → retrieved_raw →
scope_accepted → relevance_scored → ranked → entered_final_top_k →
entered_prompt`.

An absent stage snapshot is **not** an empty snapshot. A concrete first-loss
label is emitted only when two adjacent snapshots are present and prove a
transition. In particular, `retrieved_raw = 0` cannot be attributed to
extraction or persistence. Scope rejection additionally requires the matching
retrieved edge fingerprint in a recorded rejection. Negative predicates such
as `NOT_DEPENDS_ON` require explicit evidence; absence of `DEPENDS_ON` never
creates a negative edge.

The S4.6b persisted trace starts at `retrieved_raw`, so it cannot determine
whether Q01 `HAS_ROLE`, Q24 `MONITORS`, or Q41 `NOT_DEPENDS_ON` was first lost
in extraction, normalization, or persistence. The S4.7a report therefore
labels those cases `insufficient_evidence`; this is an evidence boundary, not
a claim that the expected relationship does not exist.

## S4.7b-0: cross-request ingestion journal

The S4.6b gap was lifecycle, not proof of an extraction defect: ingestion and
QA are separate HTTP requests with separate in-memory trace collectors. The
evaluation-only `EvaluationTraceJournal` closes that gap by appending safe
`extracted`, `normalized`, and post-transaction `persisted` edge events to
`.runtime/evaluation/<run_id>/graph-ingestion-trace.jsonl`. It is enabled only
when the internal benchmark runner supplies a valid run ID plus operation ID,
the request is loopback, and `EVALUATION_TRACE_ENABLED=true`; normal uploads
and normal QA create no trace and expose no trace control in their responses.

Every event binds a validated run UUID-safe ID, operation UUID, document UUID,
version, optional safe fixture key, edge fingerprint, canonical/raw predicate,
direction, safe source basename, evidence key, status and timestamp. It never
records chunks, document text, prompts, answers, embeddings, credentials,
connection strings or caller-provided paths. Stable event IDs deduplicate
retries; cleanup preserves the journal as local ignored audit evidence.

`DocumentProcessorAdapter` records extracted evidence after extraction and
normalized evidence after dangling-endpoint filtering. Each ingestion stage
also writes a safe `stage_observed` marker, so an observed zero-edge stage is
distinguishable from a missing snapshot without inventing an edge.
`KnowledgeGraphService` records persisted evidence and its marker only after
its write transaction succeeds; a failed or rolled-back transaction writes no
persisted edge or marker. At QA time the evaluation runner read-merges only
exact allowlisted document UUID/version events into the per-question graph
trace before retrieval. This adds no provider call and no database query.
Missing snapshots remain `insufficient_evidence`; there is no public trace
read API or caller-controlled trace output path.

## S4.8a: immutable relation-gap attribution

`scripts/analyze-graph-trace-gaps.py` is a second, offline-only derived
analysis. It hashes the ingestion journal plus the four existing run artifacts
before and after processing and writes only to a new sibling output directory.
It does not initialise a Provider, API application, MongoDB, Chroma or Neo4j
client.

For the observed extraction-stage misses it distinguishes an actually absent
relation edge from a bounded raw-predicate semantic signal, predicate
canonicalisation gap, endpoint alias mismatch, and reversed endpoints. The
journal is deliberately **not** a complete entity inventory and contains no
chunk ordinal or text. Therefore `entity_not_extracted` and
`cross_chunk_separation` are reported as `insufficient_evidence` unless a
future journal schema records safe identity/ordinal evidence. An exact-string
miss alone must never be called an extraction failure.

For a persisted relation that appears absent in retrieval, the analyzer checks
the exact and conservative alias-equivalent relation at `persisted`,
`retrieved_raw`, and `entered_prompt`. Query rewrite entities, hop limits and
runtime predicate filters are not present in the current journal; claims about
query-seed, traversal or ranking failures remain `insufficient_evidence` until
those fields are safely traced. Derived reports contain only relation identity,
safe source basenames, UUIDs, bounded raw predicates and counts—never fixture
body text, prompts, answers, credentials, connection strings or paths.
