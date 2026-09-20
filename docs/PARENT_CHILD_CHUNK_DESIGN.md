# Parent–Child Chunk Design — Phase A

## Scope and switch

Phase A adds Parent–Child ingestion artifacts only. It does **not** alter the
online QA path, vector retrieval, graph retrieval, heuristic rerank, prompt
assembly, Graph Evidence Trace, or any benchmark fixture.

`PARENT_CHILD_CHUNK_ENABLED=false` is the default. With the default value,
the existing parser → child chunk → Chroma / extraction / Neo4j path is kept
unchanged and no Parent–Child catalog rows are created.

When enabled, parser structural blocks produce a `ChunkBundle`:

```text
Document → structured blocks → ParentChunk → ChildChunk
                                  │              ├─ Chroma embedding
                                  │              ├─ KnowledgeExtractAgent
                                  │              └─ Neo4j evidence
                                  └─ Mongo document_chunks only
```

Parents are deliberately neither embedded, passed to knowledge extraction,
nor written into Neo4j. This avoids duplicate graph evidence and preserves
the existing extraction granularity. A later phase may expand a retrieved
child to its parent under a bounded prompt budget.

## Deterministic chunks

The initial rules are deliberately simple and testable:

- Markdown is split at headings and preserves `section_title`.
- PDF page extraction retains a real `page_number` when available.
- TXT is split at paragraph boundaries before a long paragraph falls back to
  deterministic bounded windows.
- CSV/Excel parser row groups remain table blocks with `table_id`.
- Other types use the same deterministic fallback.

`UnicodeTokenEstimator` is an injected, deterministic estimate: each CJK
character and each contiguous latin/numeric word counts once. Metadata uses
`estimated_token_count`; it is not claimed to be a provider tokenizer count.

| Setting | Default | Meaning |
|---|---:|---|
| `PARENT_TARGET_TOKENS` | 1000 | preferred Parent estimate |
| `PARENT_MAX_TOKENS` | 1400 | hard Parent estimate limit |
| `CHILD_TARGET_TOKENS` | 250 | preferred Child estimate |
| `CHILD_OVERLAP_TOKENS` | 50 | approximate Child overlap |

Stable identifiers never use Python `hash()`:

- Parent: `{document_id}:v{version}:p{parent_index}`
- Child/vector: `{document_id}:v{version}:c{chunk_index}`

Every child records `parent_chunk_id`, source-safe structural metadata,
`document_id`, `document_version`, and `content_hash`.

## Catalog and lifecycle

`ChunkRepository` owns Mongo collection `document_chunks`. Each row has
`kind=parent|child`, exact document/version identity, content, safe metadata,
`status=processing|ready|failed`, and `is_current`.

Queries always include both `document_id` and `document_version`; parent IDs
alone are never sufficient. Its supported lifecycle methods are
`stage_version`, `activate_version`, `deactivate_version`, `delete_version`,
`delete_document`, `get_parent`, `list_children`, and `list_current_children`.

The Coordinator's enabled lifecycle is:

```text
prepare
→ chunk_staged
→ vector_staged
→ graph_staged
→ vector_activated
→ graph_activated
→ chunk_activated
→ Mongo document/version ready
```

The registry cannot become ready before all three storage projections are
ready. On failure, compensation is strictly version-scoped: deactivate new
current projections, delete only `document_id + version` staged catalog/vector/
graph rows, then restore an old current version only when it was affected.
It never calls `delete_document()` as replacement-version compensation.

Deletion only cleans catalog records when that version actually has catalog
rows, so V1 documents retain their existing deletion behavior.

## Chroma and compatibility

Only Child records enter Chroma. Their existing metadata remains, with optional
scalar additions: `parent_chunk_id`, `section_title`, `page_number`,
`table_id`, and `estimated_token_count`. Parent content is never stored in
Chroma.

No legacy collection data is migrated. Existing vectors without
`parent_chunk_id` remain readable through the V1 retrieval behavior. A future
Parent Expansion implementation must explicitly fall back to child context for
such legacy hits; it must not infer a parent from neighboring chunks.

## Explicit non-goals

Phase A does not add BM25, RRF, reranking, Parent Expansion, Retrieval Trace,
new LangGraph nodes, A/B evaluation variants, Kafka/Celery/Milvus, or a
frontend change. It is an ingestion capability, not a claim that retrieval has
already improved.
