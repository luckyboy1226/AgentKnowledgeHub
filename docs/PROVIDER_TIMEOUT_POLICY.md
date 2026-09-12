# Provider timeout policy

## Boundary

This policy covers document parsing and knowledge extraction only. It does not
change the existing QA timeout/retry behavior or provider credentials, model,
and base URL selection.

The configured defaults are deliberately bounded:

| Setting | Default | Bound | Purpose |
|---|---:|---:|---|
| `CHAT_TIMEOUT_SECONDS` | 60 s | 1–600 s | Ordinary Chat provider default (including QA). |
| `EXTRACTION_REQUEST_TIMEOUT_SECONDS` | 120 s | 1–300 s | One extraction model invocation. |
| `EXTRACTION_CHUNK_DEADLINE_SECONDS` | 180 s | 1–600 s | All attempts for one chunk. |
| `DOCUMENT_PROCESSING_TIMEOUT_SECONDS` | 900 s | 1–1800 s | All extraction chunks for one document. |
| `EXTRACTION_MAX_ATTEMPTS` | 2 | 1–3 | Attempts for a side-effect-free model call. |
| `EXTRACTION_RETRY_BACKOFF_SECONDS` | 1 s | 0–5 s | Bounded pause between retryable attempts. |

The chunk deadline must cover one request; the document deadline must cover one
chunk. Invalid configurations fail validation at startup. Extraction invokes
the configured Chat Provider with an explicit request option and SDK retries
disabled; the extraction agent owns the visible, bounded retry loop. This is
one configured Provider with request-scoped transport clients, not a second
Provider or credential lifecycle.

## Failure handling

Only model calls are retried. Chroma and Neo4j stage/activate methods are not
inside this retry loop. An extraction failure occurs before either store is
staged, so the coordinator marks the new Mongo version `failed`; an older
current version remains active.

Safe categories are `provider_timeout`, `provider_connection`,
`provider_rate_limit`, `provider_auth`, `provider_http`,
`provider_invalid_response`, `validation`, and `unknown`. Timeouts include a
safe kind (`connect`, `read`, `tls_read_wait`, `request_deadline`,
`chunk_deadline`, or `document_deadline`) where known. The operation journal
also records a safe chunk index and attempt counts. It never stores provider
messages, request headers, document content, prompts, keys, or connection
strings.

HTTP mapping is: provider/processing timeout → 504; connection, rate limit, or
provider auth → 503; invalid provider response or upstream HTTP failure → 502;
document validation → 400/422; unknown internal failure → 500. A client-side
HTTP timeout is distinct from a provider timeout: callers must observe the
same `operation_id` and must not repeat the POST merely because the response
was not received.

## D03 diagnostic status

The previous D03 ingestion failed during provider extraction, before vector or
graph staging. Its 121-second duration is consistent with a 60-second client
configuration plus one retry and overhead, but exact SDK retry timing remains
`insufficient_evidence` without request-level instrumentation. This is not a
GraphRAG benchmark result. A future D03-only authorized validation should send
one POST with a new `operation_id`; on client timeout it must poll the
operation and never resubmit the document.
