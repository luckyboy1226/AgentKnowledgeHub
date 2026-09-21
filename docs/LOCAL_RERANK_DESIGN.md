# Local BGE Reranker

The optional Hybrid V2 rerank stage uses a local
`XLMRobertaForSequenceClassification` checkpoint through Transformers. It never
uses a model hub: tokenizer and model loading both set `local_files_only=True`
and `trust_remote_code=False`.

The default remains disabled. With reranking enabled, the pipeline is:

```text
Vector + BM25 + Graph -> RRF Top-30 -> local BGE Top-12
-> parent expansion -> final Top-8
```

## Configuration

Set the following only in the untracked `python/.env` file:

```env
RERANK_ENABLED=true
RERANK_PROVIDER=local_bge
RERANK_MODEL_PATH=<local-model-directory>
RERANK_DEVICE=auto
RERANK_BATCH_SIZE=4
RERANK_MAX_LENGTH=512
RERANK_INPUT_TOP_K=30
RERANK_OUTPUT_TOP_K=12
```

`auto` uses CUDA when the installed PyTorch build reports it available and
otherwise uses CPU. CUDA runs in FP16; CPU runs in FP32. Explicit `cuda` fails
closed when CUDA is unavailable. Model loading is lazy and happens at most once
per provider instance.

The model directory must contain a sequence-classification `config.json`, local
tokenizer files, and local safetensors or PyTorch weights. Paths are never
written to reports or logs.

## Failure behavior and observability

Loading, inference, timeout, OOM, malformed score, and candidate identity
failures use the existing bounded reranker error path. `ContextBuilderV2`
falls back to the deterministic RRF order. Diagnostics contain only provider
kind, device type, counts, elapsed time, and a bounded reason code; query and
candidate text are excluded.

The synthetic smoke command is local-only and is not evidence of benchmark
quality:

```powershell
conda run -n kghub python scripts/run-local-rerank-smoke.py `
  --model-path <local-model-directory> --device auto
```

Real rerank A/B remains a separately authorized R2 operation. The immutable
no-rerank baseline must not be overwritten.

R2 uses a paired protocol rather than two independent retrieval calls. See
`LOCAL_RERANK_AB_DESIGN.md`: both arms share one RRF Top-20 candidate pool,
then produce a final Top-8 after parent expansion. Benchmark failures are
explicit; only normal QA retains the availability-oriented RRF fallback.
