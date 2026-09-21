"""Configuration boundary for optional local reranking."""

from __future__ import annotations

from typing import Any

from retrieval.local_bge_reranker import LocalBGERerankProvider
from retrieval.reranker import ConfigurableModelReranker, DisabledReranker, Reranker


def create_reranker(settings: Any) -> Reranker:
    if not bool(settings.rerank_enabled) or str(settings.rerank_provider).lower() == "disabled":
        return DisabledReranker()
    if str(settings.rerank_provider).lower() != "local_bge":
        raise ValueError("rerank_provider_invalid")
    provider = LocalBGERerankProvider(
        settings.rerank_model_path,
        device=settings.rerank_device,
        batch_size=settings.rerank_batch_size,
        max_length=settings.rerank_max_length,
    )
    return ConfigurableModelReranker(
        provider,
        timeout_seconds=settings.rerank_timeout_seconds,
        max_attempts=settings.rerank_max_attempts,
    )

