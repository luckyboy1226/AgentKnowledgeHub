from __future__ import annotations

import logging

from config.settings import Settings
from .chat import OpenAICompatibleChatProvider
from .embeddings import OpenAICompatibleEmbeddingProvider

logger = logging.getLogger(__name__)

CHAT_PROVIDERS = {"qwen", "deepseek", "openai", "openai_compatible"}
EMBEDDING_PROVIDERS = {"qwen", "openai", "openai_compatible"}


def create_chat_provider(settings: Settings) -> OpenAICompatibleChatProvider:
    config = settings.chat_config
    if config.provider not in CHAT_PROVIDERS:
        raise ValueError(f"Unsupported chat provider: {config.provider}")
    if not config.api_key or not config.base_url or not config.model:
        raise ValueError("Chat provider requires an API key, base URL, and model.")
    if config.legacy:
        logger.warning("Using legacy OPENAI_* variables for chat provider configuration.")
    return OpenAICompatibleChatProvider(
        config.provider,
        config.api_key,
        config.base_url,
        config.model,
        timeout_seconds=settings.chat_timeout_seconds,
    )


def create_embedding_provider(settings: Settings) -> OpenAICompatibleEmbeddingProvider:
    config = settings.embedding_config
    if config.provider not in EMBEDDING_PROVIDERS:
        raise ValueError(f"Unsupported embedding provider: {config.provider}")
    if not config.api_key or not config.base_url or not config.model:
        raise ValueError("Embedding provider requires an API key, base URL, and model.")
    if config.dimensions <= 0:
        raise ValueError("Embedding dimensions must be positive.")
    if config.legacy:
        logger.warning("Using legacy OPENAI_* variables for embedding provider configuration.")
    return OpenAICompatibleEmbeddingProvider(
        config.provider, config.api_key, config.base_url, config.model, config.dimensions
    )
