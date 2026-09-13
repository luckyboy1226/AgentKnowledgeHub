import math

import pytest

from config.settings import Settings
from providers.embeddings import EmbeddingProviderError, embeddings_endpoint, validate_embeddings
from providers.factory import create_chat_provider, create_embedding_provider
from services.memory_service import MemoryService
from services.vector_store import VectorStoreService


class FakeEmbeddingProvider:
    provider_name = "fake"
    model_name = "fake-embedding"

    def __init__(self, dimensions=3):
        self.dimensions = dimensions
        self.calls = []

    async def aembed_documents(self, texts):
        self.calls.append(texts)
        return [[float(index)] * self.dimensions for index, _ in enumerate(texts)]

    async def aembed_query(self, text):
        self.calls.append([text])
        return [1.0] * self.dimensions


def explicit_settings(**overrides):
    values = dict(
        chat_provider="qwen", chat_api_key="chat-secret", chat_base_url="https://chat.example/v1", chat_model="chat-model",
        embedding_provider="qwen", embedding_api_key="embed-secret", embedding_base_url="https://embed.example/v1", embedding_model="embed-model", embedding_dimensions=3,
    )
    values.update(overrides)
    return Settings(**values)


def test_explicit_configuration_wins_over_legacy():
    settings = explicit_settings(openai_api_key="legacy")
    assert settings.chat_config.api_key == "chat-secret"
    assert settings.embedding_config.api_key == "embed-secret"


def test_legacy_configuration_remains_compatible(tmp_path, monkeypatch):
    # This contract must not inherit a developer's local explicit provider
    # settings or a preceding test's process environment.
    monkeypatch.chdir(tmp_path)
    for name in (
        "CHAT_PROVIDER", "CHAT_API_KEY", "CHAT_BASE_URL", "CHAT_MODEL",
        "EMBEDDING_PROVIDER", "EMBEDDING_API_KEY", "EMBEDDING_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = Settings(_env_file=None, openai_api_key="legacy", openai_base_url="https://legacy/v1", openai_model="legacy-model")
    assert settings.chat_config.legacy and settings.embedding_config.legacy
    assert settings.chat_config.api_key == "legacy"


def test_chat_and_embedding_can_be_separate():
    settings = explicit_settings(chat_provider="deepseek", embedding_provider="qwen")
    assert create_chat_provider(settings).provider_name == "deepseek"
    assert create_embedding_provider(settings).provider_name == "qwen"


def test_factory_does_not_infer_provider_from_url():
    settings = explicit_settings(chat_provider="deepseek", chat_base_url="https://dashscope.example/v1")
    assert create_chat_provider(settings).provider_name == "deepseek"


def test_bailian_glm_uses_bailian_thinking_switch_without_url_inference():
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    settings = explicit_settings(
        chat_provider="bailian",
        chat_base_url="https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        chat_model="glm-5.2",
        embedding_provider="qwen",
        embedding_model="text-embedding-v4",
        embedding_dimensions=1536,
    )
    from providers.chat import OpenAICompatibleChatProvider

    provider = OpenAICompatibleChatProvider(
        settings.chat_config.provider,
        settings.chat_config.api_key,
        settings.chat_config.base_url,
        settings.chat_config.model,
        client_factory=FakeClient,
    )
    assert provider.provider_name == "bailian"
    assert provider.model_name == "glm-5.2"
    assert provider.thinking_disabled is True
    assert captured["temperature"] == 0
    assert captured["extra_body"] == {"enable_thinking": False}
    embedding = create_embedding_provider(settings)
    assert (embedding.provider_name, embedding.model_name, embedding.dimensions) == ("qwen", "text-embedding-v4", 1536)


def test_embedding_endpoint_is_normalized():
    assert embeddings_endpoint("https://example/v1") == "https://example/v1/embeddings"
    assert embeddings_endpoint("https://example/v1/embeddings") == "https://example/v1/embeddings"


@pytest.mark.parametrize("vectors", [[[math.nan, 2.0]], [[1.0]]])
def test_embedding_validation_rejects_invalid_vectors(vectors):
    with pytest.raises(EmbeddingProviderError):
        validate_embeddings(vectors, 1, 2)


def test_embedding_validation_rejects_wrong_count():
    with pytest.raises(EmbeddingProviderError):
        validate_embeddings([[1.0, 2.0]], 2, 2)


def test_provider_configuration_repr_and_errors_do_not_expose_keys():
    settings = explicit_settings(chat_api_key="private-chat-key")
    assert "private-chat-key" not in repr(settings.chat_config)
    with pytest.raises(ValueError) as error:
        create_chat_provider(explicit_settings(chat_provider="unsupported", chat_api_key="chat-secret"))
    assert "chat-secret" not in str(error.value)


@pytest.mark.asyncio
async def test_health_handler_requires_no_network_or_provider():
    from api.main import health

    assert (await health())["status"] == "ok"


def test_services_reuse_injected_embedding_provider(tmp_path):
    provider = FakeEmbeddingProvider()
    assert VectorStoreService(provider).embeddings is provider
    assert MemoryService(provider, str(tmp_path / "memory.db")).embeddings is provider


@pytest.mark.asyncio
async def test_memory_does_not_mix_embedding_spaces(tmp_path):
    from services.memory_models import MemoryEvent

    database = str(tmp_path / "memory.db")
    old_provider = FakeEmbeddingProvider(dimensions=3)
    old_provider.provider_name = "qwen"
    old_provider.model_name = "text-embedding-v4"
    old_memory = MemoryService(old_provider, database)
    await old_memory.add_long_term(
        MemoryEvent(session_id="s", timestamp="2026-01-01T00:00:00", user_input="old", agent_response="old")
    )

    new_provider = FakeEmbeddingProvider(dimensions=2)
    new_provider.provider_name = "qwen"
    new_provider.model_name = "qwen3.7-text-embedding-flash"
    new_memory = MemoryService(new_provider, database)
    # Existing vectors remain in SQLite but cannot participate in a query in
    # the newly configured embedding space.
    assert await new_memory.retrieve_long_term("new") == []
