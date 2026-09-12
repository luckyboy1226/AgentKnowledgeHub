"""
应用配置 — 通过环境变量或 .env 文件加载
"""

from dataclasses import dataclass

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings


@dataclass(frozen=True, repr=False)
class ProviderConfig:
    provider: str
    api_key: str
    base_url: str
    model: str
    dimensions: int | None = None
    legacy: bool = False


@dataclass(frozen=True)
class ExtractionTimeoutPolicy:
    """Bounded extraction-only request and deadline policy.

    The document pipeline owns retries around the model call.  This policy is
    intentionally separate from ordinary chat so QA behavior remains stable.
    """

    request_timeout_seconds: float
    chunk_deadline_seconds: float
    document_deadline_seconds: float
    max_attempts: int
    retry_backoff_seconds: float


class Settings(BaseSettings):
    # LLM
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Provider request policy.  The ordinary chat value preserves the current
    # QA default; extraction has its own bounded policy below.
    chat_timeout_seconds: float = Field(default=60, gt=0, le=600)
    extraction_request_timeout_seconds: float = Field(default=120, gt=0, le=300)
    extraction_chunk_deadline_seconds: float = Field(default=180, gt=0, le=600)
    document_processing_timeout_seconds: float = Field(default=900, gt=0, le=1800)
    extraction_max_attempts: int = Field(default=2, ge=1, le=3)
    extraction_retry_backoff_seconds: float = Field(default=1, ge=0, le=5)

    # Explicit provider configuration. Empty values deliberately fall back to
    # the legacy variables above, keeping existing local .env files functional.
    chat_provider: str = ""
    chat_api_key: str = ""
    chat_base_url: str = ""
    chat_model: str = ""
    embedding_provider: str = ""
    embedding_api_key: str = ""
    embedding_base_url: str = ""

    # Memory System (v6.0)
    memory_db_path: str = "./memory.db"
    short_term_window: int = 10
    long_term_top_k: int = 5
    reflection_interval: int = 10

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # Vector Store
    vector_store_type: str = "chroma"  # chroma | pgvector
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    pgvector_dsn: str = "postgresql://postgres:postgres@localhost:5432/knowledge"

    # Kafka (CDC)
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_doc_changes: str = "doc-changes"
    kafka_topic_kg_updates: str = "kg-updates"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8080

    # MongoDB (LangGraph Checkpoint)
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "agenthub"

    # Document Store
    upload_dir: str = "./uploads"

    @property
    def has_usable_llm_key(self) -> bool:
        """Return false for intentionally local placeholder credentials."""
        key = self.chat_config.api_key.strip()
        placeholders = ("placeholder", "your-api-key", "not-a-real-key")
        return bool(key) and not any(token in key.lower() for token in placeholders)

    @property
    def chat_config(self) -> ProviderConfig:
        explicit = any((self.chat_provider, self.chat_api_key, self.chat_base_url, self.chat_model))
        return ProviderConfig(
            provider=(self.chat_provider or "openai_compatible").lower(),
            api_key=self.chat_api_key or self.openai_api_key,
            base_url=self.chat_base_url or self.openai_base_url,
            model=self.chat_model or self.openai_model,
            legacy=not explicit,
        )

    @property
    def embedding_config(self) -> ProviderConfig:
        explicit = any((self.embedding_provider, self.embedding_api_key, self.embedding_base_url))
        return ProviderConfig(
            provider=(self.embedding_provider or "openai_compatible").lower(),
            api_key=self.embedding_api_key or self.openai_api_key,
            base_url=self.embedding_base_url or self.openai_base_url,
            model=self.embedding_model,
            dimensions=self.embedding_dimensions,
            legacy=not explicit,
        )

    @property
    def extraction_timeout_policy(self) -> ExtractionTimeoutPolicy:
        return ExtractionTimeoutPolicy(
            request_timeout_seconds=self.extraction_request_timeout_seconds,
            chunk_deadline_seconds=self.extraction_chunk_deadline_seconds,
            document_deadline_seconds=self.document_processing_timeout_seconds,
            max_attempts=self.extraction_max_attempts,
            retry_backoff_seconds=self.extraction_retry_backoff_seconds,
        )

    @model_validator(mode="after")
    def _validate_timeout_hierarchy(self) -> "Settings":
        if self.extraction_chunk_deadline_seconds < self.extraction_request_timeout_seconds:
            raise ValueError("EXTRACTION_CHUNK_DEADLINE_SECONDS must cover one extraction request")
        if self.document_processing_timeout_seconds < self.extraction_chunk_deadline_seconds:
            raise ValueError("DOCUMENT_PROCESSING_TIMEOUT_SECONDS must cover one chunk deadline")
        return self

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
