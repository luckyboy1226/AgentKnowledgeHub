from __future__ import annotations

import asyncio
import math
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

import httpx


class EmbeddingProviderError(RuntimeError):
    """A safe domain error that never exposes credentials."""


class EmbeddingProvider(Protocol):
    provider_name: str
    model_name: str
    dimensions: int

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]: ...
    async def aembed_query(self, text: str) -> list[float]: ...


def embeddings_endpoint(base_url: str) -> str:
    """Append exactly one embeddings path to an OpenAI-compatible base URL."""
    parsed = urlsplit(base_url.rstrip("/"))
    path = parsed.path.rstrip("/")
    if path.endswith("/embeddings"):
        return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
    return urlunsplit((parsed.scheme, parsed.netloc, f"{path}/embeddings", "", ""))


def validate_embeddings(vectors: object, expected_count: int, dimensions: int) -> list[list[float]]:
    if not isinstance(vectors, list) or len(vectors) != expected_count:
        raise EmbeddingProviderError("Embedding response count does not match input count.")
    result: list[list[float]] = []
    for vector in vectors:
        if not isinstance(vector, list) or len(vector) != dimensions:
            raise EmbeddingProviderError("Embedding response dimension does not match configuration.")
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in vector):
            raise EmbeddingProviderError("Embedding response contains non-finite values.")
        result.append([float(value) for value in vector])
    return result


class OpenAICompatibleEmbeddingProvider:
    """Validated OpenAI-compatible embeddings with bounded retry behavior."""

    MAX_BATCH_SIZE = 10

    def __init__(
        self,
        provider_name: str,
        api_key: str,
        base_url: str,
        model: str,
        dimensions: int,
        timeout_seconds: float = 30,
    ) -> None:
        self.provider_name = provider_name
        self.model_name = model
        self.dimensions = dimensions
        self._api_key = api_key
        self.endpoint = embeddings_endpoint(base_url)
        self.timeout_seconds = timeout_seconds

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        batches = [texts[index : index + self.MAX_BATCH_SIZE] for index in range(0, len(texts), self.MAX_BATCH_SIZE)]
        vectors: list[list[float]] = []
        for batch in batches:
            vectors.extend(await self._request(batch))
        return vectors

    async def aembed_query(self, text: str) -> list[float]:
        return (await self.aembed_documents([text]))[0]

    async def _request(self, texts: list[str]) -> list[list[float]]:
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model_name, "input": texts, "dimensions": self.dimensions}
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_seconds, connect=10)) as client:
                    response = await client.post(self.endpoint, headers=headers, json=payload)
                if response.status_code in (401, 403):
                    raise EmbeddingProviderError("Embedding provider authentication failed.")
                if response.status_code == 429 and attempt == 0:
                    await asyncio.sleep(1)
                    continue
                response.raise_for_status()
                body = response.json()
                data = body.get("data") if isinstance(body, dict) else None
                vectors = [item.get("embedding") for item in data] if isinstance(data, list) else None
                return validate_embeddings(vectors, len(texts), self.dimensions)
            except EmbeddingProviderError:
                raise
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt == 0:
                    await asyncio.sleep(1)
                    continue
                raise EmbeddingProviderError("Embedding provider network request failed.") from None
            except (httpx.HTTPError, ValueError, TypeError):
                raise EmbeddingProviderError("Embedding provider returned an invalid response.") from None
        raise EmbeddingProviderError("Embedding provider request failed.")
