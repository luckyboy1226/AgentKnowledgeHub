from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

from langchain_openai import ChatOpenAI


@dataclass(frozen=True)
class ChatRequestOptions:
    """Per-call transport settings owned by one configured chat provider."""

    timeout_seconds: float
    # Extraction owns its bounded retry loop, so the SDK must not add hidden
    # retries for these calls.  Ordinary QA retains the provider default.
    sdk_max_retries: int = 0

    def __post_init__(self) -> None:
        if not 0 < self.timeout_seconds <= 600:
            raise ValueError("Chat request timeout must be between 0 and 600 seconds")
        if not 0 <= self.sdk_max_retries <= 3:
            raise ValueError("SDK retry count must be between 0 and 3")


class ChatProvider(Protocol):
    provider_name: str
    model_name: str

    async def ainvoke(
        self, messages: Sequence[Any], *, request_options: ChatRequestOptions | None = None, **kwargs: Any
    ) -> Any: ...


class OpenAICompatibleChatProvider:
    """A configured OpenAI-compatible chat client without request state."""

    def __init__(
        self,
        provider_name: str,
        api_key: str,
        base_url: str,
        model: str,
        *,
        timeout_seconds: float = 60,
        max_retries: int = 1,
        client_factory: Callable[..., Any] = ChatOpenAI,
    ) -> None:
        self.provider_name = provider_name
        self.model_name = model
        self._client_factory = client_factory
        self._client_args = {"model": model, "api_key": api_key, "base_url": base_url, "temperature": 0}
        self._client = self._new_client(timeout_seconds, max_retries)
        self._request_clients: dict[tuple[float, int], Any] = {}

    def _new_client(self, timeout_seconds: float, max_retries: int) -> Any:
        return self._client_factory(**self._client_args, timeout=timeout_seconds, max_retries=max_retries)

    def _client_for(self, request_options: ChatRequestOptions | None) -> Any:
        if request_options is None:
            return self._client
        key = (request_options.timeout_seconds, request_options.sdk_max_retries)
        if key not in self._request_clients:
            # This is a transport client cache inside the same Provider object,
            # not a second Provider or a separate credential lifecycle.
            self._request_clients[key] = self._new_client(*key)
        return self._request_clients[key]

    async def ainvoke(
        self, messages: Sequence[Any], *, request_options: ChatRequestOptions | None = None, **kwargs: Any
    ) -> Any:
        return await self._client_for(request_options).ainvoke(messages, **kwargs)
