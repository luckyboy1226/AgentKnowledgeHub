from __future__ import annotations

from typing import Any, Protocol, Sequence

from langchain_openai import ChatOpenAI


class ChatProvider(Protocol):
    provider_name: str
    model_name: str

    async def ainvoke(self, messages: Sequence[Any], **kwargs: Any) -> Any: ...


class OpenAICompatibleChatProvider:
    """A configured OpenAI-compatible chat client without request state."""

    def __init__(self, provider_name: str, api_key: str, base_url: str, model: str) -> None:
        self.provider_name = provider_name
        self.model_name = model
        self._client = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0,
            timeout=60,
            max_retries=1,
        )

    async def ainvoke(self, messages: Sequence[Any], **kwargs: Any) -> Any:
        return await self._client.ainvoke(messages, **kwargs)
