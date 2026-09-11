"""Retired local-path update entry point.

Document creation, update, and deletion are now coordinated exclusively by
``DocumentUpdateCoordinator``. This module remains importable only so an old
integration fails explicitly instead of silently falling back to the unsafe
delete-and-rebuild implementation that preceded S3.
"""

from __future__ import annotations

from typing import Any, NoReturn


class DeprecatedUpdatePathError(RuntimeError):
    """Raised when a caller attempts the retired local-path update workflow."""


_MESSAGE = (
    "KnowledgeUpdateAgent is retired. Use DocumentUpdateCoordinator through "
    "the versioned /api/documents multipart API; local paths and direct "
    "VectorStore/KnowledgeGraph writes are not supported."
)


class KnowledgeUpdateAgent:
    """Compatibility shell that rejects the retired direct-storage workflow.

    A future CDC adapter may validate an event and invoke
    ``DocumentUpdateCoordinator`` with a stable logical key and controlled
    content/object reference. It must not revive this class as a second
    Mongo/Chroma/Neo4j write path.
    """

    def __init__(self, *_: Any, **__: Any) -> None:
        """Accept legacy construction only to provide a deterministic error later."""

    @staticmethod
    def _retired() -> NoReturn:
        raise DeprecatedUpdatePathError(_MESSAGE)

    async def process_change(self, *_: Any, **__: Any) -> NoReturn:
        self._retired()

    async def process_batch(self, *_: Any, **__: Any) -> NoReturn:
        self._retired()

    def detect_changes(self, *_: Any, **__: Any) -> NoReturn:
        self._retired()

    def start_watching(self, *_: Any, **__: Any) -> NoReturn:
        self._retired()

    async def start_kafka_consumer(self, *_: Any, **__: Any) -> NoReturn:
        self._retired()
