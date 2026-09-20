"""Tiny duck-typed bridge so retrieval components do not depend on a store."""

from __future__ import annotations

from typing import Any


def emit(trace: Any | None, method: str, /, *args: Any, **kwargs: Any) -> None:
    """Observability is strictly best-effort and never changes retrieval output."""
    if trace is None:
        return
    try:
        callback = getattr(trace, method, None)
        if callable(callback):
            callback(*args, **kwargs)
    except Exception:
        # Trace faults must not become a retrieval availability fault.
        return
