"""Internal, fail-closed document scope for S4 retrieval evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import UUID


_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")


class EvaluationScopeError(ValueError):
    """Raised before evaluation retrieval when its document boundary is unsafe."""


def _canonical_document_id(value: object) -> str:
    try:
        return str(UUID(str(value)))
    except (TypeError, ValueError, AttributeError) as exc:
        raise EvaluationScopeError("Evaluation scope requires uploaded UUID document IDs") from exc


@dataclass(frozen=True)
class EvaluationScope:
    """A verified, immutable allowlist supplied only by the internal evaluator.

    A scope is intentionally not part of public request schemas.  The caller
    must create it from the exact IDs returned by the S3 upload operation.
    """

    run_id: str
    allowed_document_ids: frozenset[str]
    scope_verified: bool = True

    @classmethod
    def from_uploaded_document_ids(
        cls, run_id: str, document_ids: object
    ) -> "EvaluationScope":
        if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
            raise EvaluationScopeError("Evaluation scope run_id is invalid")
        if isinstance(document_ids, (str, bytes)):
            raise EvaluationScopeError("Evaluation scope requires a non-empty document ID collection")
        try:
            normalized = frozenset(_canonical_document_id(item) for item in document_ids)
        except TypeError as exc:
            raise EvaluationScopeError("Evaluation scope requires a document ID collection") from exc
        if not normalized:
            raise EvaluationScopeError("Evaluation scope cannot be empty")
        return cls(run_id=run_id, allowed_document_ids=normalized, scope_verified=True)

    def is_verified(self) -> bool:
        if not self.scope_verified or not isinstance(self.run_id, str) or not _RUN_ID_RE.fullmatch(self.run_id):
            return False
        if not self.allowed_document_ids:
            return False
        try:
            return all(_canonical_document_id(item) == item for item in self.allowed_document_ids)
        except EvaluationScopeError:
            return False


def require_verified_scope(scope: EvaluationScope | None) -> EvaluationScope:
    """Fail closed rather than allowing a scoped run to fall back to all data."""
    if not isinstance(scope, EvaluationScope) or not scope.is_verified():
        raise EvaluationScopeError("A verified non-empty evaluation scope is required")
    return scope
