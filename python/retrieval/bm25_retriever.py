"""Small deterministic, derived BM25 index over ready/current child catalog rows."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from retrieval.candidates import RetrievalCandidate, child_candidate_id, safe_source


_PART = re.compile(r"/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+|[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*|[\u3400-\u9fff\uf900-\ufaff]+")


class DeterministicChineseTokenizer:
    """Preserve identifiers and add CJK phrase + bigram terms without NLP IO."""

    def tokenize(self, text: str) -> list[str]:
        normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
        tokens: list[str] = []
        for part in _PART.findall(normalized):
            if any("\u3400" <= char <= "\u9fff" or "\uf900" <= char <= "\ufaff" for char in part):
                tokens.append(part)
                tokens.extend(part[index:index + 2] for index in range(max(len(part) - 1, 0)))
            else:
                tokens.append(part)
        return tokens


@dataclass(frozen=True)
class _IndexedChild:
    row: dict[str, Any]
    tokens: tuple[str, ...]


class BM25Retriever:
    """In-process derived index; catalog remains the sole source of truth."""

    def __init__(self, chunk_repository: Any, *, max_indexed_children: int = 50_000, tokenizer: DeterministicChineseTokenizer | None = None) -> None:
        if max_indexed_children < 1:
            raise ValueError("BM25 max indexed children must be positive")
        self.chunk_repository = chunk_repository
        self.max_indexed_children = int(max_indexed_children)
        self.tokenizer = tokenizer or DeterministicChineseTokenizer()
        self._rows: list[_IndexedChild] = []
        self._document_frequency: dict[str, int] = {}
        self._average_length = 0.0
        self.stale = True
        self.available = False
        self.last_error: str | None = None

    @staticmethod
    def _eligible(row: dict[str, Any]) -> bool:
        return (
            row.get("kind") == "child" and row.get("status") == "ready" and row.get("is_current") is True
            and bool(row.get("document_id")) and isinstance(row.get("document_version"), int)
            and bool(row.get("chunk_id") or row.get("child_chunk_id")) and bool(str(row.get("content") or "").strip())
        )

    def mark_stale(self) -> None:
        """A best-effort notification; durable writes never depend on this index."""
        self.stale = True

    async def rebuild(self) -> dict[str, int | bool | str | None]:
        try:
            source_rows = list(self.chunk_repository.list_current_children())
            rows = [dict(row) for row in source_rows if self._eligible(dict(row))]
            rows.sort(key=lambda row: (str(row["document_id"]), int(row["document_version"]), int(row.get("chunk_index", 0)), str(row.get("chunk_id") or row.get("child_chunk_id"))))
            if len(rows) > self.max_indexed_children:
                self._rows, self._document_frequency, self._average_length = [], {}, 0.0
                self.available, self.stale = False, True
                self.last_error = "max_indexed_children_exceeded"
                return self.status()
            indexed = [_IndexedChild(row=row, tokens=tuple(self.tokenizer.tokenize(str(row["content"])))) for row in rows]
            frequencies: Counter[str] = Counter()
            for child in indexed:
                frequencies.update(set(child.tokens))
            self._rows = indexed
            self._document_frequency = dict(frequencies)
            self._average_length = sum(len(child.tokens) for child in indexed) / len(indexed) if indexed else 0.0
            self.available, self.stale, self.last_error = True, False, None
        except Exception as exc:
            # Never let a derived-index failure roll back a durable document operation.
            self._rows, self._document_frequency, self._average_length = [], {}, 0.0
            self.available, self.stale = False, True
            self.last_error = type(exc).__name__
        return self.status()

    def status(self) -> dict[str, int | bool | str | None]:
        return {"indexed_children": len(self._rows), "available": self.available, "stale": self.stale, "last_error": self.last_error}

    async def search(self, query: str, top_k: int, *, allowed_document_ids: frozenset[str] | None = None) -> list[RetrievalCandidate]:
        if not str(query or "").strip() or int(top_k) <= 0:
            return []
        if allowed_document_ids is not None and not allowed_document_ids:
            return []
        if self.stale:
            await self.rebuild()
        if not self.available or not self._rows:
            return []
        query_terms = self.tokenizer.tokenize(query)
        if not query_terms:
            return []
        total = len(self._rows)
        k1, b = 1.5, 0.75
        scored: list[tuple[float, _IndexedChild]] = []
        for child in self._rows:
            row = child.row
            # Repeat lifecycle and Scope filtering at return time. No stale,
            # legacy, processing, or scope-external child may escape.
            if not self._eligible(row):
                continue
            if allowed_document_ids is not None and str(row["document_id"]) not in allowed_document_ids:
                continue
            frequencies = Counter(child.tokens)
            length = max(len(child.tokens), 1)
            score = 0.0
            for term in query_terms:
                count = frequencies.get(term, 0)
                if not count:
                    continue
                df = self._document_frequency.get(term, 0)
                idf = math.log(1.0 + (total - df + 0.5) / (df + 0.5))
                denominator = count + k1 * (1.0 - b + b * length / max(self._average_length, 1.0))
                score += idf * count * (k1 + 1.0) / denominator
            if score > 0:
                scored.append((score, child))
        scored.sort(key=lambda item: (-item[0], str(item[1].row["document_id"]), int(item[1].row["document_version"]), int(item[1].row.get("chunk_index", 0))))
        output: list[RetrievalCandidate] = []
        for rank, (score, child) in enumerate(scored[:int(top_k)], start=1):
            row = child.row
            metadata = dict(row.get("metadata") or {}) | {
                "parent_chunk_id": row.get("parent_chunk_id"), "section_title": row.get("section_title"),
                "page_number": row.get("page_number"), "table_id": row.get("table_id"),
                "estimated_token_count": row.get("estimated_token_count"), "status": row.get("status"),
                "is_current": row.get("is_current"),
            }
            output.append(RetrievalCandidate(
                candidate_id=child_candidate_id(row, str(row["content"])), content=str(row["content"]), source=safe_source(row.get("source")),
                document_id=str(row["document_id"]), document_version=int(row["document_version"]),
                chunk_id=str(row.get("chunk_id") or row.get("child_chunk_id")), parent_chunk_id=str(row.get("parent_chunk_id")) if row.get("parent_chunk_id") else None,
                retrieval_type="bm25", raw_score=round(float(score), 8), rank=rank, metadata=metadata,
            ))
        return output
