"""
知识抽取 Agent — 从文档块中提取实体、关系、事件，构建知识图谱三元组

核心能力:
  1. 命名实体识别 (NER)
  2. 关系抽取 (RE)
  3. 事件抽取
  4. 三元组生成 → 写入 Neo4j
"""

from __future__ import annotations

import json
import asyncio
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from agents.doc_parser_agent import DocumentChunk
from config.settings import ExtractionTimeoutPolicy
from providers.chat import ChatProvider, ChatRequestOptions

EXTRACTION_SYSTEM_PROMPT = """\
你是一个专业的知识抽取引擎。给定一段文本，请提取其中的：
1. **实体 (entities)**：人名、组织、地点、产品、技术、概念等
2. **关系 (relations)**：实体之间的关系，用三元组 (头实体, 关系, 尾实体) 表示
3. **事件 (events)**：文本中提到的事件，包含触发词和参与者

请严格按照以下 JSON 格式返回：
{
  "entities": [
    {"name": "实体名", "type": "实体类型", "description": "简短描述"}
  ],
  "relations": [
    {"head": "头实体", "relation": "关系类型", "raw_predicate": "原文关系短语", "tail": "尾实体", "confidence": 0.95}
  ],
  "events": [
    {"trigger": "触发词", "type": "事件类型", "participants": ["参与者1"]}
  ]
}

注意:
- 实体类型包括: Person, Organization, Location, Product, Technology, Concept, Event, Time
- 关系必须保持方向：head 是施事/起点，tail 是受事/终点。
- 使用明确关系：depends_on, provides_index, responsible_for, uses, works_at,
  member_of, co_delivers, related_to。保留 raw_predicate 为原文短语。
- “A 为 B 提供检索索引”只能写为 A -> provides_index -> B；不得改写为 depends_on。
- 仅当原文明确写出依赖时才使用 depends_on；无法确定时使用 related_to，不得猜测。
- confidence 为 0-1 之间的浮点数
- 只返回 JSON，不要包含其他文字
"""


@dataclass
class Entity:
    name: str
    type: str
    description: str = ""
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def node_label(self) -> str:
        return self.type.replace(" ", "_")


@dataclass
class Relation:
    head: str
    relation: str
    tail: str
    confidence: float = 0.0
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeEvent:
    trigger: str
    type: str
    participants: list[str] = field(default_factory=list)


@dataclass
class ExtractionResult:
    entities: list[Entity]
    relations: list[Relation]
    events: list[KnowledgeEvent]
    source_chunk_id: str = ""


class ExtractionInvocationError(RuntimeError):
    """Safe metadata boundary for one failed, side-effect-free model call."""

    safe_processing_phase = "extract"

    def __init__(
        self,
        *,
        chunk_index: int | None,
        attempt: int,
        max_attempts: int,
        timeout_kind: str | None = None,
    ) -> None:
        super().__init__("Knowledge extraction provider call failed")
        self.chunk_index = chunk_index
        self.attempt = attempt
        self.max_attempts = max_attempts
        self.timeout_kind = timeout_kind


class ExtractionDeadlineError(TimeoutError):
    """A bounded chunk or document deadline expired before persistence."""

    safe_processing_phase = "extract"

    def __init__(self, *, chunk_index: int | None, timeout_kind: str, attempt: int, max_attempts: int) -> None:
        super().__init__("Knowledge extraction deadline expired")
        self.chunk_index = chunk_index
        self.timeout_kind = timeout_kind
        self.attempt = attempt
        self.max_attempts = max_attempts


class KnowledgeExtractAgent:
    """
    知识抽取 Agent

    工作流:
      receive_chunks → extract_per_chunk → deduplicate → resolve_entities → output_triples
    """

    BATCH_SIZE = 5

    def __init__(
        self,
        chat_provider: ChatProvider,
        *,
        timeout_policy: ExtractionTimeoutPolicy | None = None,
    ) -> None:
        self.llm = chat_provider
        self.timeout_policy = timeout_policy or ExtractionTimeoutPolicy(
            request_timeout_seconds=120,
            chunk_deadline_seconds=180,
            document_deadline_seconds=900,
            max_attempts=2,
            retry_backoff_seconds=1,
        )

    # ── public API ───────────────────────────────────────────

    async def extract(self, chunks: list[DocumentChunk]) -> list[ExtractionResult]:
        """从一组文档块中抽取知识"""
        try:
            return await asyncio.wait_for(self._extract_all(chunks), self.timeout_policy.document_deadline_seconds)
        except asyncio.TimeoutError as exc:
            if isinstance(exc, ExtractionDeadlineError):
                raise
            # A document-wide deadline cannot be safely attributed to one
            # chunk: the active coroutine may have been cancelled between work.
            error = ExtractionDeadlineError(
                chunk_index=None,
                timeout_kind="document_deadline",
                attempt=0,
                max_attempts=self.timeout_policy.max_attempts,
            )
            raise error from exc

    async def _extract_all(self, chunks: list[DocumentChunk]) -> list[ExtractionResult]:
        results: list[ExtractionResult] = []
        for i in range(0, len(chunks), self.BATCH_SIZE):
            batch = chunks[i : i + self.BATCH_SIZE]
            for chunk in batch:
                result = await self._extract_chunk_with_deadline(chunk)
                results.append(result)
        merged = self._deduplicate(results)
        return merged

    async def extract_single(self, text: str, chunk_id: str = "") -> ExtractionResult:
        """从单段文本中抽取知识"""
        return await self._extract_from_text(text, chunk_id, chunk_index=None)

    # ── core extraction ──────────────────────────────────────

    async def _extract_from_chunk(self, chunk: DocumentChunk) -> ExtractionResult:
        return await self._extract_from_text(chunk.content, chunk.chunk_id, chunk_index=chunk.chunk_index)

    async def _extract_chunk_with_deadline(self, chunk: DocumentChunk) -> ExtractionResult:
        try:
            return await asyncio.wait_for(self._extract_from_chunk(chunk), self.timeout_policy.chunk_deadline_seconds)
        except asyncio.TimeoutError as exc:
            if isinstance(exc, ExtractionDeadlineError):
                raise
            error = ExtractionDeadlineError(
                chunk_index=chunk.chunk_index,
                timeout_kind="chunk_deadline",
                attempt=0,
                max_attempts=self.timeout_policy.max_attempts,
            )
            raise error from exc

    async def _extract_from_text(self, text: str, source_id: str, *, chunk_index: int | None) -> ExtractionResult:
        messages = [
            SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(content=f"请从以下文本中抽取知识：\n\n{text}"),
        ]
        resp = await self._invoke_with_policy(messages, chunk_index=chunk_index)
        return self._parse_response(resp.content, source_id)

    async def _invoke_with_policy(self, messages: list[Any], *, chunk_index: int | None) -> Any:
        """Retry only the provider call; no storage side effect exists here."""
        policy = self.timeout_policy
        last_error: BaseException | None = None
        for attempt in range(1, policy.max_attempts + 1):
            try:
                return await asyncio.wait_for(
                    self.llm.ainvoke(
                        messages,
                        request_options=ChatRequestOptions(
                            timeout_seconds=policy.request_timeout_seconds,
                            sdk_max_retries=0,
                        ),
                    ),
                    timeout=policy.request_timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                last_error = exc
                timeout_kind = "request_deadline"
            except Exception as exc:
                last_error = exc
                timeout_kind = None

            if attempt >= policy.max_attempts or not self._is_retryable(last_error):
                error = ExtractionInvocationError(
                    chunk_index=chunk_index,
                    attempt=attempt,
                    max_attempts=policy.max_attempts,
                    timeout_kind=timeout_kind,
                )
                raise error from last_error
            # Bounded sleep is still inside the caller's chunk/document
            # deadlines, so cancellation cannot lead to a later storage write.
            if policy.retry_backoff_seconds:
                await asyncio.sleep(policy.retry_backoff_seconds)
        raise AssertionError("bounded extraction retry loop unexpectedly completed")

    @staticmethod
    def _is_retryable(error: BaseException | None) -> bool:
        if isinstance(error, (asyncio.TimeoutError, TimeoutError)):
            return True
        try:
            import httpx
            from openai import APIConnectionError, APITimeoutError, RateLimitError
        except ImportError:  # pragma: no cover - dependencies are installed in production
            return False
        return isinstance(error, (httpx.TimeoutException, httpx.NetworkError, APITimeoutError, APIConnectionError, RateLimitError))

    def _parse_response(self, raw: str, source_id: str) -> ExtractionResult:
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                cleaned = cleaned.rsplit("```", 1)[0]
            data = json.loads(cleaned)
        except (json.JSONDecodeError, IndexError):
            return ExtractionResult(entities=[], relations=[], events=[], source_chunk_id=source_id)

        entities = [
            Entity(
                name=e.get("name", ""),
                type=e.get("type", "Concept"),
                description=e.get("description", ""),
            )
            for e in data.get("entities", [])
            if e.get("name")
        ]
        relations = [
            Relation(
                head=r.get("head", ""),
                relation=r.get("relation", "related_to"),
                tail=r.get("tail", ""),
                confidence=float(r.get("confidence", 0.5)),
                properties={"raw_predicate": str(r.get("raw_predicate") or r.get("relation") or "")},
            )
            for r in data.get("relations", [])
            if r.get("head") and r.get("tail")
        ]
        events = [
            KnowledgeEvent(
                trigger=ev.get("trigger", ""),
                type=ev.get("type", ""),
                participants=ev.get("participants", []),
            )
            for ev in data.get("events", [])
        ]
        return ExtractionResult(
            entities=entities,
            relations=relations,
            events=events,
            source_chunk_id=source_id,
        )

    # ── deduplication & entity resolution ────────────────────

    @staticmethod
    def _deduplicate(results: list[ExtractionResult]) -> list[ExtractionResult]:
        """
        跨 chunk 去重: 同名同类型实体合并，关系去重
        """
        seen_entities: dict[str, Entity] = {}
        seen_relations: set[tuple[str, str, str]] = set()
        deduped: list[ExtractionResult] = []

        for result in results:
            unique_entities: list[Entity] = []
            for ent in result.entities:
                key = f"{ent.name}::{ent.type}"
                if key not in seen_entities:
                    seen_entities[key] = ent
                    unique_entities.append(ent)

            unique_relations: list[Relation] = []
            for rel in result.relations:
                key = (rel.head, rel.relation, rel.tail)
                if key not in seen_relations:
                    seen_relations.add(key)
                    unique_relations.append(rel)

            deduped.append(ExtractionResult(
                entities=unique_entities,
                relations=unique_relations,
                events=result.events,
                source_chunk_id=result.source_chunk_id,
            ))
        return deduped
