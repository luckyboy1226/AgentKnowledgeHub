"""Neo4j graph service with legacy reads and document-version provenance."""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from agents.knowledge_extract_agent import Entity, Relation
from config import settings


RELATION_TYPE_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


class KnowledgeGraphService:
    """Neo4j graph service. New provenance methods are not yet wired to ingest."""

    def __init__(self) -> None:
        self._driver: Any = None

    # ── lifecycle ────────────────────────────────────────────

    async def init(self) -> None:
        from neo4j import AsyncGraphDatabase

        self._driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        await self._ensure_indexes()

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()

    async def _ensure_indexes(self) -> None:
        """Create additive indexes/constraints only; never drop existing schema."""
        index_queries = [
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.type)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.source)",
            "CREATE CONSTRAINT document_version_key IF NOT EXISTS FOR (dv:DocumentVersion) REQUIRE dv.key IS UNIQUE",
            "CREATE INDEX document_version_document IF NOT EXISTS FOR (dv:DocumentVersion) ON (dv.document_id, dv.document_version)",
        ]
        async with self._driver.session() as session:
            for query in index_queries:
                await session.run(query)

    # ── safe identifiers and transaction plumbing ────────────

    @staticmethod
    def document_version_key(document_id: str, version: int) -> str:
        return f"{document_id}:v{int(version)}"

    @staticmethod
    def evidence_key(
        document_id: str, version: int, subject: str, predicate: str, object_: str
    ) -> str:
        """Stable evidence identity for an asserted document-version fact."""
        material = "\x1f".join(
            (str(document_id), str(int(version)), str(subject), str(predicate), str(object_))
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    @staticmethod
    def safe_relationship_type(predicate: str) -> str:
        """Allow only a bounded Cypher relationship identifier, never user text."""
        candidate = re.sub(r"\s+", "_", str(predicate).strip().upper())
        return candidate if RELATION_TYPE_RE.fullmatch(candidate) else "RELATED_TO"

    @staticmethod
    def safe_source(source: str | Path) -> str:
        return str(source or "").replace("\\", "/").rsplit("/", 1)[-1]

    async def _execute_write(
        self, operation: Callable[..., Awaitable[Any]], *args: Any
    ) -> Any:
        if self._driver is None:
            raise RuntimeError("Knowledge graph is not initialized")
        async with self._driver.session() as session:
            execute_write = getattr(session, "execute_write", None)
            if execute_write is not None:
                return await execute_write(operation, *args)
            transaction = await session.begin_transaction()
            try:
                result = await operation(transaction, *args)
                await transaction.commit()
                return result
            except Exception:
                await transaction.rollback()
                raise

    # ── legacy entity/relationship operations ────────────────

    async def upsert_entity(self, entity: Entity, version: int = 1, source: str = "") -> None:
        cypher = """
        MERGE (e:Entity {name: $name})
        ON CREATE SET e.type = $type, e.description = $description,
          e.version = $version, e.source = $source, e.created_at = $now, e.updated_at = $now
        ON MATCH SET e.description = CASE WHEN $description <> '' THEN $description ELSE e.description END,
          e.version = $version, e.updated_at = $now
        """
        async with self._driver.session() as session:
            await session.run(
                cypher,
                {
                    "name": entity.name,
                    "type": entity.type,
                    "description": entity.description,
                    "version": version,
                    "source": self.safe_source(source),
                    "now": int(time.time()),
                },
            )

    async def add_relation(self, relation: Relation, source: str = "") -> None:
        """Create a legacy business relationship with a validated type identifier."""
        relationship_type = self.safe_relationship_type(relation.relation)
        cypher = f"""
        MATCH (h:Entity {{name: $head}})
        MATCH (t:Entity {{name: $tail}})
        MERGE (h)-[r:{relationship_type}]->(t)
        SET r.confidence = $confidence, r.source = $source, r.updated_at = $now
        """
        async with self._driver.session() as session:
            await session.run(
                cypher,
                {
                    "head": relation.head,
                    "tail": relation.tail,
                    "confidence": relation.confidence,
                    "source": self.safe_source(source),
                    "now": int(time.time()),
                },
            )

    # ── document-version provenance lifecycle ────────────────

    async def stage_document_version(
        self,
        document_id: str,
        version: int,
        content_hash: str,
        source: str | Path,
        entities: list[Entity],
        relations: list[Relation],
    ) -> dict[str, int]:
        """Atomically stage one processing version with entity mentions and evidence."""
        return await self._execute_write(
            self._stage_document_version_tx,
            str(document_id),
            int(version),
            str(content_hash),
            self.safe_source(source),
            entities,
            relations,
        )

    async def _stage_document_version_tx(
        self,
        tx: Any,
        document_id: str,
        version: int,
        content_hash: str,
        source: str,
        entities: list[Entity],
        relations: list[Relation],
    ) -> dict[str, int]:
        now = int(time.time())
        key = self.document_version_key(document_id, version)
        await tx.run(
            """
            MERGE (dv:DocumentVersion {key: $key})
            ON CREATE SET dv.document_id = $document_id, dv.document_version = $version,
              dv.content_hash = $content_hash, dv.source = $source, dv.status = 'processing',
              dv.is_current = false, dv.created_at = $now, dv.updated_at = $now
            ON MATCH SET dv.updated_at = $now
            """,
            key=key,
            document_id=document_id,
            version=version,
            content_hash=content_hash,
            source=source,
            now=now,
        )

        mentioned: set[str] = set()
        for entity in entities:
            await self._merge_entity_and_mention(tx, key, entity.name, entity.type, entity.description, now)
            mentioned.add(entity.name)

        for relation in relations:
            # Relation endpoints can be present even when extraction omitted an explicit entity.
            await self._merge_entity_and_mention(tx, key, relation.head, "Unknown", "", now)
            await self._merge_entity_and_mention(tx, key, relation.tail, "Unknown", "", now)
            mentioned.update((relation.head, relation.tail))
            predicate = str(relation.relation)
            relationship_type = self.safe_relationship_type(predicate)
            evidence_key = self.evidence_key(
                document_id, version, relation.head, predicate, relation.tail
            )
            await tx.run(
                f"""
                MATCH (h:Entity {{name: $head}})
                MATCH (t:Entity {{name: $tail}})
                MERGE (h)-[r:{relationship_type} {{evidence_key: $evidence_key}}]->(t)
                ON CREATE SET r.document_id = $document_id, r.document_version = $version,
                  r.predicate = $predicate, r.confidence = $confidence, r.source = $source,
                  r.status = 'processing', r.is_current = false, r.created_at = $now, r.updated_at = $now
                ON MATCH SET r.updated_at = $now
                """,
                head=relation.head,
                tail=relation.tail,
                evidence_key=evidence_key,
                document_id=document_id,
                version=version,
                predicate=predicate,
                confidence=relation.confidence,
                source=source,
                now=now,
            )
        return {"mentions": len(mentioned), "evidence": len(relations)}

    @staticmethod
    async def _merge_entity_and_mention(
        tx: Any, version_key: str, name: str, entity_type: str, description: str, now: int
    ) -> None:
        await tx.run(
            """
            MATCH (dv:DocumentVersion {key: $key})
            MERGE (e:Entity {name: $name})
            ON CREATE SET e.type = $type, e.description = $description,
              e.created_at = $now, e.updated_at = $now
            ON MATCH SET e.description = CASE WHEN $description <> '' THEN $description ELSE e.description END,
              e.updated_at = $now
            MERGE (dv)-[:MENTIONS]->(e)
            """,
            key=version_key,
            name=name,
            type=entity_type,
            description=description,
            now=now,
        )

    async def activate_document_version(self, document_id: str, version: int) -> None:
        """Atomically switch one document's current provenance to a staged version."""
        await self._execute_write(self._activate_document_version_tx, str(document_id), int(version))

    async def _activate_document_version_tx(self, tx: Any, document_id: str, version: int) -> None:
        now = int(time.time())
        key = self.document_version_key(document_id, version)
        await tx.run(
            """
            MATCH (target:DocumentVersion {key: $key, document_id: $document_id, document_version: $version})
            OPTIONAL MATCH (old:DocumentVersion {document_id: $document_id})
            WHERE old.key <> target.key AND old.is_current = true
            SET old.is_current = false, old.updated_at = $now
            SET target.is_current = true, target.status = 'ready', target.updated_at = $now
            """,
            key=key,
            document_id=document_id,
            version=version,
            now=now,
        )
        await tx.run(
            """
            MATCH ()-[old]->()
            WHERE old.document_id = $document_id AND old.document_version <> $version
              AND old.is_current = true
            SET old.is_current = false, old.updated_at = $now
            """,
            document_id=document_id,
            version=version,
            now=now,
        )
        await tx.run(
            """
            MATCH ()-[current]->()
            WHERE current.document_id = $document_id AND current.document_version = $version
            SET current.is_current = true, current.status = 'ready', current.updated_at = $now
            """,
            document_id=document_id,
            version=version,
            now=now,
        )

    async def deactivate_document_version(self, document_id: str, version: int) -> None:
        await self._execute_write(self._deactivate_document_version_tx, str(document_id), int(version))

    async def _deactivate_document_version_tx(self, tx: Any, document_id: str, version: int) -> None:
        now = int(time.time())
        await tx.run(
            """
            MATCH (dv:DocumentVersion {document_id: $document_id, document_version: $version})
            SET dv.is_current = false, dv.updated_at = $now
            """,
            document_id=document_id,
            version=version,
            now=now,
        )
        await tx.run(
            """
            MATCH ()-[r]->()
            WHERE r.document_id = $document_id AND r.document_version = $version
            SET r.is_current = false, r.updated_at = $now
            """,
            document_id=document_id,
            version=version,
            now=now,
        )

    async def delete_document_version(self, document_id: str, version: int) -> int:
        return await self._execute_write(self._delete_document_version_tx, str(document_id), int(version))

    async def _delete_document_version_tx(self, tx: Any, document_id: str, version: int) -> int:
        records = await (await tx.run(
            """
            MATCH (:DocumentVersion {document_id: $document_id, document_version: $version})-[:MENTIONS]->(e:Entity)
            RETURN collect(DISTINCT e.name) AS entity_names
            """,
            document_id=document_id,
            version=version,
        )).data()
        entity_names = records[0].get("entity_names", []) if records else []
        await tx.run(
            """
            MATCH ()-[r]->()
            WHERE r.document_id = $document_id AND r.document_version = $version
            DELETE r
            """,
            document_id=document_id,
            version=version,
        )
        await tx.run(
            """
            MATCH (dv:DocumentVersion {document_id: $document_id, document_version: $version})-[m:MENTIONS]->()
            DELETE m
            WITH dv DELETE dv
            """,
            document_id=document_id,
            version=version,
        )
        await tx.run(
            """
            MATCH (e:Entity) WHERE e.name IN $entity_names
              AND NOT EXISTS { MATCH (:DocumentVersion)-[:MENTIONS]->(e) }
              AND NOT (e)--()
            DELETE e
            """,
            entity_names=entity_names,
        )
        return len(entity_names)

    async def delete_document(self, document_id: str) -> int:
        return await self._execute_write(self._delete_document_tx, str(document_id))

    async def _delete_document_tx(self, tx: Any, document_id: str) -> int:
        records = await (await tx.run(
            """
            MATCH (:DocumentVersion {document_id: $document_id})-[:MENTIONS]->(e:Entity)
            RETURN collect(DISTINCT e.name) AS entity_names
            """,
            document_id=document_id,
        )).data()
        entity_names = records[0].get("entity_names", []) if records else []
        await tx.run(
            "MATCH ()-[r]->() WHERE r.document_id = $document_id DELETE r",
            document_id=document_id,
        )
        await tx.run(
            """
            MATCH (dv:DocumentVersion {document_id: $document_id})-[m:MENTIONS]->()
            DELETE m
            WITH dv DELETE dv
            """,
            document_id=document_id,
        )
        await tx.run(
            """
            MATCH (e:Entity) WHERE e.name IN $entity_names
              AND NOT EXISTS { MATCH (:DocumentVersion)-[:MENTIONS]->(e) }
              AND NOT (e)--()
            DELETE e
            """,
            entity_names=entity_names,
        )
        return len(entity_names)

    async def count_document_evidence(self, document_id: str, version: int) -> int:
        records = await self.execute_cypher(
            """
            MATCH ()-[r]->()
            WHERE r.document_id = $document_id AND r.document_version = $version
            RETURN count(r) AS cnt
            """,
            {"document_id": str(document_id), "version": int(version)},
        )
        return int(records[0]["cnt"]) if records else 0

    async def list_document_evidence(self, document_id: str, version: int) -> list[dict]:
        return await self.execute_cypher(
            """
            MATCH (head:Entity)-[r]->(tail:Entity)
            WHERE r.document_id = $document_id AND r.document_version = $version
            RETURN r.evidence_key AS evidence_key, head.name AS head, type(r) AS predicate,
              tail.name AS tail, r.source AS source, r.status AS status, r.is_current AS is_current
            ORDER BY r.evidence_key
            """,
            {"document_id": str(document_id), "version": int(version)},
        )

    # ── query operations ─────────────────────────────────────

    async def execute_cypher(self, cypher: str, params: dict | None = None) -> list[dict]:
        async with self._driver.session() as session:
            result = await session.run(cypher, params or {})
            return await result.data()

    async def get_entity(self, name: str) -> dict | None:
        records = await self.execute_cypher("MATCH (e:Entity {name: $name}) RETURN e", {"name": name})
        return records[0] if records else None

    @staticmethod
    def _current_or_legacy_relationship_filter(variable: str = "r") -> str:
        return (
            f"type({variable}) <> 'MENTIONS' AND "
            f"({variable}.document_id IS NULL OR "
            f"({variable}.is_current = true AND {variable}.status = 'ready'))"
        )

    @staticmethod
    def _in_evaluation_scope(record: dict[str, Any], allowed_document_ids: frozenset[str]) -> bool:
        """Scoped graph reads require complete per-edge provenance, never legacy facts."""
        edges = record.get("evidence_edges")
        if not isinstance(edges, list) or not edges:
            return False
        for edge in edges:
            if not isinstance(edge, dict):
                return False
            if not all(isinstance(edge.get(field), str) and edge[field].strip() for field in (
                "subject", "predicate", "object", "direction", "document_id", "source", "evidence_key",
            )):
                return False
            if edge.get("direction") not in {"forward", "reverse"}:
                return False
            if str(edge.get("document_id")) not in allowed_document_ids:
                return False
            version = edge.get("document_version")
            if not isinstance(version, int) or isinstance(version, bool) or version < 1:
                return False
        return True

    async def get_neighbors(
        self,
        entity_name: str,
        hops: int = 2,
        *,
        allowed_document_ids: frozenset[str] | None = None,
        scope_diagnostics: dict[str, int] | None = None,
    ) -> list[dict]:
        """Retrieve current/legacy facts, or only exact provenance for scoped evaluation."""
        safe_hops = min(max(int(hops), 1), 5)
        if allowed_document_ids is None:
            relationship_filter = self._current_or_legacy_relationship_filter("rel")
            parameters: dict[str, Any] = {"name": entity_name, "limit": 50}
        else:
            # This parameterized condition applies to every hop: empty and
            # legacy provenance cannot satisfy it and there is no full-graph fallback.
            relationship_filter = (
                "type(rel) <> 'MENTIONS' AND rel.document_id IN $allowed_document_ids AND "
                "rel.is_current = true AND rel.status = 'ready'"
            )
            parameters = {
                "name": entity_name,
                "limit": 50,
                "allowed_document_ids": sorted(allowed_document_ids),
            }
        if allowed_document_ids is None:
            cypher = f"""
            MATCH path = (start:Entity {{name: $name}})-[rels*1..{safe_hops}]-(neighbor:Entity)
            WHERE ALL(rel IN rels WHERE {relationship_filter})
            RETURN start.name AS source,
              [rel IN rels | type(rel)] AS relations,
              neighbor.name AS target, neighbor.type AS target_type, neighbor.description AS target_desc,
              head([rel IN rels WHERE rel.document_id IS NOT NULL | rel.document_id]) AS document_id,
              head([rel IN rels WHERE rel.document_id IS NOT NULL | rel.document_version]) AS document_version,
              head([rel IN rels WHERE rel.document_id IS NOT NULL | rel.source]) AS provenance_source
            LIMIT $limit
            """
        else:
            # The traversal is undirected for recall, but each returned edge
            # preserves its stored head -> tail direction and says whether the
            # path traversed it forward or reverse.  This data shape is only
            # used by the internal, scoped evaluation path.
            cypher = f"""
            MATCH path = (start:Entity {{name: $name}})-[rels*1..{safe_hops}]-(neighbor:Entity)
            WITH path, nodes(path) AS path_nodes, relationships(path) AS rels
            WHERE ALL(rel IN rels WHERE {relationship_filter})
            RETURN path_nodes[0].name AS source,
              path_nodes[size(path_nodes) - 1].name AS target,
              [index IN range(0, size(rels) - 1) |
                CASE WHEN startNode(rels[index]) = path_nodes[index]
                  THEN {{
                    subject: path_nodes[index].name,
                    predicate: coalesce(rels[index].predicate, type(rels[index])),
                    object: path_nodes[index + 1].name,
                    direction: 'forward',
                    document_id: rels[index].document_id,
                    document_version: rels[index].document_version,
                    source: rels[index].source,
                    evidence_key: rels[index].evidence_key
                  }}
                  ELSE {{
                    subject: path_nodes[index + 1].name,
                    predicate: coalesce(rels[index].predicate, type(rels[index])),
                    object: path_nodes[index].name,
                    direction: 'reverse',
                    document_id: rels[index].document_id,
                    document_version: rels[index].document_version,
                    source: rels[index].source,
                    evidence_key: rels[index].evidence_key
                  }}
                END
              ] AS evidence_edges
            LIMIT $limit
            """
        records = await self.execute_cypher(cypher, parameters)
        if allowed_document_ids is None:
            return records
        scoped_records: list[dict] = []
        for record in records:
            if not self._in_evaluation_scope(record, allowed_document_ids):
                if scope_diagnostics is not None:
                    scope_diagnostics["graph_scope_rejected_count"] = (
                        scope_diagnostics.get("graph_scope_rejected_count", 0) + 1
                    )
                continue
            scoped_records.append(record)
        return scoped_records

    async def get_current_paths(self, name_a: str, name_b: str, limit: int = 3) -> list[dict]:
        relationship_filter = self._current_or_legacy_relationship_filter("rel")
        cypher = f"""
        MATCH path = (a:Entity {{name: $name_a}})-[rels*1..5]-(b:Entity {{name: $name_b}})
        WHERE ALL(rel IN rels WHERE {relationship_filter})
        WITH path, rels ORDER BY length(path) ASC LIMIT $limit
        RETURN [n IN nodes(path) | n.name] AS node_names,
          [rel IN rels | type(rel)] AS rel_types,
          head([rel IN rels WHERE rel.document_id IS NOT NULL | rel.document_id]) AS document_id,
          head([rel IN rels WHERE rel.document_id IS NOT NULL | rel.document_version]) AS document_version,
          head([rel IN rels WHERE rel.document_id IS NOT NULL | rel.source]) AS provenance_source
        """
        return await self.execute_cypher(
            cypher, {"name_a": name_a, "name_b": name_b, "limit": min(max(int(limit), 1), 10)}
        )

    async def search_entities(self, keyword: str, limit: int = 20) -> list[dict]:
        cypher = """
        MATCH (e:Entity)
        WHERE e.name CONTAINS $keyword OR e.description CONTAINS $keyword
        RETURN e.name AS name, e.type AS type, e.description AS description
        LIMIT $limit
        """
        return await self.execute_cypher(cypher, {"keyword": keyword, "limit": limit})

    # ── safe legacy delete and stats ──────────────────────────

    async def delete_by_source(self, source: str) -> int:
        """Maintenance-only cleanup for pre-S3 source-only entities.

        Formal document deletion is handled exclusively by
        ``DocumentUpdateCoordinator`` through exact provenance deletion. This
        legacy helper must not be used by an API route or CDC adapter.
        """
        records = await self.execute_cypher(
            """
            MATCH (e:Entity {source: $source})
            WHERE NOT EXISTS { MATCH (:DocumentVersion)-[:MENTIONS]->(e) } AND NOT (e)--()
            DELETE e
            RETURN count(e) AS deleted
            """,
            {"source": self.safe_source(source)},
        )
        return records[0].get("deleted", 0) if records else 0

    async def get_stats(self) -> dict:
        entity_count = await self.execute_cypher("MATCH (e:Entity) RETURN count(e) AS cnt")
        rel_count = await self.execute_cypher("MATCH ()-[r]->() RETURN count(r) AS cnt")
        return {
            "total_entities": entity_count[0]["cnt"] if entity_count else 0,
            "total_relations": rel_count[0]["cnt"] if rel_count else 0,
        }
