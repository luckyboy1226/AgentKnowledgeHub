"""Run-scoped frozen query embeddings for controlled retrieval evaluation."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _vector_hash(vector: Sequence[float]) -> str:
    return _hash([float(value) for value in vector])


@dataclass(frozen=True)
class FrozenQueryEmbeddingSnapshot:
    payload: dict[str, Any]

    @staticmethod
    def planned_queries(plans: Sequence[dict[str, Any]]) -> list[dict[str, str | int]]:
        """Derive exact identities only from the frozen query plans."""
        planned: list[dict[str, str | int]]=[]; seen=set()
        for plan in plans:
            question_id=str(plan.get("question_id") or ""); plan_hash=str(plan.get("plan_hash") or "")
            queries=plan.get("queries") or []
            if not question_id or not plan_hash or not isinstance(queries,list):
                raise ValueError("snapshot_plan_identity_invalid")
            for ordinal, value in enumerate(queries):
                query=str(value).strip(); key=(question_id,ordinal)
                if not query or key in seen: raise ValueError("snapshot_query_identity_invalid")
                seen.add(key)
                planned.append({"question_id":question_id,"query_ordinal":ordinal,"query_text_sha256":_text_hash(query),"plan_hash":plan_hash,"query":query})
        if not planned: raise ValueError("snapshot_plans_empty")
        return planned

    @classmethod
    def empty(cls, *, run_id: str, query_plans_hash: str, embedding: dict[str, Any]) -> "FrozenQueryEmbeddingSnapshot":
        dimension=int(embedding["dimension"])
        if dimension < 1: raise ValueError("snapshot_embedding_dimension_invalid")
        return cls({"schema_version":"query-embedding-snapshot-v2","run_id":str(run_id),
                    "query_plans_hash":str(query_plans_hash),"embedding":dict(embedding),"records":[]})

    def _validate_base(self, *, run_id: str, query_plans_hash: str, embedding: dict[str, Any]) -> None:
        root=self.payload
        if (root.get("schema_version")!="query-embedding-snapshot-v2" or root.get("run_id")!=str(run_id)
                or root.get("query_plans_hash")!=str(query_plans_hash) or root.get("embedding")!=dict(embedding)):
            raise ValueError("snapshot_root_identity_mismatch")

    @staticmethod
    def _record_hash(record: dict[str, Any]) -> str:
        return _hash({key:record[key] for key in record if key not in {"vector","snapshot_hash"}})

    def validate_partial(self, *, plans: Sequence[dict[str, Any]], run_id: str, query_plans_hash: str,
                         embedding: dict[str, Any]) -> set[tuple[str, int]]:
        """Validate exact existing records; permits only a root-hash-less partial file."""
        self._validate_base(run_id=run_id,query_plans_hash=query_plans_hash,embedding=embedding)
        expected={(str(row["question_id"]),int(row["query_ordinal"])):row for row in self.planned_queries(plans)}
        records=self.payload.get("records")
        if not isinstance(records,list): raise ValueError("snapshot_records_invalid")
        seen=set(); dimension=int(embedding["dimension"])
        for record in records:
            if not isinstance(record,dict): raise ValueError("snapshot_record_invalid")
            key=(str(record.get("question_id") or ""),record.get("query_ordinal"))
            if not isinstance(key[1],int) or key in seen or key not in expected: raise ValueError("snapshot_record_identity_mismatch")
            seen.add(key); planned=expected[key]
            vector=[float(value) for value in record.get("vector",[])]
            if (record.get("run_id")!=str(run_id) or record.get("plan_hash")!=planned["plan_hash"]
                    or record.get("query_text_sha256")!=planned["query_text_sha256"] or record.get("embedding")!=dict(embedding)
                    or record.get("vector_dimension")!=dimension or len(vector)!=dimension
                    or not all(math.isfinite(value) for value in vector) or record.get("vector_sha256")!=_vector_hash(vector)
                    or record.get("snapshot_hash")!=self._record_hash(record)):
                raise ValueError("snapshot_vector_integrity_mismatch")
        root_hash=self.payload.get("snapshot_hash")
        if root_hash is not None:
            if len(seen)!=len(expected) or root_hash!=_hash({key:self.payload[key] for key in self.payload if key!="snapshot_hash"}):
                raise ValueError("snapshot_root_hash_mismatch")
        return seen

    async def freeze_missing(self, *, plans: Sequence[dict[str, Any]], run_id: str, query_plans_hash: str,
                             embedding: dict[str, Any], provider: Any,
                             persist_partial: Callable[[dict[str, Any]], None]) -> dict[str, int]:
        existing=self.validate_partial(plans=plans,run_id=run_id,query_plans_hash=query_plans_hash,embedding=embedding)
        if self.payload.get("snapshot_hash") is not None:
            return {"planned_query_count":len(self.planned_queries(plans)),"provider_calls":0,"skipped_records":len(existing),"new_records":0}
        calls=0
        for planned in self.planned_queries(plans):
            key=(str(planned["question_id"]),int(planned["query_ordinal"]))
            if key in existing: continue
            vector=[float(value) for value in await provider.aembed_query(str(planned["query"]))]; calls+=1
            dimension=int(embedding["dimension"])
            if len(vector)!=dimension or not all(math.isfinite(value) for value in vector): raise ValueError("snapshot_vector_invalid")
            record={"run_id":str(run_id),"question_id":key[0],"query_ordinal":key[1],
                    "query_text_sha256":planned["query_text_sha256"],"plan_hash":planned["plan_hash"],
                    "embedding":dict(embedding),"vector_dimension":dimension,"vector_sha256":_vector_hash(vector),"vector":vector}
            record["snapshot_hash"]=self._record_hash(record); self.payload["records"].append(record)
            persist_partial(self.payload); existing.add(key)
        self.validate_partial(plans=plans,run_id=run_id,query_plans_hash=query_plans_hash,embedding=embedding)
        self.payload["snapshot_hash"]=_hash({key:self.payload[key] for key in self.payload})
        return {"planned_query_count":len(self.planned_queries(plans)),"provider_calls":calls,"skipped_records":len(existing)-calls,"new_records":calls}

    def vector_for(self, *, run_id: str, plan: dict[str, Any], query_plans_hash: str,
                   embedding: dict[str, Any], query_ordinal: int=0) -> tuple[float, ...]:
        self._validate_base(run_id=run_id,query_plans_hash=query_plans_hash,embedding=embedding)
        expected=self.planned_queries([plan]); self.validate_partial(plans=[plan],run_id=run_id,query_plans_hash=query_plans_hash,embedding=embedding)
        if self.payload.get("snapshot_hash") is None: raise ValueError("snapshot_incomplete")
        key=(str(plan.get("question_id") or ""),int(query_ordinal))
        expected_record=next((row for row in expected if (row["question_id"],row["query_ordinal"])==key),None)
        record=next((row for row in self.payload["records"] if (row.get("question_id"),row.get("query_ordinal"))==key),None)
        if expected_record is None or not isinstance(record,dict): raise ValueError("snapshot_question_missing")
        return tuple(float(value) for value in record["vector"])

    def vector_hash_for(self, **kwargs: Any) -> str:
        return _vector_hash(self.vector_for(**kwargs))
