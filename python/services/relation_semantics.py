"""Finite, auditable relationship semantics for document provenance.

The extractor may return a natural-language predicate.  This module preserves
that original value while mapping only reviewed aliases to canonical graph
predicates.  It deliberately does not infer transitive or related semantics:
for example, providing an index is not a dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


RELATION_SEMANTICS_VERSION = "relation-semantics-v1"
RELATED_TO = "RELATED_TO"


def _alias_key(value: object) -> str:
    """Normalize spacing only; never perform semantic fuzzy matching."""
    text = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text)


_ALIASES: dict[str, str] = {
    # Direction is always the source extractor's head -> tail direction.
    "provides index": "PROVIDES_INDEX",
    "provides index to": "PROVIDES_INDEX",
    "提供索引": "PROVIDES_INDEX",
    "提供检索索引": "PROVIDES_INDEX",
    "为其提供检索索引": "PROVIDES_INDEX",
    "为天枢提供检索索引": "PROVIDES_INDEX",
    "depends on": "DEPENDS_ON",
    "依赖": "DEPENDS_ON",
    "依赖于": "DEPENDS_ON",
    "not depends on": "NOT_DEPENDS_ON",
    "不依赖": "NOT_DEPENDS_ON",
    "不依赖于": "NOT_DEPENDS_ON",
    "responsible for": "RESPONSIBLE_FOR",
    "负责": "RESPONSIBLE_FOR",
    "共同负责": "CO_RESPONSIBLE_FOR",
    "co responsible for": "CO_RESPONSIBLE_FOR",
    "works at": "WORKS_AT",
    "任职于": "WORKS_AT",
    "隶属": "MEMBER_OF",
    "belongs to": "MEMBER_OF",
    "member of": "MEMBER_OF",
    "uses": "USES",
    "使用": "USES",
    "共同交付": "CO_DELIVERS",
    "co delivers": "CO_DELIVERS",
    "related to": RELATED_TO,
    "相关": RELATED_TO,
    "owns": "OWNS",
    "拥有": "OWNS",
    "developed by": "DEVELOPED_BY",
    "part of": "PART_OF",
    "located in": "LOCATED_IN",
}

# Canonical spelling is accepted as input, including persisted benchmark
# predicates.  Do not add broad identifier fallbacks here: unknown values must
# stay visibly non-specific instead of becoming silently invented semantics.
for _predicate in set(_ALIASES.values()):
    _ALIASES[_alias_key(_predicate)] = _predicate

# These predicates are part of the reviewed enterprise benchmark vocabulary.
# They are explicit vocabulary entries, not a pass-through for arbitrary Neo4j
# identifiers or model-generated relationship labels.
for _predicate in (
    "AFFECTED", "AUTHENTICATES_VIA", "CALLS", "CO_LEADS", "CURRENTLY_USES",
    "DISTINCT_FROM", "EVALUATES", "HAS_ROLE", "HEAD_OF", "MONITORS",
    "NOT_DIRECTLY_CALLS", "NOT_RESPONSIBLE_FOR", "NOT_USES", "NOT_YET_MIGRATED_TO",
    "PROVIDES_AUTH_TO", "PROVIDES_DATA_TO",
):
    _ALIASES[_alias_key(_predicate)] = _predicate

# The immutable enterprise fixture uses this historical spelling. It has the
# same directed semantics as PROVIDES_INDEX, so normalize it for scoring and
# future storage without modifying the fixture itself.
_ALIASES[_alias_key("PROVIDES_INDEX_TO")] = "PROVIDES_INDEX"


@dataclass(frozen=True)
class CanonicalRelation:
    predicate: str
    raw_predicate: str
    semantics_version: str = RELATION_SEMANTICS_VERSION


def canonicalize_relation(value: object, *, raw_predicate: object | None = None) -> CanonicalRelation:
    """Return a reviewed canonical predicate and the safely bounded raw label."""
    raw = " ".join(str(raw_predicate if raw_predicate is not None else value or "").split())
    predicate = _ALIASES.get(_alias_key(value), RELATED_TO)
    return CanonicalRelation(predicate=predicate, raw_predicate=raw[:160])


def canonical_predicate(value: object) -> str:
    """Convenience form for storage and read validation."""
    return canonicalize_relation(value).predicate


def is_canonical_predicate(value: object) -> bool:
    return str(value or "") in set(_ALIASES.values())
