"""Strict, deterministic entity identity keys for graph lookup only.

This is intentionally not fuzzy matching.  It accepts only Unicode form,
case, and whitespace differences; business suffixes and arbitrary substrings
remain part of the identity.
"""

from __future__ import annotations

import unicodedata


def entity_match_key(value: object) -> str:
    """Return NFKC/casefolded identity with all Unicode whitespace removed."""
    normalized = unicodedata.normalize("NFKC", str(value or "")).strip().casefold()
    return "".join(char for char in normalized if not char.isspace())


def entity_names_match(left: object, right: object) -> bool:
    """Match only equal non-empty strict identity keys."""
    left_key, right_key = entity_match_key(left), entity_match_key(right)
    return bool(left_key and right_key and left_key == right_key)
