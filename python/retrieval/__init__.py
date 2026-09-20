"""Feature-gated retrieval V2 building blocks.

They are intentionally not wired into ordinary QA until a later phase.
"""
"""Internal Retrieval V2 building blocks; never the default QA path."""

from retrieval.fusion import FusedCandidate, RRFFusion, RRFFusionDiagnostics, RRFFusionResult
from retrieval.context_builder import ContextBuilderResult, ContextBuilderV2
from retrieval.parent_expander import FinalContext, ParentExpander
from retrieval.reranker import RerankedCandidate

__all__ = [
    "ContextBuilderResult", "ContextBuilderV2", "FinalContext", "FusedCandidate",
    "ParentExpander", "RRFFusion", "RRFFusionDiagnostics", "RRFFusionResult", "RerankedCandidate",
]
