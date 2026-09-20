"""Feature-gated retrieval V2 building blocks.

They are intentionally not wired into ordinary QA until a later phase.
"""
"""Internal Retrieval V2 building blocks; never the default QA path."""

from retrieval.fusion import FusedCandidate, RRFFusion, RRFFusionDiagnostics, RRFFusionResult

__all__ = ["FusedCandidate", "RRFFusion", "RRFFusionDiagnostics", "RRFFusionResult"]
