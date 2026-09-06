"""Discovery — Relational Fingerprinting (zero-knowledge structure learning).

Learns, with zero labels, from raw telemetry: signal dependence (linear +
nonlinear), operating regimes, and the per-regime relationship graph
("relational fingerprint"). This structure is the basis for regime-conditional
structural-drift anomaly detection.
"""
from .dependence import DependenceResult, pairwise_dependence, classify
from .regimes import segment_regimes, RegimeResult

__all__ = [
    "DependenceResult", "pairwise_dependence", "classify",
    "segment_regimes", "RegimeResult",
]
