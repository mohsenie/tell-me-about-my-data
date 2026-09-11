"""Discovery — Relational Fingerprinting (zero-knowledge structure learning).

Learns, with zero labels, from raw telemetry: signal dependence (linear +
nonlinear), operating regimes, and the per-regime relationship graph
("relational fingerprint"). This structure is the basis for regime-conditional
structural-drift anomaly detection.
"""
from .dependence import (
    DependenceResult, pairwise_dependence, classify, partial_correlation_matrix,
    block_permutation_pvalue)
from .regimes import segment_regimes, RegimeResult, assign_labels, label_sequence
from .loader import load_numeric, load_numeric_with_time

__all__ = [
    "DependenceResult", "pairwise_dependence", "classify",
    "partial_correlation_matrix", "block_permutation_pvalue",
    "segment_regimes", "RegimeResult", "assign_labels", "label_sequence",
    "load_numeric", "load_numeric_with_time",
]
