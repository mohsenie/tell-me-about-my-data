"""Dependence measures between signals: Pearson, distance correlation, MI.

Pure numeric functions + a classifier that combines them into a relationship
KIND (none / linear / nonlinear). No IO, no catalog knowledge.

- Pearson  : linear strength (misses nonlinear -> the zero-linear-corr blind spot)
- dCor     : any dependence; 0 iff independent (catches nonlinear)
- MI       : shared information (nonlinear-capable; estimation-sensitive)
Comparing Pearson vs dCor reveals the relationship's SHAPE.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

# Thresholds for classifying a relationship from (pearson, dcor).
DCOR_NONE = 0.15        # below this: treat as no relationship
PEARSON_LINEAR = 0.5    # dcor present AND pearson high -> linear
NONLINEAR_GAP = 0.2     # dcor - pearson beyond this -> nonlinear


class Kind(str, Enum):
    NONE = "none"
    LINEAR = "linear"
    NONLINEAR = "nonlinear"


@dataclass
class DependenceResult:
    a: str
    b: str
    pearson: float
    dcor: float
    mi: float | None
    kind: Kind

    def to_dict(self) -> dict:
        return {
            "a": self.a, "b": self.b,
            "pearson": round(self.pearson, 4),
            "dcor": round(self.dcor, 4),
            "mi": round(self.mi, 4) if self.mi is not None else None,
            "kind": self.kind.value,
        }


def _distance_correlation_naive(x: np.ndarray, y: np.ndarray) -> float:
    """Reference O(n^2) implementation (biased/sample estimator). Exact.

    Kept to verify the fast path and as a fallback if the `dcor` library is
    unavailable. Memory O(n^2) — only use for small n / testing.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3:
        return 0.0
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    A = a - a.mean(0) - a.mean(1)[:, None] + a.mean()
    B = b - b.mean(0) - b.mean(1)[:, None] + b.mean()
    dcov2 = (A * B).mean()
    vx = (A * A).mean(); vy = (B * B).mean()
    denom = np.sqrt(vx * vy)
    return float(np.sqrt(max(dcov2, 0.0) / denom)) if denom > 0 else 0.0


# Prefer the fast O(n log n) implementation from the `dcor` library when present.
# It computes the SAME biased distance correlation (Huo-Szekely fast algorithm),
# so results match the naive version to floating-point precision.
try:
    import dcor as _dcor_lib

    def distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x, float); y = np.asarray(y, float)
        if len(x) < 3:
            return 0.0
        # AVL method = fast O(n log n) for the univariate case.
        val = _dcor_lib.distance_correlation(x, y, method="avl")
        return float(val) if np.isfinite(val) else 0.0

    _DCOR_BACKEND = "dcor.avl (fast O(n log n))"
except ImportError:  # pragma: no cover - fallback path
    distance_correlation = _distance_correlation_naive
    _DCOR_BACKEND = "naive O(n^2) (install `dcor` for the fast path)"


def pearson_abs(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(abs(np.corrcoef(x, y)[0, 1]))


def classify(pearson: float, dcor: float) -> Kind:
    """Combine linear + any-dependence measures into a relationship kind."""
    if dcor < DCOR_NONE:
        return Kind.NONE
    if pearson > PEARSON_LINEAR:
        return Kind.LINEAR
    if dcor - pearson > NONLINEAR_GAP:
        return Kind.NONLINEAR
    return Kind.LINEAR  # weak/mixed -> treat as (weak) linear


def _mi_matrix(data: np.ndarray) -> np.ndarray | None:
    """Normalized mutual-information matrix, or None if sklearn unavailable."""
    try:
        from sklearn.feature_selection import mutual_info_regression
    except ImportError:
        return None
    p = data.shape[1]
    mi = np.zeros((p, p))
    for i in range(p):
        mi[i] = mutual_info_regression(data, data[:, i], random_state=0)
    mx = mi.max()
    return mi / mx if mx > 0 else mi


def pairwise_dependence(
    columns: list[str], data: np.ndarray, with_mi: bool = True,
) -> list[DependenceResult]:
    """Compute Pearson, dCor, (optional) MI and kind for every column pair.

    `data` is (n_samples, n_columns), already cleaned (no NaN, no constants).
    """
    p = data.shape[1]
    mi = _mi_matrix(data) if with_mi else None
    out: list[DependenceResult] = []
    for i in range(p):
        for j in range(i + 1, p):
            r = pearson_abs(data[:, i], data[:, j])
            d = distance_correlation(data[:, i], data[:, j])
            m = float(mi[i, j]) if mi is not None else None
            out.append(DependenceResult(columns[i], columns[j], r, d, m, classify(r, d)))
    return out
