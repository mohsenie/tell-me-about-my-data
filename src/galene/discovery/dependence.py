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


# A pair whose PARTIAL correlation (controlling for all other signals) collapses
# to below this is treated as an INDUCED edge — its marginal association is mostly
# explained by common drivers, not a direct link.
PARTIAL_DIRECT = 0.1
# ...and only flagged as induced if the marginal Pearson was clearly higher, i.e.
# the association really did shrink under conditioning.
PARTIAL_SHRINK = 0.15


@dataclass
class DependenceResult:
    a: str
    b: str
    pearson: float
    dcor: float
    mi: float | None
    kind: Kind
    partial: float | None = None   # partial correlation |controlling for the rest|
    direct: bool | None = None     # False => association is mostly common-driver haze
    pvalue: float | None = None    # block-permutation significance (time-aware)
    significant: bool | None = None  # False => likely autocorrelation phantom

    def to_dict(self) -> dict:
        d = {
            "a": self.a, "b": self.b,
            "pearson": round(self.pearson, 4),
            "dcor": round(self.dcor, 4),
            "mi": round(self.mi, 4) if self.mi is not None else None,
            "kind": self.kind.value,
        }
        if self.partial is not None:
            d["partial"] = round(self.partial, 4)
            d["direct"] = bool(self.direct)
        if self.pvalue is not None:
            d["pvalue"] = round(self.pvalue, 4)
            d["significant"] = bool(self.significant)
        return d


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


def _block_permute(y: np.ndarray, n_blocks: int, rng) -> np.ndarray:
    """Shuffle y in CONTIGUOUS blocks, preserving local time structure within each
    block (so an autocorrelated series keeps its short-range shape). Breaking only
    the cross-block alignment is what tests whether x~y is real or an artifact of
    both being slow / autocorrelated."""
    n = len(y)
    if n_blocks < 2 or n_blocks > n:
        return y[rng.permutation(n)]
    # split into ~equal contiguous blocks, then reorder the blocks
    bounds = np.array_split(np.arange(n), n_blocks)
    order = rng.permutation(len(bounds))
    return np.concatenate([y[bounds[i]] for i in order])


def block_permutation_pvalue(x: np.ndarray, y: np.ndarray, observed: float | None = None,
                             n_perm: int = 99, n_blocks: int = 10,
                             seed: int = 0) -> float:
    """Significance of x~y dCor under a BLOCK permutation null (time-aware).

    Row-shuffling destroys ALL structure, so a slow/autocorrelated pair looks
    "significant" against it even when the association is a shared-trend artifact.
    Block permutation keeps each series' short-range time structure and only breaks
    the cross-series alignment, giving an HONEST null. Returns the fraction of
    permutations whose dCor >= the observed (a p-value); high => phantom edge.
    Deterministic (seeded)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 10:
        return 1.0
    if observed is None:
        observed = distance_correlation(x, y)
    if observed <= 0:
        return 1.0
    rng = np.random.default_rng(seed)
    ge = 1  # +1 (the observed counts as one draw) -> never a zero p-value
    for _ in range(n_perm):
        yp = _block_permute(y, n_blocks, rng)
        if distance_correlation(x, yp) >= observed:
            ge += 1
    return ge / (n_perm + 1)


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


def partial_correlation_matrix(data: np.ndarray) -> np.ndarray | None:
    """|Partial correlation| between every pair, controlling for ALL other signals.

    Computed from the PRECISION matrix (inverse of the correlation matrix):
        partial(i,j) = -P[i,j] / sqrt(P[i,i] * P[j,j]).
    A pair with high marginal correlation but near-zero partial correlation is an
    INDUCED (spurious) association — a shared driver, not a direct link. Ridge-
    regularized so it stays invertible on collinear / thin data. Returns a |value|
    matrix in [0,1], or None if it can't be computed (too few cols/rows)."""
    n, p = data.shape
    if p < 3 or n < p + 2:
        return None                      # partialling needs enough cols & rows
    # correlation matrix (standardize); guard zero-variance columns
    std = data.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    z = (data - data.mean(axis=0)) / std
    # a constant column yields all-zeros here; np.corrcoef then divides by a zero
    # norm and emits a benign RuntimeWarning. The ridge below handles the singular
    # result, so silence just that warning rather than leak it to the user.
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(z, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)             # constant-col rows -> 0 corr
    corr = np.atleast_2d(corr) + 1e-3 * np.eye(p)   # ridge for invertibility
    try:
        prec = np.linalg.inv(corr)
    except np.linalg.LinAlgError:
        prec = np.linalg.pinv(corr)
    dinv = np.sqrt(np.abs(np.diag(prec)))
    denom = np.outer(dinv, dinv)
    denom[denom == 0] = 1.0
    pcorr = -prec / denom
    return np.abs(np.clip(pcorr, -1.0, 1.0))


# an edge whose block-permutation p-value is above this is likely a time-structure
# (autocorrelation) phantom rather than a real dependence
PVALUE_SIG = 0.05


def pairwise_dependence(
    columns: list[str], data: np.ndarray, with_mi: bool = True,
    with_significance: bool = False, n_perm: int = 99,
) -> list[DependenceResult]:
    """Compute Pearson, dCor, (optional) MI, kind, PARTIAL correlation, and
    (optional) block-permutation SIGNIFICANCE for every column pair.

    `data` is (n_samples, n_columns), already cleaned (no NaN, no constants).
    The partial correlation controls for all other signals, distinguishing a
    DIRECT link from a common-driver-INDUCED one. with_significance adds a
    time-aware block-permutation p-value per candidate edge (dcor>=DCOR_NONE only,
    to bound cost) so autocorrelation phantoms can be flagged. Both extras are
    None when not computed / not applicable — reported honestly."""
    p = data.shape[1]
    mi = _mi_matrix(data) if with_mi else None
    pcorr = partial_correlation_matrix(data)
    out: list[DependenceResult] = []
    for i in range(p):
        for j in range(i + 1, p):
            r = pearson_abs(data[:, i], data[:, j])
            d = distance_correlation(data[:, i], data[:, j])
            m = float(mi[i, j]) if mi is not None else None
            partial = float(pcorr[i, j]) if pcorr is not None else None
            direct = None
            if partial is not None:
                # induced = partial collapses AND it clearly shrank vs marginal
                induced = (partial < PARTIAL_DIRECT
                           and (r - partial) > PARTIAL_SHRINK)
                direct = not induced
            pval = sig = None
            if with_significance and d >= DCOR_NONE:
                pval = block_permutation_pvalue(data[:, i], data[:, j], observed=d,
                                                n_perm=n_perm, seed=0)
                sig = pval <= PVALUE_SIG
            out.append(DependenceResult(columns[i], columns[j], r, d, m,
                                        classify(r, d), partial=partial, direct=direct,
                                        pvalue=pval, significant=sig))
    return out
