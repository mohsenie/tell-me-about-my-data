"""Unsupervised operating-regime segmentation.

Separates USAGE PATTERNS (off / idle / cruise / maneuver ...) from ANOMALIES:
anomaly detection later compares like-for-like regime, never across regimes.

Approach (label-free): standardize signals, cluster time windows with KMeans,
pick k by silhouette. Each cluster = an operating mode. Returns per-row regime
labels + a human-readable profile of each regime (its signal centroid).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Regime:
    label: int
    size: int
    fraction: float
    centroid: dict[str, float]          # mean of each signal in this regime
    def to_dict(self) -> dict:
        return {
            "label": self.label, "size": self.size,
            "fraction": round(self.fraction, 4),
            "centroid": {k: round(v, 3) for k, v in self.centroid.items()},
        }


@dataclass
class RegimeResult:
    k: int
    labels: np.ndarray                  # per-row regime label
    regimes: list[Regime] = field(default_factory=list)
    silhouette: float | None = None
    # fitted model (for ASSIGNING new data to THESE regimes later — anomaly
    # detection must judge like-for-like regime, so a new window is assigned to
    # the baseline's regimes, not re-clustered). None on the degenerate path.
    columns: list[str] | None = None
    scaler_mean: np.ndarray | None = None
    scaler_scale: np.ndarray | None = None
    centers: np.ndarray | None = None   # KMeans cluster_centers_ in SCALED space

    def summary(self) -> dict:
        return {
            "k": self.k,
            "silhouette": round(self.silhouette, 4) if self.silhouette else None,
            "regimes": [r.to_dict() for r in self.regimes],
        }

    def model_params(self) -> dict | None:
        """Serializable regime model: column order + scaler + cluster centers.
        Enough to ASSIGN any new row to one of these regimes with no sklearn
        object and no re-clustering. None if no model was fit (degenerate)."""
        if self.centers is None or self.columns is None:
            return None
        return {
            "columns": list(self.columns),
            "scaler_mean": [float(x) for x in self.scaler_mean],
            "scaler_scale": [float(x) for x in self.scaler_scale],
            "centers": [[float(x) for x in row] for row in self.centers],
        }


def label_sequence(model: dict, columns: list[str], data: np.ndarray,
                   ts: np.ndarray):
    """Time-ordered (label, timestamp) sequence for rows, assigning each row to a
    regime via the persisted `model`. Rows whose MODEL columns are all-NaN are
    dropped (can't assign). `data`/`ts` must be time-ordered and aligned (as from
    load_numeric_with_time). Returns (labels, times) as aligned np arrays. Used
    for regime-transition detection — needs temporal order the graph path loses."""
    mcols = model["columns"]
    idx = {c: i for i, c in enumerate(columns)}
    present = [c for c in mcols if c in idx]
    if not present or len(data) == 0:
        return np.empty(0, int), np.empty(0)
    cols_i = [idx[c] for c in present]
    sub = data[:, cols_i]
    ok = ~np.isnan(sub).all(axis=1)          # keep rows with >=1 model signal
    labels = assign_labels(model, columns, data[ok])
    times = ts[ok] if len(ts) == len(data) else np.arange(int(ok.sum()))
    return labels, times


def assign_labels(model: dict, columns: list[str], data: np.ndarray) -> np.ndarray:
    """Assign each row of `data` (with its own `columns` order) to the nearest
    regime in a persisted `model` (from RegimeResult.model_params). Standardizes
    with the model's scaler, then nearest cluster center — exactly what
    KMeans.predict does, but from plain arrays so no sklearn state is needed.

    Columns are aligned BY NAME to the model's training columns; any model column
    absent from `data` is filled with the scaler mean (-> 0 after scaling, i.e.
    neutral). Returns an int label per row (index into model["centers"])."""
    mcols = model["columns"]
    mean = np.asarray(model["scaler_mean"], float)
    scale = np.asarray(model["scaler_scale"], float)
    centers = np.asarray(model["centers"], float)
    scale = np.where(scale == 0, 1.0, scale)

    idx = {c: i for i, c in enumerate(columns)}
    # build a (n_rows, n_model_cols) matrix in the model's column order
    n = len(data)
    X = np.empty((n, len(mcols)), float)
    for j, c in enumerate(mcols):
        if c in idx:
            X[:, j] = data[:, idx[c]]
        else:
            X[:, j] = mean[j]        # missing column -> neutral (mean)
    # scale, then nearest center (squared Euclidean)
    Xs = (X - mean) / scale
    # (n, 1, d) - (1, k, d) -> distances (n, k)
    d2 = ((Xs[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return d2.argmin(axis=1).astype(int)


def segment_regimes(
    columns: list[str], data: np.ndarray, k_range=(2, 6), max_samples=5000,
) -> RegimeResult:
    """Cluster rows into operating regimes. Chooses k by silhouette score.

    `data` is (n_samples, n_columns), cleaned. Returns labels for ALL rows
    (model is fit on a subsample for speed, then predicts all).
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

    scaler = StandardScaler()
    scaled_all = scaler.fit_transform(data)

    # fit/select k on a subsample for speed
    rng = np.random.default_rng(0)
    idx = (rng.choice(len(scaled_all), max_samples, replace=False)
           if len(scaled_all) > max_samples else np.arange(len(scaled_all)))
    sample = scaled_all[idx]

    best = None
    for k in range(k_range[0], k_range[1] + 1):
        if k >= len(sample):
            break
        km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(sample)
        try:
            score = silhouette_score(sample, km.labels_)
        except ValueError:
            continue
        if best is None or score > best[0]:
            best = (score, k, km)

    if best is None:
        # degenerate: everything one regime
        return RegimeResult(k=1, labels=np.zeros(len(data), int),
                            regimes=[_profile(0, columns, data, np.ones(len(data), bool))])

    score, k, km = best
    labels = km.predict(scaled_all)

    regimes = []
    for lab in range(k):
        mask = labels == lab
        if mask.any():
            regimes.append(_profile(lab, columns, data, mask))
    regimes.sort(key=lambda r: r.size, reverse=True)
    return RegimeResult(
        k=k, labels=labels, regimes=regimes, silhouette=score,
        columns=list(columns),
        scaler_mean=scaler.mean_.copy(), scaler_scale=scaler.scale_.copy(),
        centers=km.cluster_centers_.copy(),
    )


def _profile(label: int, columns: list[str], data: np.ndarray, mask: np.ndarray) -> Regime:
    n = int(mask.sum())
    centroid = {c: float(data[mask, i].mean()) for i, c in enumerate(columns)}
    return Regime(label=label, size=n, fraction=n / len(data), centroid=centroid)
