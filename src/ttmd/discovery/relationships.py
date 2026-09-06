"""Relationship-graph discovery: pairwise dependence, optionally per regime.

Produces the meaning-free relationship graph (nodes = signals, edges = learned
dependence with strength + kind), globally and/or split by operating regime.
This per-regime graph is the "relational fingerprint" — the basis for
regime-conditional structural-drift anomaly detection.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dependence import pairwise_dependence, DependenceResult
from .regimes import segment_regimes, RegimeResult


@dataclass
class RelationshipGraph:
    scope: str                                   # "global" or "regime:<label>"
    n_samples: int
    edges: list[DependenceResult] = field(default_factory=list)

    def significant(self, min_dcor: float = 0.15) -> list[DependenceResult]:
        return [e for e in self.edges if e.dcor >= min_dcor]

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "n_samples": self.n_samples,
            "edges": [e.to_dict() for e in self.significant()],
        }


def clean_frame(columns: list[str], data: np.ndarray,
                max_col_null: float = 0.5) -> tuple[list[str], np.ndarray]:
    """Prepare data for dependence analysis.

    Drops (a) constant / all-NaN columns and (b) columns that are mostly NaN
    (default >50%) BEFORE dropping rows — otherwise a few sparse columns (e.g.
    AIS static fields present only in some message types) would wipe every row.
    Then drops any remaining rows with NaN. Returns ([], empty) if nothing usable.
    """
    n = len(data)
    if n == 0:
        return [], data.reshape(0, 0)

    keep = []
    for i in range(data.shape[1]):
        col = data[:, i]
        null_frac = np.isnan(col).mean()
        if null_frac > max_col_null:
            continue
        if np.all(np.isnan(col)) or np.nanstd(col) == 0:
            continue
        keep.append(i)

    cols = [columns[i] for i in keep]
    if not cols:
        return [], data[:0, :0]

    sub = data[:, keep]
    row_ok = ~np.isnan(sub).any(axis=1)
    return cols, sub[row_ok]


def build_graph(columns, data, scope="global", with_mi=True, max_samples=None):
    """Relationship graph over the given rows (a window/day/month/regime slice).

    dCor is O(n^2) per pair and O(p^2) in pairs, so for wide sources (many
    signals) we shrink the row sample to keep runtime bounded. Sample cap scales
    down with signal count unless max_samples is given explicitly.
    """
    cols, clean = clean_frame(columns, data)
    if max_samples is None:
        # Fast O(n log n) dCor lets us use far more samples than the old naive
        # O(n^2) path allowed -> more accurate estimates at similar runtime.
        p = max(len(cols), 1)
        max_samples = 10000 if p <= 15 else (5000 if p <= 30 else 2500)
    if len(clean) > max_samples:
        rng = np.random.default_rng(0)
        clean = clean[rng.choice(len(clean), max_samples, replace=False)]
    edges = pairwise_dependence(cols, clean, with_mi=with_mi)
    return RelationshipGraph(scope=scope, n_samples=len(clean), edges=edges)


def build_per_regime_graphs(columns, data, with_mi=True) -> dict:
    """Segment into regimes, then build a relationship graph per regime.

    Returns {"regimes": RegimeResult.summary, "graphs": {scope: graph.to_dict}}.
    Per-regime graphs are the trustworthy ones: relationships that hold WITHIN a
    usage mode, not blurred across modes.
    """
    cols, clean = clean_frame(columns, data)
    if not cols or len(clean) < 50:
        return {
            "regimes": {"k": 0, "silhouette": None, "regimes": [],
                        "note": "insufficient varying data after cleaning"},
            "graphs": {},
        }
    reg = segment_regimes(cols, clean)
    graphs = {"global": build_graph(cols, clean, "global", with_mi).to_dict()}
    for r in reg.regimes:
        mask = reg.labels == r.label
        if mask.sum() >= 50:  # need enough rows for a meaningful graph
            g = build_graph(cols, clean[mask], f"regime:{r.label}", with_mi)
            graphs[f"regime:{r.label}"] = g.to_dict()
    out = {"regimes": reg.summary(), "graphs": graphs}
    # Surface the fitted regime model so callers (baseline store) can persist it
    # and ASSIGN future windows to THESE regimes instead of re-clustering.
    model = reg.model_params()
    if model is not None:
        out["regime_model"] = model
    return out


def graphs_for_fixed_regimes(columns, data, model, with_mi=False) -> dict:
    """Build per-regime relationship graphs by ASSIGNING rows to an EXISTING
    regime model (no re-clustering). This is what anomaly detection uses so the
    new window's regimes have the SAME identity as the baseline's — a like-for-
    like structural comparison, not two independent clusterings.

    Returns {"graphs": {"global":..., "regime:<label>":...},
             "fractions": {label: fraction}}  keyed by the MODEL's labels.
    """
    from .regimes import assign_labels

    cols, clean = clean_frame(columns, data)
    if not cols or len(clean) < 50:
        return {"graphs": {}, "fractions": {}}
    labels = assign_labels(model, cols, clean)
    graphs = {"global": build_graph(cols, clean, "global", with_mi).to_dict()}
    fractions = {}
    total = len(clean)
    for lab in range(len(model["centers"])):
        mask = labels == lab
        frac = int(mask.sum()) / total if total else 0.0
        fractions[str(lab)] = round(frac, 4)
        if mask.sum() >= 50:
            g = build_graph(cols, clean[mask], f"regime:{lab}", with_mi)
            graphs[f"regime:{lab}"] = g.to_dict()
    return {"graphs": graphs, "fractions": fractions}
