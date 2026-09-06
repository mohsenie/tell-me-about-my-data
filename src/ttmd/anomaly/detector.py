"""Drift detector — compare a new window's fingerprint to the known-good baseline.

Two layers, reported distinctly (never asserting cause — a route change and a
developing fault look alike from data):

  Layer 1 — behavioral anomaly (within-regime structure drift = likely FAULT):
    match each new regime to a baseline regime by nearest centroid, diff the
    per-edge dcor maps, flag edges whose change exceeds the normal-variation
    band, rank by magnitude (invariant #6), scale confidence by data volume (#7).

  Layer 2 — regime events (usage changed, often benign):
    NEW regime not in the baseline (invariant #8: neutral 'unseen mode', not a
    fault), and regime DISTRIBUTION shift (e.g. 60% cruise -> 90% idle).
"""
from __future__ import annotations

import math

import numpy as np

import config
from ttmd.discovery.loader import load_numeric
from ttmd.discovery.relationships import (
    build_per_regime_graphs, graphs_for_fixed_regimes)
from .baseline import _fingerprint, _fingerprint_from_fixed, _edge_key


# how many multiples of the normal-variation band counts as a flag
_BAND_MULT = 3.0
# a new regime's centroid must be at least this far (scaled) from every baseline
# centroid to count as genuinely unseen
_NEW_REGIME_DIST = 2.5
# a distribution shift this large (absolute change in fraction) is reported
_DIST_SHIFT = 0.15


def _centroid_distance(c1: dict, c2: dict) -> float:
    """Euclidean distance between two centroids over their SHARED signals,
    normalized by signal count so it's comparable across regimes. Standardized
    per-signal would need the scaler; centroids are in raw units, so we scale
    each dim by (|c1|+|c2|)/2 to keep it unit-agnostic (relative difference)."""
    keys = set(c1) & set(c2)
    if not keys:
        return float("inf")
    acc = 0.0
    for k in keys:
        a, b = c1[k], c2[k]
        scale = (abs(a) + abs(b)) / 2.0 or 1.0
        acc += ((a - b) / scale) ** 2
    return math.sqrt(acc / len(keys))


def _match_regimes(new_fp: dict, base_fp: dict) -> tuple[dict, list[str]]:
    """Match NEW regimes to BASELINE regimes by centroid, ONE-TO-ONE (greedy on
    the globally-closest pairs). KMeans labels aren't stable across runs, so a
    naive nearest-match can send two new regimes to the same baseline regime and
    manufacture phantom drift. Enforcing a bijection avoids that: each baseline
    regime is claimed by at most one new regime (its closest), and a new regime
    too far from EVERY remaining baseline regime is 'unseen' (invariant #8).
    Returns ({new_label: base_label}, [unmatched new_labels])."""
    base = base_fp["regimes"]
    # all candidate pairs with their centroid distance, closest first
    pairs = []
    for nlab, nrg in new_fp["regimes"].items():
        for blab, brg in base.items():
            d = _centroid_distance(nrg.get("centroid", {}), brg.get("centroid", {}))
            pairs.append((d, nlab, blab))
    pairs.sort(key=lambda p: p[0])

    matches, used_base, used_new = {}, set(), set()
    for d, nlab, blab in pairs:
        if nlab in used_new or blab in used_base:
            continue
        if d > _NEW_REGIME_DIST:
            break   # remaining pairs are only farther
        matches[nlab] = blab
        used_new.add(nlab); used_base.add(blab)
    unmatched = [nlab for nlab in new_fp["regimes"] if nlab not in used_new]
    return matches, unmatched


def _volume_confidence(n_samples: int) -> str:
    if n_samples >= 5000:
        return "high"
    if n_samples >= 1000:
        return "medium"
    return "low"


def _layer1_structure_drift(new_fp, base_fp, band, matches) -> list[dict]:
    """Within-regime edge-structure changes beyond the variation band."""
    per_edge_std = band.get("per_edge_std", {})
    default_std = band.get("summary", {}).get("p90_edge_std", 0.12)
    findings = []
    for nlab, blab in matches.items():
        new_edges = new_fp["regimes"][nlab]["edges"]
        base_edges = base_fp["regimes"][blab]["edges"]
        n_samples = new_fp["regimes"][nlab].get("n_samples", 0)
        edge_std = per_edge_std.get(blab, {})
        changed = []
        for ek in set(new_edges) | set(base_edges):
            nd = new_edges.get(ek, 0.0)
            bd = base_edges.get(ek, 0.0)
            delta = nd - bd
            thr = _BAND_MULT * edge_std.get(ek, default_std)
            if abs(delta) > max(thr, 0.05):   # floor so tiny bands don't over-flag
                a, b = ek.split("::")
                kind = ("weakened" if delta < 0 else "strengthened")
                if bd == 0.0:
                    kind = "appeared"
                elif nd == 0.0:
                    kind = "disappeared"
                changed.append({
                    "edge": [a, b], "baseline_dcor": round(bd, 3),
                    "window_dcor": round(nd, 3), "delta": round(delta, 3),
                    "threshold": round(thr, 3), "change": kind,
                })
        if changed:
            changed.sort(key=lambda c: abs(c["delta"]), reverse=True)
            findings.append({
                "regime": nlab, "baseline_regime": blab,
                "confidence": _volume_confidence(n_samples),
                "n_samples": n_samples,
                "changed_edges": changed,
            })
    return findings


def _layer2_regime_events(new_fp, base_fp, matches, unmatched) -> list[dict]:
    """New regimes (unseen mode) + regime distribution shifts. Uses the SAME
    one-to-one matches as Layer 1 so the two layers can't disagree."""
    events = []
    # new/unseen regimes (invariant #8: neutral, not a fault)
    for nlab in unmatched:
        rg = new_fp["regimes"][nlab]
        events.append({
            "type": "new_regime",
            "regime": nlab,
            "fraction": round(rg.get("fraction", 0.0), 3),
            "note": ("an operating mode not seen in the baseline — reported as an "
                     "unseen mode, NOT a confirmed problem"),
        })
    # distribution shift on matched regimes
    for nlab, blab in matches.items():
        bf = base_fp["regimes"][blab].get("fraction", 0.0)
        nf = new_fp["regimes"][nlab].get("fraction", 0.0)
        if abs(nf - bf) >= _DIST_SHIFT:
            events.append({
                "type": "distribution_shift",
                "regime": nlab, "baseline_regime": blab,
                "baseline_fraction": round(bf, 3),
                "window_fraction": round(nf, 3),
                "delta": round(nf - bf, 3),
                "note": ("time spent in this mode changed — usage change "
                         "(e.g. different route/operation), not necessarily a fault"),
            })
    return events


def detect_drift(source: str, dates: list[str], vessel: str,
                 baseline: dict, with_mi: bool = False) -> dict:
    """Compare the new window (given date partitions) to the known-good baseline.
    Returns a structured report with layer1 (structure drift) + layer2 (events).

    If the baseline persists a regime MODEL, the new window is ASSIGNED to those
    exact regimes (stable identity, like-for-like diff). Older baselines with no
    model fall back to re-clustering + centroid matching."""
    globs = [g for g in config.source_globs_for_dates(source, dates, vessel)
             if _exists(g)]
    if not globs:
        return {"error": f"no data for '{source}' in the requested window."}
    cols, data = load_numeric(globs)
    base_fp = baseline["fingerprint"]
    band = baseline.get("variation", {})
    model = baseline.get("regime_model")
    window = dates[0] if len(dates) == 1 else f"{dates[0]}..{dates[-1]}"

    if model is not None:
        # ASSIGN to baseline regimes -> identity match (label L vs baseline L).
        fixed = graphs_for_fixed_regimes(cols, data, model, with_mi=with_mi)
        new_fp = _fingerprint_from_fixed(fixed)
        matches = {lab: lab for lab in new_fp["regimes"]}   # identity
        layer1 = _layer1_structure_drift(new_fp, base_fp, band, matches)
        layer2 = _layer2_regime_events(new_fp, base_fp, matches, [])
        return {
            "source": source, "baseline_window": baseline.get("window"),
            "window": window, "assignment": "fixed-model (stable regime identity)",
            "layer1_structure_drift": layer1,
            "layer2_regime_events": layer2,
            "regime_matches": matches, "unmatched_regimes": [],
        }

    # legacy path: re-cluster + centroid match (baselines built before model reuse)
    disc = build_per_regime_graphs(cols, data, with_mi=with_mi)
    new_fp = _fingerprint(disc)
    matches, unmatched = _match_regimes(new_fp, base_fp)
    layer1 = _layer1_structure_drift(new_fp, base_fp, band, matches)
    layer2 = _layer2_regime_events(new_fp, base_fp, matches, unmatched)
    return {
        "source": source, "baseline_window": baseline.get("window"),
        "window": window, "assignment": "re-clustered + centroid-matched (legacy)",
        "layer1_structure_drift": layer1,
        "layer2_regime_events": layer2,
        "regime_matches": matches, "unmatched_regimes": unmatched,
    }


def _exists(glob_str: str) -> bool:
    import glob as _g
    return bool(_g.glob(glob_str))
