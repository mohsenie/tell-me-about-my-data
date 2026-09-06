"""Baseline store — persist a KNOWN-GOOD fingerprint to compare against later.

A baseline captures, from a fixed known-good window (invariant #5, NOT rolling):
  - per-regime relationship fingerprints (the edge dcor map per regime),
  - each regime's centroid + fraction (for regime matching + distribution),
  - a normal-variation BAND: how much same-regime fingerprints differ day-to-day
    within the baseline window (so drift is judged against natural variation,
    invariant beyond which we flag).

Stored as JSON under artifacts/, keyed by source. Built from the SAME discovery
machinery (build_per_regime_graphs), so it stays consistent with what the rest of
the system computes — nothing here re-implements dependence.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np

import config
from ttmd.discovery.loader import load_numeric
from ttmd.discovery.relationships import (
    build_per_regime_graphs, graphs_for_fixed_regimes)


def _baseline_path(source: str) -> Path:
    return config.ARTIFACTS_DIR / f"baseline_{source}.json"


def has_baseline(source: str) -> bool:
    return _baseline_path(source).exists()


def _edge_key(a: str, b: str) -> str:
    """Order-independent key for an undirected edge (dcor/pearson are symmetric)."""
    return "::".join(sorted((a, b)))


def _edge_map(graph: dict) -> dict:
    """{edge_key: dcor} for a graph dict (graphs[scope]). Edges are already the
    significant ones (dcor >= 0.15) per RelationshipGraph.to_dict()."""
    return {_edge_key(e["a"], e["b"]): e["dcor"] for e in graph.get("edges", [])}


def _fingerprint(discovery: dict) -> dict:
    """Reduce a discovery result to a comparable fingerprint:
    per-regime edge maps + regime centroids/fractions + the global edge map."""
    graphs = discovery.get("graphs", {})
    regimes = discovery.get("regimes", {}).get("regimes", [])
    per_regime = {}
    for r in regimes:
        scope = f"regime:{r['label']}"
        if scope in graphs:
            per_regime[str(r["label"])] = {
                "edges": _edge_map(graphs[scope]),
                "n_samples": graphs[scope].get("n_samples", 0),
                "centroid": r.get("centroid", {}),
                "fraction": r.get("fraction", 0.0),
            }
    return {
        "global_edges": _edge_map(graphs.get("global", {})),
        "regimes": per_regime,
        "regime_count": discovery.get("regimes", {}).get("k", 0),
    }


def _variation_band(day_fingerprints: list[dict]) -> dict:
    """Estimate the NORMAL day-to-day variation of same-regime edge strengths
    across the baseline window. For each regime, match its edges across the
    per-day fingerprints and record the stddev of each edge's dcor. A single
    aggregate band (median + p90 of per-edge stddevs) is used as the threshold
    scale when we don't have enough days for a per-edge estimate.

    Fewer than 2 usable days -> a conservative default band (so we still flag big
    moves but don't over-alert on thin evidence, invariant #7)."""
    # collect, per regime label, per edge, the list of dcor values seen per day
    by_regime: dict[str, dict[str, list[float]]] = {}
    for fp in day_fingerprints:
        for lab, rg in fp.get("regimes", {}).items():
            reg = by_regime.setdefault(lab, {})
            for ek, d in rg["edges"].items():
                reg.setdefault(ek, []).append(d)

    per_edge_std: dict[str, dict[str, float]] = {}
    all_stds: list[float] = []
    for lab, edges in by_regime.items():
        per_edge_std[lab] = {}
        for ek, vals in edges.items():
            if len(vals) >= 2:
                s = statistics.pstdev(vals)
                per_edge_std[lab][ek] = round(s, 4)
                all_stds.append(s)

    if all_stds:
        band = {
            "median_edge_std": round(statistics.median(all_stds), 4),
            "p90_edge_std": round(float(np.percentile(all_stds, 90)), 4),
            "n_days": len(day_fingerprints),
        }
    else:
        # not enough days to measure variation -> conservative default
        band = {"median_edge_std": 0.08, "p90_edge_std": 0.12,
                "n_days": len(day_fingerprints), "note": "default band (few days)"}
    return {"per_edge_std": per_edge_std, "summary": band}


def build_baseline(source: str, dates: list[str], vessel: str,
                   with_mi: bool = False) -> dict:
    """Build a known-good baseline for `source` over the given date partitions.

    - Runs discovery over the WHOLE window -> the reference fingerprint.
    - Runs discovery per-day over the window -> estimates the normal-variation
      band (how much same-regime fingerprints wobble day-to-day).
    Returns the baseline dict (also persist it with save_baseline)."""
    globs = config.source_globs_for_dates(source, dates, vessel)
    globs = [g for g in globs if _exists(g)]
    if not globs:
        return {}
    cols, data = load_numeric(globs)
    whole = build_per_regime_graphs(cols, data, with_mi=with_mi)
    ref = _fingerprint(whole)
    model = whole.get("regime_model")   # fitted regime model (may be None)

    # Per-day fingerprints for the variation band. Assign each day to the SAME
    # baseline regimes (fixed model) so the band measures how a FIXED regime's
    # edges wobble day-to-day — not the noise of re-clustering each day.
    day_fps = []
    for d in dates:
        dg = [g for g in config.source_globs_for_dates(source, [d], vessel) if _exists(g)]
        if not dg:
            continue
        c, dat = load_numeric(dg)
        if len(dat) < 50:
            continue
        if model is not None:
            fixed = graphs_for_fixed_regimes(c, dat, model, with_mi=with_mi)
            day_fps.append(_fingerprint_from_fixed(fixed))
        else:
            day_fps.append(_fingerprint(build_per_regime_graphs(c, dat, with_mi=with_mi)))

    band = _variation_band(day_fps)
    out = {
        "source": source,
        "vessel": vessel,
        "window": (dates[0] if len(dates) == 1 else f"{dates[0]}..{dates[-1]}"),
        "dates": dates,
        "fingerprint": ref,
        "variation": band,
        "regime_summary": whole.get("regimes", {}),
    }
    if model is not None:
        out["regime_model"] = model
    return out


def _fingerprint_from_fixed(fixed: dict) -> dict:
    """Fingerprint from graphs_for_fixed_regimes output (already keyed by model
    labels). Mirrors _fingerprint's per-regime edge-map shape so the variation
    band and the detector diff operate on the same structure."""
    graphs = fixed.get("graphs", {})
    fractions = fixed.get("fractions", {})
    per_regime = {}
    for scope, g in graphs.items():
        if not scope.startswith("regime:"):
            continue
        lab = scope.split(":", 1)[1]
        per_regime[lab] = {
            "edges": _edge_map(g),
            "n_samples": g.get("n_samples", 0),
            "fraction": fractions.get(lab, 0.0),
        }
    return {
        "global_edges": _edge_map(graphs.get("global", {})),
        "regimes": per_regime,
        "regime_count": len(fractions),
    }


def save_baseline(baseline: dict) -> Path:
    path = _baseline_path(baseline["source"])
    path.write_text(json.dumps(baseline, indent=2, default=str))
    return path


def load_baseline(source: str) -> dict:
    return json.loads(_baseline_path(source).read_text())


def _exists(glob_str: str) -> bool:
    import glob as _g
    return bool(_g.glob(glob_str))
