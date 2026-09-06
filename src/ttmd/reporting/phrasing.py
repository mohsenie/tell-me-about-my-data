"""Deterministic number -> plain-language phrasing helpers.

Each function maps a measured quantity to an honest, human-readable phrase.
No invented meaning: only describes measured structure.
"""
from __future__ import annotations


def strength_word(dcor: float) -> str:
    if dcor >= 0.85:
        return "very strongly"
    if dcor >= 0.65:
        return "strongly"
    if dcor >= 0.45:
        return "moderately"
    if dcor >= 0.25:
        return "weakly"
    return "barely"


def kind_phrase(kind: str) -> str:
    return {
        "linear": "in a straight-line (linear) way",
        "nonlinear": "in a non-linear way (a curved or threshold relationship)",
        "none": "no meaningful relationship",
    }.get(kind, kind)


def describe_edge(edge: dict, cross_source: bool = False) -> str:
    """One sentence describing a relationship edge."""
    a, b = _pretty(edge["a"]), _pretty(edge["b"])
    strength = strength_word(edge["dcor"])
    kind = kind_phrase(edge["kind"])
    note = ""
    if edge["kind"] == "nonlinear" and edge["pearson"] < 0.2:
        note = (" — notably, ordinary correlation would miss this "
                f"(linear score only {edge['pearson']:.2f})")
    tag = " [cross-source]" if cross_source else ""
    return (f"{a} and {b} are {strength} related {kind} "
            f"(dependence {edge['dcor']:.2f}){note}.{tag}")


def describe_regime(regime: dict, top_signals: list[str] | None = None) -> str:
    frac = regime["fraction"] * 100
    line = f"Operating mode {regime['label']}: present {frac:.0f}% of the time"
    if top_signals:
        c = regime["centroid"]
        parts = [f"{_pretty(s)}~{c[s]:.1f}" for s in top_signals if s in c]
        if parts:
            line += " (typical: " + ", ".join(parts) + ")"
    return line + "."


def separation_quality(silhouette: float | None) -> str:
    if silhouette is None:
        return "not assessed"
    if silhouette >= 0.6:
        return f"well-separated (score {silhouette:.2f}) — distinct, trustworthy modes"
    if silhouette >= 0.35:
        return f"moderately separated (score {silhouette:.2f})"
    return (f"weakly separated (score {silhouette:.2f}) — modes overlap; "
            "treat these regimes with caution")


def _pretty(col: str) -> str:
    """Make a column/prefixed name human-friendly (no meaning invented)."""
    if "__" in col:
        src, name = col.split("__", 1)
        return f"{name} ({src})"
    return col
