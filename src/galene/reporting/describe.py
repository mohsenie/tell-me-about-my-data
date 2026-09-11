"""Build a data-intelligence report (dict) from a discovery artifact, and render
it to markdown. Deterministic: describes only measured structure.
"""
from __future__ import annotations

from . import phrasing as ph


def describe_discovery(artifact: dict, source_name: str) -> dict:
    """Turn a discovery artifact into a structured, human-readable report dict."""
    reg = artifact.get("regimes", {})
    graphs = artifact.get("graphs", {})

    # pick informative signals to characterize regimes (highest variance across centroids)
    top_signals = _top_varying_signals(reg.get("regimes", []))

    report = {
        "source": source_name,
        "overview": _overview(source_name, reg, graphs),
        "regimes": {
            "quality": ph.separation_quality(reg.get("silhouette")),
            "count": reg.get("k", 0),
            "descriptions": [ph.describe_regime(r, top_signals) for r in reg.get("regimes", [])],
        },
        "relationships": _relationship_section(graphs),
        # machine-facing edge facts (for the interpretation layer), strongest first
        "_edges": _edge_facts(graphs),
    }
    return report


def _edge_facts(graphs: dict) -> list[dict]:
    """Structured edge fields the interpretation layer consumes."""
    g = graphs.get("global", {})
    edges = [e for e in g.get("edges", []) if e["kind"] != "none"]
    edges.sort(key=lambda e: e["dcor"], reverse=True)
    out = []
    for e in edges:
        fact = {
            "a": ph._pretty(e["a"]),
            "b": ph._pretty(e["b"]),
            "dcor": e["dcor"],
            "pearson": e["pearson"],
            "kind": e["kind"],
            "strength": ph.strength_word(e["dcor"]),
            "fact": ph.describe_edge(e, _cross(e)),
        }
        # partial-correlation refinement (direct link vs common-driver haze)
        if e.get("direct") is not None:
            fact["direct"] = e["direct"]
            fact["partial"] = e.get("partial")
        out.append(fact)
    return out


def _overview(source, reg, graphs) -> str:
    k = reg.get("k", 0)
    g = graphs.get("global", {})
    n_edges = len(g.get("edges", []))
    nonlin = sum(1 for e in g.get("edges", []) if e["kind"] == "nonlinear")
    cross = sum(1 for e in g.get("edges", []) if _cross(e))
    parts = [f"Analyzed '{source}' with zero prior knowledge."]
    if k:
        parts.append(f"Found {k} distinct operating mode(s)")
    if n_edges:
        seg = f"{n_edges} signal relationship(s)"
        if nonlin:
            seg += f", {nonlin} of them non-linear (invisible to ordinary correlation)"
        if cross:
            seg += f", {cross} spanning different data sources"
        parts.append("and " + seg)
    return " ".join(parts) + "."


def _relationship_section(graphs: dict) -> dict:
    g = graphs.get("global", {})
    edges = [e for e in g.get("edges", []) if e["kind"] != "none"]
    edges.sort(key=lambda e: e["dcor"], reverse=True)

    strongest = [ph.describe_edge(e, _cross(e)) for e in edges[:10]]
    nonlinear = [ph.describe_edge(e, _cross(e)) for e in edges
                 if e["kind"] == "nonlinear"][:8]
    cross = [ph.describe_edge(e, True) for e in edges if _cross(e)][:8]

    return {
        "total": len(edges),
        "strongest": strongest,
        "nonlinear_highlights": nonlinear,
        "cross_source_highlights": cross,
    }


def _top_varying_signals(regimes: list, k: int = 4) -> list[str]:
    if not regimes:
        return []
    signals = list(regimes[0].get("centroid", {}).keys())
    spread = []
    for s in signals:
        vals = [r["centroid"].get(s) for r in regimes if s in r.get("centroid", {})]
        vals = [v for v in vals if v is not None]
        if len(vals) > 1:
            rng = max(vals) - min(vals)
            spread.append((abs(rng), s))
    spread.sort(reverse=True)
    return [s for _, s in spread[:k]]


def _cross(edge: dict) -> bool:
    """Cross-source only applies to fused (prefixed 'source__col') names."""
    if "__" not in edge["a"] or "__" not in edge["b"]:
        return False
    return edge["a"].split("__")[0] != edge["b"].split("__")[0]


def render_markdown(report: dict) -> str:
    L = []
    L.append(f"# Data Intelligence Report — {report['source']}")
    L.append("")
    L.append("> Generated with zero prior knowledge of the data. Describes only")
    L.append("> measured structure (relationships and operating modes), not meaning.")
    L.append("")
    L.append("## Overview")
    L.append(report["overview"])
    L.append("")

    r = report["regimes"]
    L.append(f"## Operating modes ({r['count']})")
    L.append(f"Separation quality: {r['quality']}.")
    L.append("")
    for d in r["descriptions"]:
        L.append(f"- {d}")
    L.append("")

    rel = report["relationships"]
    L.append(f"## Relationships found ({rel['total']})")
    L.append("")
    if rel["strongest"]:
        L.append("### Strongest relationships")
        for s in rel["strongest"]:
            L.append(f"- {s}")
        L.append("")
    if rel["nonlinear_highlights"]:
        L.append("### Non-linear relationships (ordinary correlation would miss these)")
        for s in rel["nonlinear_highlights"]:
            L.append(f"- {s}")
        L.append("")
    if rel["cross_source_highlights"]:
        L.append("### Cross-source relationships (different sensors that move together)")
        for s in rel["cross_source_highlights"]:
            L.append(f"- {s}")
        L.append("")

    interp = report.get("interpretation")
    if interp:
        L.append(f"## Possible explanations ({interp['asset_type']})")
        L.append("")
        L.append("> Hypotheses, not facts. Suggested causes for the measured")
        L.append("> relationships. Confirmed items reflect expert input.")
        L.append(f"> Knowledge base: {interp['knowledge_entries']} confirmed item(s).")
        L.append("")

        cross = interp.get("cross_relationship", [])
        if cross:
            L.append("### Cross-relationship analysis")
            L.append("_When several signals relate to the same one, a single mechanism may explain them._")
            L.append("")
            for c in cross:
                L.append(f"- **{c['hub']}** is related to: {', '.join(c['related'])}")
                L.append(f"  - _hypothesis:_ {c['analysis']}")
                if c.get("citations"):
                    L.append(f"  - _sources:_ {', '.join(dict.fromkeys(c['citations']))}")
            L.append("")

        L.append("### Per-relationship hypotheses")
        for it in interp["relationships"]:
            L.append(f"- **{it['a']} ~ {it['b']}** — {it['fact']}")
            if it["source"] == "user_confirmed":
                L.append(f"  - _confirmed by {it.get('author','user')}:_ {it['explanation']}")
            else:
                L.append(f"  - _hypothesis (unconfirmed):_ {it['explanation']}")
                if it.get("citations"):
                    L.append(f"  - _sources:_ {', '.join(dict.fromkeys(it['citations']))}")
        L.append("")

        L.append("### Help improve these")
        L.append("These explanations are the system's best guesses. If one is wrong "
                 "**please say why** — even a short reason is captured and makes every "
                 "future report smarter. Use:")
        L.append("")
        L.append("```")
        L.append('galene correct "SignalA" "SignalB" "your reason" \\')
        L.append('    --general-fact "the general truth about this asset"')
        L.append("```")
        L.append("")
    return "\n".join(L)
