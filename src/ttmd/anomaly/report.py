"""Human-readable rendering of a drift-detection result.

Presentation rules (from the spec):
  - Label the TWO types distinctly and with different severity: behavioral
    anomaly (likely fault) vs regime event (usage change, often benign).
  - Report the OBSERVED change, never assert the cause.
  - Novelty != fault: a new regime is described neutrally.
"""
from __future__ import annotations


def render_drift(result: dict) -> str:
    if result.get("error"):
        return result["error"]
    lines = []
    lines.append(f"Drift check for '{result['source']}': "
                 f"window {result['window']} vs known-good {result['baseline_window']}")
    lines.append("")

    l1 = result.get("layer1_structure_drift", [])
    l2 = result.get("layer2_regime_events", [])

    # Layer 1 — behavioral anomaly (likely fault)
    lines.append("BEHAVIORAL ANOMALY (within-mode relationship drift — possible fault):")
    if not l1:
        lines.append("  none — relationship structure matches the baseline within "
                     "normal variation.")
    for f in l1:
        lines.append(f"  regime {f['regime']} (vs baseline regime {f['baseline_regime']}, "
                     f"{f['n_samples']:,} samples, confidence: {f['confidence']}):")
        for c in f["changed_edges"][:8]:
            a, b = c["edge"]
            lines.append(f"    - {a} ~ {b}: {c['change']} "
                         f"({c['baseline_dcor']:.2f} -> {c['window_dcor']:.2f}, "
                         f"Δ{c['delta']:+.2f}, threshold {c['threshold']:.2f})")
    lines.append("")

    # Layer 2 — regime events (usage change)
    lines.append("REGIME EVENTS (usage/operating-mode change — often benign):")
    if not l2:
        lines.append("  none — operating modes and their distribution match the baseline.")
    for e in l2:
        if e["type"] == "new_regime":
            lines.append(f"  - NEW operating mode (regime {e['regime']}, "
                         f"{e['fraction']*100:.0f}% of the window): {e['note']}")
        elif e["type"] == "distribution_shift":
            lines.append(f"  - usage shift in regime {e['regime']}: "
                         f"{e['baseline_fraction']*100:.0f}% -> "
                         f"{e['window_fraction']*100:.0f}% of time "
                         f"(Δ{e['delta']*100:+.0f} points). {e['note']}")
    lines.append("")
    lines.append("Note: these are OBSERVED changes in the data. A change can come "
                 "from a developing fault OR a legitimate operational change "
                 "(route, load, weather) — the data alone can't tell which.")
    return "\n".join(lines)
