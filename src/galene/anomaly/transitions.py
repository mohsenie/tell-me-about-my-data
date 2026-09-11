"""Regime-transition detection (Layer-2, temporal) — has the ORDER in which the
asset moves between operating modes changed?

The relational drift detector watches sensor-coupling structure WITHIN a regime;
the regime-event layer watches how much TIME is spent in each mode. This layer
watches the SEQUENCE: which mode tends to follow which, and how often. A pump that
used to go idle -> ramp -> run and now jumps idle -> run, or a ship that used to
cycle port <-> transit and now sits in one mode, is a change in *behavior over
time* that the time-fraction view alone can miss.

Method (deterministic, data-agnostic):
  - From a time-ordered per-row regime label sequence (discovery.label_sequence),
    collapse consecutive identical labels into RUNS, so a "transition" is a real
    mode change (i -> j, i != j), not the sampling rate.
  - Count i -> j transitions -> a transition matrix; normalize per source-mode to
    get P(next = j | current = i). This is the baseline transition fingerprint.
  - For a new window, compare its observed transitions to the baseline:
      * UNSEEN transition (i -> j never in baseline)      -> new sequencing
      * RARE transition (baseline P below a floor)         -> unusual sequencing
      * ABSENT transition (common in baseline, missing now)-> a usual step dropped
    Each is classified by significance, and confidence scales with how many
    baseline transitions defined "normal" (invariant #7).

Discipline: report the OBSERVED sequence change, never assert the cause. A new
idle->run jump can be a fault OR a deliberate operating choice — only the human
knows. Regime labels carry no hardcoded meaning; everything is by label index.
"""
from __future__ import annotations

import numpy as np


def runs_from_labels(labels) -> list[int]:
    """Collapse a time-ordered label sequence into consecutive RUNS, returning the
    ordered list of regime labels visited (each distinct from its predecessor).
    [0,0,0,1,1,2,2,2,0] -> [0,1,2,0]. This makes a 'transition' a genuine mode
    change rather than an artifact of the sampling rate."""
    labels = np.asarray(labels).astype(int)
    if labels.size == 0:
        return []
    keep = np.concatenate(([True], labels[1:] != labels[:-1]))
    return labels[keep].tolist()


def build_transition_matrix(labels, k: int | None = None) -> dict:
    """Build the transition fingerprint from a time-ordered label sequence.

    Returns:
      k                : number of regimes
      counts           : {"i->j": count} for i != j (self-loops excluded)
      probs            : {"i->j": P(next=j | current=i)} normalized per source i
      source_totals    : {"i": total transitions leaving i}
      n_transitions    : total observed mode changes
      visited          : sorted list of regime labels actually visited
    """
    seq = runs_from_labels(labels)
    if k is None:
        k = (max(seq) + 1) if seq else 0
    counts: dict[str, int] = {}
    source_totals: dict[str, int] = {}
    for a, b in zip(seq[:-1], seq[1:]):
        key = f"{a}->{b}"
        counts[key] = counts.get(key, 0) + 1
        source_totals[str(a)] = source_totals.get(str(a), 0) + 1
    probs = {}
    for key, c in counts.items():
        i = key.split("->", 1)[0]
        tot = source_totals.get(i, 0)
        probs[key] = round(c / tot, 4) if tot else 0.0
    return {
        "k": int(k),
        "counts": counts,
        "probs": probs,
        "source_totals": source_totals,
        "n_transitions": int(sum(counts.values())),
        "visited": sorted(set(seq)),
    }


# a baseline transition rarer than this share of its source-mode's exits is "rare"
_RARE_P = 0.05
# a baseline transition at least this common that vanishes in the window is "absent"
_COMMON_P = 0.20
# need at least this many baseline transitions before we trust "unseen/rare" calls
_MIN_BASELINE_TRANSITIONS = 8


def _confidence(n_baseline: int) -> str:
    if n_baseline >= 40:
        return "high"
    if n_baseline >= 15:
        return "medium"
    return "low"


def detect_transition_anomalies(window_labels, baseline_tm: dict) -> dict:
    """Compare a new window's regime-transition pattern to the baseline matrix.

    Classifies each notable difference:
      unseen_transition    : i->j occurred now, never in the baseline
      rare_transition      : i->j occurred now, baseline P(i->j) < _RARE_P
      absent_transition    : i->j common in baseline (P >= _COMMON_P), absent now
    Returns {findings:[...], window:{...}, confidence, n_baseline_transitions,
    reliable:bool}. Never asserts cause — reports the observed sequence change."""
    win = build_transition_matrix(window_labels, k=baseline_tm.get("k"))
    base_probs = baseline_tm.get("probs", {})
    base_counts = baseline_tm.get("counts", {})
    n_base = baseline_tm.get("n_transitions", 0)
    reliable = n_base >= _MIN_BASELINE_TRANSITIONS
    conf = _confidence(n_base)

    findings = []
    # transitions that HAPPENED in the window
    for key, c in win["counts"].items():
        a, b = key.split("->", 1)
        wp = win["probs"].get(key, 0.0)
        if key not in base_counts:
            findings.append({
                "type": "unseen_transition", "from": int(a), "to": int(b),
                "window_count": c, "window_prob": wp, "baseline_prob": 0.0,
                "note": ("this mode change was not seen in the baseline — a new "
                         "way of sequencing operation, not necessarily a fault"),
            })
        elif base_probs.get(key, 0.0) < _RARE_P:
            findings.append({
                "type": "rare_transition", "from": int(a), "to": int(b),
                "window_count": c, "window_prob": wp,
                "baseline_prob": base_probs.get(key, 0.0),
                "note": ("a mode change that is unusual in the baseline is now "
                         "happening more — an observed sequencing change"),
            })
    # transitions COMMON in the baseline but missing now
    for key, bp in base_probs.items():
        if bp >= _COMMON_P and key not in win["counts"]:
            a, b = key.split("->", 1)
            findings.append({
                "type": "absent_transition", "from": int(a), "to": int(b),
                "window_count": 0, "window_prob": 0.0, "baseline_prob": bp,
                "note": ("a mode change that usually happens did not occur in this "
                         "window — a usual step is missing"),
            })
    # order: unseen first, then by how far from baseline expectation
    order = {"unseen_transition": 0, "absent_transition": 1, "rare_transition": 2}
    findings.sort(key=lambda f: (order.get(f["type"], 9),
                                 -abs(f["window_prob"] - f["baseline_prob"])))
    return {
        "window": win,
        "findings": findings,
        "confidence": conf,
        "n_baseline_transitions": n_base,
        "reliable": reliable,
    }
