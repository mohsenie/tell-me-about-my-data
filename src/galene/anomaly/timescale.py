"""Multi-timescale fingerprints — separate a SUDDEN break from SLOW drift.

The drift detector answers "does this window differ from the known-good baseline?"
It can't, by itself, tell WHEN or HOW the change arrived:
  - a SUDDEN break: one period looks abruptly different from the period right
    before it (a step change — often a discrete event: a fault, a config change);
  - SLOW drift: no single step is large, but the accumulation over many periods
    is (wear, fouling, seasonal shift). Each period looked fine next to its
    neighbour, yet the latest period is far from the original baseline.

Method (deterministic, timescale-agnostic, reuses the fingerprint machinery):
  - group the available dates into PERIODS at a chosen granularity
    (day / month / year — derived from the YYYY-MM-DD partition key, nothing
    hardcoded), preserving chronological order;
  - build ONE fingerprint per period, all ASSIGNED to the SAME persisted regime
    model, so periods are compared like-for-like (stable regime identity);
  - reduce each pair of fingerprints to a single scalar DISTANCE (mean absolute
    per-edge dcor change across shared regimes);
  - report, per period: distance to the PREVIOUS period (adjacent step) and
    distance to the FIRST/long baseline period (cumulative);
  - classify: a large adjacent step -> SUDDEN break at that period; a large
    cumulative distance with only small steps -> SLOW drift.

Discipline: reports the OBSERVED temporal pattern of change, never the cause.
Confidence scales with how many periods / samples define the comparison.
"""
from __future__ import annotations

import config
from galene.discovery.loader import load_numeric
from galene.discovery.relationships import graphs_for_fixed_regimes
from .baseline import _fingerprint_from_fixed, _edge_key


def period_key(date_str: str, granularity: str) -> str:
    """Period bucket for a YYYY-MM-DD date at the given granularity.
    day -> 'YYYY-MM-DD', month -> 'YYYY-MM', year -> 'YYYY'. Data-agnostic:
    parses the partition key, hardcodes no specific date."""
    parts = date_str.split("-")
    if granularity == "year":
        return parts[0]
    if granularity == "month":
        return "-".join(parts[:2])
    return date_str


def group_dates(dates: list[str], granularity: str) -> list[tuple[str, list[str]]]:
    """Group sorted dates into (period_key, [dates]) in chronological order."""
    groups: dict[str, list[str]] = {}
    order: list[str] = []
    for d in sorted(dates):
        k = period_key(d, granularity)
        if k not in groups:
            groups[k] = []
            order.append(k)
    for d in sorted(dates):
        groups[period_key(d, granularity)].append(d)
    return [(k, groups[k]) for k in order]


def _period_fingerprint(source, period_dates, vessel, model, with_mi=False):
    """Fingerprint for one period, assigned to the fixed regime model. None if the
    period has no/insufficient data."""
    globs = [g for g in config.source_globs_for_dates(source, period_dates, vessel)
             if _exists(g)]
    if not globs:
        return None
    cols, data = load_numeric(globs)
    if len(data) < 50:
        return None
    fixed = graphs_for_fixed_regimes(cols, data, model, with_mi=with_mi)
    return _fingerprint_from_fixed(fixed)


def fingerprint_distance(fp_a: dict, fp_b: dict) -> dict:
    """Scalar distance between two fingerprints: the mean absolute per-edge dcor
    change across regimes present in BOTH, plus the number of edges compared.
    Symmetric. Missing edges count as dcor 0 (appeared/disappeared)."""
    regimes = set(fp_a.get("regimes", {})) & set(fp_b.get("regimes", {}))
    deltas = []
    for lab in regimes:
        ea = fp_a["regimes"][lab].get("edges", {})
        eb = fp_b["regimes"][lab].get("edges", {})
        for ek in set(ea) | set(eb):
            deltas.append(abs(ea.get(ek, 0.0) - eb.get(ek, 0.0)))
    if not deltas:
        return {"distance": 0.0, "n_edges": 0}
    return {"distance": round(sum(deltas) / len(deltas), 4), "n_edges": len(deltas)}


# an adjacent-step distance this many times the median step is a SUDDEN break
_STEP_SPIKE_MULT = 2.5
# a cumulative (vs first period) distance this large flags drift overall
_CUMULATIVE_FLOOR = 0.08


def _classify(steps: list[float], cumulative: list[float]) -> dict:
    """Given adjacent-step distances and cumulative (vs first) distances, decide
    whether the change is a SUDDEN break, SLOW drift, both, or none."""
    if not steps:
        return {"pattern": "insufficient_history",
                "note": "need at least two periods to compare."}
    med = sorted(steps)[len(steps) // 2]
    spike_idx = [i for i, s in enumerate(steps)
                 if s > max(_STEP_SPIKE_MULT * med, _CUMULATIVE_FLOOR)]
    final_cumulative = cumulative[-1] if cumulative else 0.0
    sudden = len(spike_idx) > 0
    # slow drift = cumulative is meaningful but no single step spiked
    slow = (final_cumulative >= _CUMULATIVE_FLOOR) and not sudden
    if sudden and slow:
        pattern = "sudden_and_slow"
    elif sudden:
        pattern = "sudden_break"
    elif slow:
        pattern = "slow_drift"
    else:
        pattern = "stable"
    return {
        "pattern": pattern,
        "spike_period_indices": spike_idx,
        "median_step": round(med, 4),
        "final_cumulative": round(final_cumulative, 4),
        "note": {
            "sudden_break": "an abrupt change between two adjacent periods.",
            "slow_drift": "a gradual accumulation — no single step is large, but the "
                          "latest period is far from the earliest.",
            "sudden_and_slow": "both an abrupt step and gradual accumulation are present.",
            "stable": "no meaningful change across periods.",
        }.get(pattern, ""),
    }


def multiscale_drift(source: str, dates: list[str], vessel: str, model: dict,
                     granularity: str = "day", with_mi: bool = False) -> dict:
    """Build a per-period fingerprint series and separate SUDDEN vs SLOW change.

    Returns {granularity, periods:[{period, n_dates, step_distance, cumulative_
    distance}], classification:{pattern,...}}. `step_distance` is vs the previous
    period; `cumulative_distance` is vs the first period (the long baseline).
    Reports the observed temporal pattern, never the cause."""
    groups = group_dates(dates, granularity)
    fps = []
    for k, ds in groups:
        fp = _period_fingerprint(source, ds, vessel, model, with_mi=with_mi)
        if fp is not None:
            fps.append((k, ds, fp))
    if not fps:
        return {"error": f"no usable periods for '{source}' at {granularity} scale."}

    first_fp = fps[0][2]
    periods = []
    steps, cumulative = [], []
    prev_fp = None
    for k, ds, fp in fps:
        step = (fingerprint_distance(prev_fp, fp)["distance"]
                if prev_fp is not None else None)
        cum = fingerprint_distance(first_fp, fp)["distance"]
        periods.append({
            "period": k, "n_dates": len(ds),
            "step_distance": step, "cumulative_distance": cum,
        })
        if step is not None:
            steps.append(step)
        cumulative.append(cum)
        prev_fp = fp

    return {
        "source": source, "granularity": granularity,
        "n_periods": len(periods),
        "periods": periods,
        "classification": _classify(steps, cumulative),
    }


def _exists(glob_str: str) -> bool:
    import glob as _g
    return bool(_g.glob(glob_str))
