"""Behavioral-norm detection — is the asset OPERATING the way it usually does?

Complements the relational drift detector (which watches sensor-coupling
structure). This layer watches interpretable BEHAVIORS derived from the position
track and flags departures from the asset's own historical norm, in plain terms:
  "the ship has stayed in its current location ~19h — about 3x its usual stay."

First behavior: DWELL TIME at a location (how long the asset stays near-stationary
in one place). More behaviors (daily distance, speed profile) fit the same shape:
derive a behavior -> learn its normal distribution from history -> flag outliers.

Discipline (invariants): report the OBSERVED behavior change, never assert the
cause (broke down? waiting? weather? — the data can't say). Confidence scales with
how many historical samples define "normal".
"""
from __future__ import annotations

import math
import statistics

import duckdb


def _src(globs):
    arr = ", ".join("'" + g + "'" for g in globs)
    return f"read_parquet([{arr}], union_by_name=true)"


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * r * math.asin(math.sqrt(a))


def detect_stops(globs, lat_col, lon_col, move_km=1.0, min_stop_s=1800):
    """Segment the position track into STOPS (near-stationary periods) and MOVES.

    Walk the track in time order; accumulate a 'stop' while consecutive fixes stay
    within `move_km` of the stop's anchor. When the asset moves beyond that, the
    stop ends. Returns the list of stops that lasted >= min_stop_s:
      [{lat, lon, t_start, t_end, duration_s}], most recent last.

    Deterministic, column names are parameters (role-resolved by the caller)."""
    con = duckdb.connect()
    rows = con.execute(
        f'SELECT timestamp, "{lat_col}", "{lon_col}" FROM {_src(globs)} '
        f'WHERE "{lat_col}" IS NOT NULL AND "{lon_col}" IS NOT NULL '
        f'ORDER BY timestamp'
    ).fetchall()
    con.close()
    if len(rows) < 2:
        return []

    stops = []
    anchor_lat, anchor_lon = float(rows[0][1]), float(rows[0][2])
    stop_start_t = float(rows[0][0])
    last_t = float(rows[0][0])
    for ts, la, lo in rows[1:]:
        ts = float(ts); la = float(la); lo = float(lo)
        if _haversine_km(anchor_lat, anchor_lon, la, lo) <= move_km:
            # still within the stop radius of the current anchor
            last_t = ts
            continue
        # moved away -> close the current stop if it was long enough
        dur = last_t - stop_start_t
        if dur >= min_stop_s:
            stops.append({"lat": anchor_lat, "lon": anchor_lon,
                          "t_start": stop_start_t, "t_end": last_t,
                          "duration_s": dur})
        # start a new potential stop anchored here
        anchor_lat, anchor_lon = la, lo
        stop_start_t = ts
        last_t = ts
    # trailing stop (the CURRENT one, if the asset ended near-stationary)
    dur = last_t - stop_start_t
    if dur >= min_stop_s:
        stops.append({"lat": anchor_lat, "lon": anchor_lon,
                      "t_start": stop_start_t, "t_end": last_t,
                      "duration_s": dur, "ongoing": True})
    return stops


def dwell_norm(stops):
    """Normal dwell-time statistics from historical stops (excluding the most
    recent, which is what we test). Returns {median_h, p90_h, n} or None."""
    if len(stops) < 3:
        return None
    hist = stops[:-1]           # all but the most recent
    durs_h = [s["duration_s"] / 3600.0 for s in hist]
    return {
        "median_h": round(statistics.median(durs_h), 2),
        "p90_h": round(_percentile(durs_h, 90), 2),
        "max_h": round(max(durs_h), 2),
        "n": len(hist),
    }


def flag_current_dwell(stops):
    """Compare the MOST RECENT stop's dwell time to the historical norm. Returns a
    finding dict if it's notably longer than usual, else None.

    'Notably longer' = beyond the historical p90 AND >= 1.5x the median (so a
    normal long-ish stop doesn't over-alert). Confidence by history size."""
    norm = dwell_norm(stops)
    if not norm:
        return None
    current = stops[-1]
    cur_h = current["duration_s"] / 3600.0
    if cur_h <= norm["p90_h"] or cur_h < 1.5 * max(norm["median_h"], 1e-6):
        return None
    ratio = cur_h / norm["median_h"] if norm["median_h"] else float("inf")
    conf = "high" if norm["n"] >= 10 else ("medium" if norm["n"] >= 5 else "low")
    return {
        "behavior": "dwell_time",
        "current_h": round(cur_h, 1),
        "usual_median_h": norm["median_h"],
        "usual_p90_h": norm["p90_h"],
        "ratio_vs_median": round(ratio, 1),
        "ongoing": bool(current.get("ongoing")),
        "lat": round(current["lat"], 4), "lon": round(current["lon"], 4),
        "n_history": norm["n"],
        "confidence": conf,
    }


def _percentile(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(math.floor(k)); hi = int(math.ceil(k))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)
