"""Functional tests for the deterministic query layer: compute, capabilities,
timeparse, spatial. Assert OUTCOMES (values, structure) on the real vessel data.
No LLM.
"""
from __future__ import annotations

import datetime as dt
import calendar

import pytest

import config
from galene.query import (
    aggregate, list_capabilities, describe_capabilities,
    current_position, ships_nearby, voyage_window, track_distance_km,
    distance_segments, entity_position_at, haversine_km, signal_by_location,
    nearest_place, place_label, detect_legs)
from galene.query.timeparse import parse_instant


# ---------------- compute ----------------
def test_aggregate_avg_and_count(globs, has_data):
    g = globs("engine")
    res = aggregate(g, "EngineSpeed", "avg", "win")
    assert res.value is not None and res.value > 0
    assert res.n_rows > 0
    assert "EngineSpeed" in res.human()


def test_aggregate_max_ge_avg(globs, has_data):
    g = globs("engine")
    avg = aggregate(g, "EngineSpeed", "avg", "win").value
    mx = aggregate(g, "EngineSpeed", "max", "win").value
    assert mx >= avg


def test_integral_is_positive_and_differs_from_avg(globs, has_data):
    g = globs("engine")
    integ = aggregate(g, "EngineFuelRate", "integral", "win").value
    avg = aggregate(g, "EngineFuelRate", "avg", "win").value
    assert integ is not None and integ > 0
    # a time-integral of a rate is a total, not the mean rate
    assert abs(integ - avg) > 1e-6


def test_aggregate_t_range_subsets(globs, has_data):
    g = globs("engine")
    full = aggregate(g, "EngineSpeed", "count", "win").n_rows
    # restrict to a 1-hour window somewhere inside the data
    import duckdb
    arr = ", ".join("'" + x + "'" for x in g)
    lo, hi = duckdb.connect().execute(
        f"SELECT min(timestamp), max(timestamp) FROM read_parquet([{arr}], union_by_name=true)"
    ).fetchone()
    # a middle third of the range must contain some (but not all) rows
    q1, q3 = lo + (hi - lo) / 3, lo + 2 * (hi - lo) / 3
    sub = aggregate(g, "EngineSpeed", "count", "win", t_range=(q1, q3)).n_rows
    assert 0 < sub < full


# ---------------- capabilities ----------------
def test_capabilities_lists_signals_and_excludes_constants(has_data):
    caps = list_capabilities({"engine": config.source_glob("engine")})
    info = caps["engine"]
    names = {s["name"] for s in info["signals"]}
    assert "EngineSpeed" in names
    # every queryable signal reports a min/max range
    for s in info["signals"]:
        assert s["min"] is not None and s["max"] is not None and s["min"] != s["max"]
    # excluded ones carry a reason
    for e in info.get("excluded", []):
        assert e["reason"]
    text = describe_capabilities(caps)
    assert "engine" in text and "queryable" in text


def test_capabilities_examples_use_real_signals(has_data):
    """Example requests are built from the source's ACTUAL signals — not hardcoded
    engine/fuel names. A vibration query must not mention fuel/engine/coolant."""
    from galene.query.capabilities import _example_requests
    caps = list_capabilities({"vibration": config.source_glob("vibration")})
    real = {s["name"] for s in caps["vibration"]["signals"]}
    ex = _example_requests(caps)
    assert ex, "expected example requests"
    # every example references a real vibration signal, none the old hardcoded ones
    for e in ex:
        assert any(sig in e for sig in real)
    text = describe_capabilities(caps).lower()
    for bad in ("fuel consumption", "engine speed", "coolant temperature"):
        assert bad not in text
    # empty caps -> no examples, no crash
    assert _example_requests({}) == []


# ---------------- timeparse ----------------
def _win():
    lo = float(calendar.timegm(dt.datetime(2026, 9, 1, 0, 0).timetuple()))
    hi = float(calendar.timegm(dt.datetime(2026, 9, 5, 12, 2).timetuple()))
    return lo, hi


@pytest.mark.parametrize("phrase,expected", [
    ("on 2026-09-02 at 19:00", "2026-09-02 19:00"),
    ("2026-09-02", "2026-09-02 12:00"),
    ("on 2026-09-03 at 9am", "2026-09-03 09:00"),
    ("yesterday at 15:00", "2026-09-04 15:00"),
    ("2 days ago at noon", "2026-09-03 12:00"),
    ("frequency of x axis", None),
])
def test_parse_instant(phrase, expected):
    lo, hi = _win()
    e = parse_instant(phrase, lo, hi)
    got = dt.datetime.utcfromtimestamp(e).strftime("%Y-%m-%d %H:%M") if e else None
    assert got == expected


# ---------------- spatial ----------------
def test_current_position(pos_source, globs):
    src, lat, lon = pos_source
    pos = current_position(globs(src), lat, lon)
    assert pos and "lat" in pos and "lon" in pos and "timestamp" in pos


def test_track_distance_positive(pos_source, globs):
    src, lat, lon = pos_source
    km = track_distance_km(globs(src), lat, lon)
    assert km > 0


def test_distance_segments_cover_track(pos_source, globs):
    src, lat, lon = pos_source
    total = track_distance_km(globs(src), lat, lon)
    segs = distance_segments(globs(src), lat, lon, 20.0)
    assert len(segs) >= 1
    # segments are time-ordered with positive durations
    for i, s in enumerate(segs):
        assert s["t_end"] >= s["t_start"]
        if i:
            assert s["t_start"] >= segs[i - 1]["t_start"]
    # summed segment spans approximate the total track (last may be partial)
    covered = segs[-1]["km_end"]
    assert covered <= total + 1.0


def test_voyage_window_directional(pos_source, globs):
    src, lat, lon = pos_source
    g = globs(src)
    import duckdb
    arr = ", ".join("'" + x + "'" for x in g)
    s = duckdb.connect().execute(
        f'SELECT "{lat}","{lon}" FROM read_parquet([{arr}], union_by_name=true) '
        f'WHERE "{lat}" IS NOT NULL ORDER BY timestamp ASC LIMIT 1').fetchone()
    e = duckdb.connect().execute(
        f'SELECT "{lat}","{lon}" FROM read_parquet([{arr}], union_by_name=true) '
        f'WHERE "{lat}" IS NOT NULL ORDER BY timestamp DESC LIMIT 1').fetchone()
    win = voyage_window(g, lat, lon, s[0], s[1], e[0], e[1])
    assert win and win["t_end"] >= win["t_start"] and win["track_km"] > 0


# ---------------- reverse geocoding (offline port list) ----------------
_PLACES = [
    {"name": "Inverness Marina", "lat": 57.4870, "lon": -4.2530, "radius_km": 5},
    {"name": "Fort William", "lat": 56.8198, "lon": -5.1052, "radius_km": 5},
]


def test_nearest_place_within_radius():
    hit = nearest_place(57.487, -4.253, _PLACES)
    assert hit is not None and hit["name"] == "Inverness Marina"
    assert hit["distance_km"] < 0.1


def test_nearest_place_open_water_none():
    assert nearest_place(50.0, -10.0, _PLACES) is None    # nothing in range


def test_nearest_place_picks_closest():
    # near Fort William, far from Inverness -> Fort William wins
    hit = nearest_place(56.82, -5.10, _PLACES)
    assert hit["name"] == "Fort William"


def test_place_label_named_vs_raw():
    assert "Inverness Marina" in place_label(57.487, -4.253, _PLACES)
    # off the list -> raw coordinates, no name
    lbl = place_label(50.0, -10.0, _PLACES)
    assert "Marina" not in lbl and "50.0" in lbl


def test_known_places_loads_from_yaml(has_data):
    places = config.known_places()
    # sample vessel ships a places list; names are present with coords
    if not places:
        pytest.skip("no places configured for this vessel")
    assert all("name" in p and "lat" in p and "lon" in p for p in places)


# ---------------- voyage / leg auto-detection ----------------
def test_detect_legs_wellformed(pos_source, globs):
    """Auto-detected legs are chronological, positive-distance, place-labeled."""
    src, lat, lon = pos_source
    legs = detect_legs(globs(src), lat, lon, places=config.known_places())
    if not legs:
        pytest.skip("no distinct legs on this track (needs >=2 stops)")
    for lg in legs:
        assert lg["t_end"] > lg["t_start"]           # forward in time
        assert lg["distance_km"] > 0                 # actually moved
        assert isinstance(lg["from"], str) and isinstance(lg["to"], str)
    # most recent last (non-decreasing start times)
    starts = [lg["t_start"] for lg in legs]
    assert starts == sorted(starts)


def test_is_last_trip_guard():
    from galene.chat_deps import ChatDeps
    assert ChatDeps._is_last_trip("how much fuel on the last voyage")
    assert ChatDeps._is_last_trip("the most recent trip")
    assert not ChatDeps._is_last_trip("average engine speed")
    assert not ChatDeps._is_last_trip("fuel from 54.5,18 to 57,-4")


def test_list_voyages_capability(deps, has_data):
    """list_voyages returns either detected legs or a clear 'no distinct voyages'
    message — never an error."""
    out = deps.list_voyages("engine", "list my voyages")
    assert isinstance(out, str) and len(out) > 0
    assert ("voyage leg" in out.lower()) or ("couldn't identify" in out.lower())


def test_haversine_km():
    # ~1 degree of latitude is ~111 km
    d = haversine_km(57.0, -4.0, 58.0, -4.0)
    assert 105 < d < 115


def test_ships_nearby_shape(deps, has_data, globs):
    # find a multi-entity source (lat/lon + id)
    multi = next((s for s in deps.all_sources()
                  if all(deps._latlon_cols(s)) and deps._id_col(s)), None)
    if not multi:
        pytest.skip("no multi-entity source")
    lat, lon = deps._latlon_cols(multi)
    idc = deps._id_col(multi)
    g = globs(multi)
    tmin, tmax = deps._time_range(g)
    # use own position at tmax as the reference point
    own = deps._own_position_source()
    ol, oo = deps._latlon_cols(own)
    op = current_position(globs(own), ol, oo)
    found = ships_nearby(g, tmax, op["lat"], op["lon"], lat, lon, idc,
                         radius_km=100.0, name_col=deps._name_col(multi))
    assert isinstance(found, list)
    for f in found:
        assert "id" in f and "distance_km" in f and f["distance_km"] <= 100.0


def test_signal_by_location_cells(deps, pos_source, globs):
    src, lat, lon = pos_source
    vib = next((s for s in deps.all_sources()
                if deps.signal_family(s, "vibration")[0]), None)
    if not vib:
        pytest.skip("no vibration-described source")
    _, mag = deps.signal_family(vib, "vibration")
    cells, is_mag = signal_by_location(globs(src), lat, lon, globs(vib), mag,
                                       cell_deg=0.2)
    assert len(cells) >= 1
    for c in cells:
        assert c["n"] > 0 and "value" in c
