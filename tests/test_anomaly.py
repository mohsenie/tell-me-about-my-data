"""Functional tests for the anomaly layer: baseline build/roundtrip, drift
self-consistency, behavioral-norm detection, regime-model persist + assign.
Deterministic; no LLM.
"""
from __future__ import annotations

import pytest

import config
from ttmd.anomaly import (
    build_baseline, detect_drift, detect_stops, dwell_norm, flag_current_dwell)

# Baseline-building runs discovery repeatedly -> seconds each. Mark slow so
# `pytest -m "not slow"` skips them for fast targeted runs.
pytestmark_slow = pytest.mark.slow


def _first_multiday_source():
    for s in config.discover_sources():
        if len(config.available_dates(s)) >= 2:
            return s
    return None


@pytest.mark.slow
def test_baseline_build_has_fingerprint_and_model(has_data):
    src = _first_multiday_source()
    if not src:
        import pytest; pytest.skip("no multi-day source")
    dates = config.available_dates(src)
    bl = build_baseline(src, dates, config.DEFAULT_VESSEL, with_mi=False)
    assert bl and bl["source"] == src
    assert bl["fingerprint"]["regimes"]           # per-regime edge maps present
    assert "variation" in bl                       # normal-variation band
    # regime model persisted (columns + scaler + centers) for stable reuse
    m = bl.get("regime_model")
    assert m and m["columns"] and m["centers"] and m["scaler_mean"]


@pytest.mark.slow
def test_drift_self_consistency_zero(has_data):
    """A window compared to its OWN baseline shows no structure drift."""
    src = _first_multiday_source()
    if not src:
        import pytest; pytest.skip("no multi-day source")
    dates = config.available_dates(src)
    bl = build_baseline(src, dates, config.DEFAULT_VESSEL, with_mi=False)
    res = detect_drift(src, dates, config.DEFAULT_VESSEL, bl, with_mi=False)
    assert res.get("layer1_structure_drift", []) == []
    assert res["assignment"].startswith("fixed-model")  # used the persisted model


@pytest.mark.slow
def test_drift_report_structure(has_data):
    src = "engine" if "engine" in config.discover_sources() else _first_multiday_source()
    if not src:
        import pytest; pytest.skip("no source")
    dates = config.available_dates(src)
    if len(dates) < 3:
        import pytest; pytest.skip("need >=3 days to split baseline/window")
    bl = build_baseline(src, dates[:len(dates) // 2], config.DEFAULT_VESSEL, with_mi=False)
    res = detect_drift(src, dates[len(dates) // 2:], config.DEFAULT_VESSEL, bl,
                       with_mi=False)
    assert "layer1_structure_drift" in res and "layer2_regime_events" in res
    # each layer-1 finding is well-formed
    for f in res["layer1_structure_drift"]:
        assert "regime" in f and "changed_edges" in f and f["confidence"] in (
            "high", "medium", "low")
        for c in f["changed_edges"]:
            assert c["change"] in ("strengthened", "weakened", "appeared",
                                   "disappeared")


# ---------------- behavioral ----------------
def test_detect_stops_wellformed(pos_source, globs):
    src, lat, lon = pos_source
    stops = detect_stops(globs(src), lat, lon)
    for s in stops:
        assert s["duration_s"] > 0 and s["t_end"] >= s["t_start"]
    # stops are time-ordered
    for i in range(1, len(stops)):
        assert stops[i]["t_start"] >= stops[i - 1]["t_start"]


def test_dwell_norm_and_flag(pos_source, globs):
    src, lat, lon = pos_source
    stops = detect_stops(globs(src), lat, lon)
    norm = dwell_norm(stops)
    if norm is None:
        import pytest; pytest.skip("not enough stops for a norm")
    assert norm["median_h"] >= 0 and norm["p90_h"] >= norm["median_h"]
    flag = flag_current_dwell(stops)
    # flag is either None or a well-formed finding
    if flag:
        assert flag["current_h"] > 0 and flag["ratio_vs_median"] > 0
        assert flag["confidence"] in ("high", "medium", "low")


def test_assign_labels_matches_training(has_data):
    """assign_labels on the training columns reproduces cluster membership shape."""
    from ttmd.discovery.regimes import segment_regimes, assign_labels
    from ttmd.discovery.loader import load_numeric
    from ttmd.discovery.relationships import clean_frame
    src = _first_multiday_source()
    if not src:
        import pytest; pytest.skip("no source")
    import glob
    g = glob.glob(config.source_glob(src))
    cols, data = load_numeric(g)
    cols, clean = clean_frame(cols, data)
    if not cols or len(clean) < 100:
        import pytest; pytest.skip("insufficient data")
    reg = segment_regimes(cols, clean)
    model = reg.model_params()
    if model is None:
        import pytest; pytest.skip("degenerate clustering")
    labels = assign_labels(model, cols, clean[:2000])
    assert len(labels) == len(clean[:2000])
    assert set(labels).issubset(set(range(len(model["centers"]))))


# ---------------- regime transitions (Layer-2 temporal) ----------------
def test_runs_from_labels_collapses_consecutive():
    from ttmd.anomaly import runs_from_labels
    assert runs_from_labels([0, 0, 0, 1, 1, 2, 2, 2, 0]) == [0, 1, 2, 0]
    assert runs_from_labels([]) == []
    assert runs_from_labels([5, 5, 5]) == [5]


def test_transition_matrix_and_self_consistency():
    """A window with the SAME sequence pattern as the baseline flags nothing."""
    from ttmd.anomaly import build_transition_matrix, detect_transition_anomalies
    btm = build_transition_matrix([0, 1, 2] * 20, k=3)
    assert btm["n_transitions"] == 59
    assert btm["probs"]["0->1"] == 1.0
    same = detect_transition_anomalies([0, 1, 2] * 5, btm)
    assert same["findings"] == []
    assert same["reliable"] is True


def test_transition_detects_unseen_and_absent():
    """A new mode-jump is 'unseen'; usual steps that vanish are 'absent'."""
    from ttmd.anomaly import build_transition_matrix, detect_transition_anomalies
    btm = build_transition_matrix([0, 1, 2] * 20, k=3)
    res = detect_transition_anomalies([0, 2, 0, 2, 0, 2], btm)
    types = {(f["type"], f["from"], f["to"]) for f in res["findings"]}
    assert ("unseen_transition", 0, 2) in types
    assert ("absent_transition", 0, 1) in types


@pytest.mark.slow
def test_baseline_persists_transitions_and_detector_surfaces(has_data):
    """End-to-end: baseline persists a transition matrix; detect_drift surfaces
    a layer2_transition_events block on a later window."""
    src = "engine" if "engine" in config.discover_sources() else _first_multiday_source()
    if not src:
        pytest.skip("no source")
    dates = config.available_dates(src)
    if len(dates) < 3:
        pytest.skip("need >=3 days")
    bl = build_baseline(src, dates[:2], config.DEFAULT_VESSEL, with_mi=False)
    if "regime_model" not in bl:
        pytest.skip("degenerate clustering (no model)")
    assert "transitions" in bl
    tm = bl["transitions"]
    assert tm["n_transitions"] >= 0 and "probs" in tm
    res = detect_drift(src, dates[2:], config.DEFAULT_VESSEL, bl, with_mi=False)
    te = res.get("layer2_transition_events")
    assert te is not None
    assert "findings" in te and "confidence" in te and "reliable" in te
