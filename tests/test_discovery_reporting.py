"""Functional tests for discovery internals + reporting: dependence.classify
thresholds, fusion shape, relationships (per-regime + fixed-regime graphs),
and the report dict structure.
"""
from __future__ import annotations

import glob

import numpy as np
import pytest

import config
from ttmd.discovery.dependence import classify, Kind, pearson_abs
from ttmd.discovery.loader import load_numeric
from ttmd.discovery.relationships import (
    build_per_regime_graphs, graphs_for_fixed_regimes, clean_frame)


# ---------------- dependence classification ----------------
def test_classify_thresholds():
    assert classify(0.9, 0.05) == Kind.NONE       # below dcor floor
    assert classify(0.9, 0.95) == Kind.LINEAR      # strong linear
    assert classify(0.1, 0.8) == Kind.NONLINEAR    # dcor >> pearson
    # weak/mixed defaults to linear (documented behavior)
    assert classify(0.3, 0.3) == Kind.LINEAR


def test_pearson_abs_bounds():
    x = np.arange(100.0)
    assert abs(pearson_abs(x, 2 * x + 1) - 1.0) < 1e-9   # perfect line
    assert pearson_abs(x, np.zeros(100)) == 0.0          # zero variance


# ---------------- relationships ----------------
def test_build_per_regime_graphs_shape(has_data):
    g = glob.glob(config.source_glob("engine"))
    cols, data = load_numeric(g)
    res = build_per_regime_graphs(cols, data, with_mi=False)
    assert res["regimes"]["k"] >= 1
    assert "global" in res["graphs"]
    for e in res["graphs"]["global"]["edges"]:
        assert set(("a", "b", "pearson", "dcor", "kind")).issubset(e)
        assert 0.0 <= e["dcor"] <= 1.0001
    # a regime model is surfaced for reuse
    assert "regime_model" in res


def test_graphs_for_fixed_regimes(has_data):
    g = glob.glob(config.source_glob("engine"))
    cols, data = load_numeric(g)
    res = build_per_regime_graphs(cols, data, with_mi=False)
    model = res.get("regime_model")
    if not model:
        pytest.skip("degenerate clustering")
    fixed = graphs_for_fixed_regimes(cols, data, model, with_mi=False)
    assert "graphs" in fixed and "fractions" in fixed
    # fractions sum to ~1 across assigned regimes
    total = sum(fixed["fractions"].values())
    assert 0.9 <= total <= 1.0001


# ---------------- fusion ----------------
def test_fuse_sources_shape(has_data):
    from ttmd.discovery.fusion import fuse_sources
    srcs = [s for s in config.discover_sources()][:3]
    if len(srcs) < 2:
        pytest.skip("need >=2 sources")
    globs = {s: config.source_glob(s) for s in srcs}
    cols, data = fuse_sources(globs)
    assert len(cols) > 0
    assert data.shape[1] == len(cols)
    # fused columns are namespaced source__signal
    assert any("__" in c for c in cols)


# ---------------- reporting ----------------
def test_describe_discovery_structure(has_data):
    from ttmd.reporting import describe_discovery
    g = glob.glob(config.source_glob("engine"))
    cols, data = load_numeric(g)
    disc = build_per_regime_graphs(cols, data, with_mi=False)
    report = describe_discovery(disc, "engine")
    assert "relationships" in report
    rel = report["relationships"]
    assert "total" in rel and "strongest" in rel
    assert isinstance(rel["strongest"], list)


def test_render_markdown_nonempty(has_data):
    from ttmd.reporting import describe_discovery, render_markdown
    g = glob.glob(config.source_glob("engine"))
    cols, data = load_numeric(g)
    disc = build_per_regime_graphs(cols, data, with_mi=False)
    md = render_markdown(describe_discovery(disc, "engine"))
    assert isinstance(md, str) and len(md) > 50 and "##" in md
