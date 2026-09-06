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


# ---------------- timestamp / label-sequence plumbing ----------------
def test_load_numeric_with_time_aligned_sorted(has_data):
    """load_numeric_with_time returns row-aligned, time-ordered timestamps."""
    from ttmd.discovery.loader import load_numeric_with_time
    g = glob.glob(config.source_glob("engine"))
    cols, data, ts = load_numeric_with_time(g)
    assert data.shape[0] == ts.shape[0]          # row-aligned
    assert data.shape[1] == len(cols)
    assert (ts[1:] >= ts[:-1]).all()             # chronological


def test_label_sequence_time_ordered(has_data):
    """label_sequence assigns a per-row regime label aligned with sorted times."""
    from ttmd.discovery.loader import load_numeric, load_numeric_with_time
    from ttmd.discovery.regimes import segment_regimes, label_sequence
    g = glob.glob(config.source_glob("engine"))
    cols, data, ts = load_numeric_with_time(g)
    c2, d2 = load_numeric(g)
    cc, cd = clean_frame(c2, d2)
    model = segment_regimes(cc, cd).model_params()
    labels, times = label_sequence(model, cols, data, ts)
    assert labels.shape[0] == times.shape[0]     # aligned
    assert (times[1:] >= times[:-1]).all()       # chronological
    # every label is a valid regime index
    assert set(labels.tolist()).issubset(set(range(len(model["centers"]))))


# ---------------- partial / conditional dependence ----------------
def test_partial_correlation_prunes_common_driver():
    """A common driver C->A, C->B makes A~B strong marginally but its PARTIAL
    correlation collapses -> flagged as induced (direct=False)."""
    from ttmd.discovery.dependence import pairwise_dependence
    rng = np.random.default_rng(0)
    C = rng.normal(size=4000)
    A = C + rng.normal(0, 0.3, size=4000)
    B = C + rng.normal(0, 0.3, size=4000)
    D = rng.normal(size=4000)
    data = np.column_stack([A, B, C, D])
    res = {(r.a, r.b): r for r in pairwise_dependence(["A", "B", "C", "D"], data,
                                                      with_mi=False)}
    ab = res[("A", "B")]
    assert ab.pearson > 0.8 and ab.partial < 0.2 and ab.direct is False   # induced
    ac = res[("A", "C")]
    assert ac.partial > 0.3 and ac.direct is True                          # real link


def test_partial_correlation_matrix_shape_and_none():
    from ttmd.discovery.dependence import partial_correlation_matrix
    rng = np.random.default_rng(1)
    data = rng.normal(size=(500, 4))
    m = partial_correlation_matrix(data)
    assert m is not None and m.shape == (4, 4)
    assert (m >= 0).all() and (m <= 1).all()
    # too few columns / rows -> None
    assert partial_correlation_matrix(rng.normal(size=(500, 2))) is None
    assert partial_correlation_matrix(rng.normal(size=(3, 5))) is None


def test_induced_edge_phrasing():
    """An induced edge is described as INDIRECT, not a direct relationship."""
    from ttmd.reporting import phrasing as ph
    edge = {"a": "boost_pressure", "b": "oil_pressure", "dcor": 0.8,
            "pearson": 0.81, "kind": "linear", "direct": False, "partial": 0.004}
    txt = ph.describe_edge(edge, False)
    assert "INDIRECT" in txt and "boost" in txt.lower()


# ---------------- block-permutation significance ----------------
def test_block_permutation_flags_autocorrelation_phantom():
    """Two INDEPENDENT random walks look dependent (shared autocorrelation), but a
    time-aware block-permutation null does NOT call it strongly significant."""
    from ttmd.discovery.dependence import (block_permutation_pvalue,
                                           distance_correlation)
    rng = np.random.default_rng(0)
    a = np.cumsum(rng.normal(size=2000))
    b = np.cumsum(rng.normal(size=2000))          # independent walk
    assert distance_correlation(a, b) > 0.2       # phantom looks dependent
    p = block_permutation_pvalue(a, b, n_perm=99, n_blocks=10)
    assert p > 0.05                               # not significant -> flagged phantom


def test_block_permutation_short_series_pvalue_one():
    from ttmd.discovery.dependence import block_permutation_pvalue
    assert block_permutation_pvalue(np.arange(5.0), np.arange(5.0)) == 1.0


def test_pairwise_with_significance_annotates_edges():
    """with_significance adds pvalue + significant to candidate edges."""
    from ttmd.discovery.dependence import pairwise_dependence
    rng = np.random.default_rng(0)
    x = rng.normal(size=800)
    data = np.column_stack([x, 2 * x + rng.normal(0, 0.1, size=800),
                            rng.normal(size=800)])
    res = pairwise_dependence(["x", "y", "z"], data, with_mi=False,
                              with_significance=True, n_perm=49)
    strong = next(r for r in res if {r.a, r.b} == {"x", "y"})
    assert strong.pvalue is not None and strong.significant is True   # real link


# ---------------- fused clustering blur diagnostic ----------------
def test_fused_relationship_graph_reports_quality(has_data):
    from ttmd.discovery.fusion import fused_relationship_graph
    srcs = [s for s in config.discover_sources()][:3]
    if len(srcs) < 2:
        pytest.skip("need >=2 sources")
    globs = {s: config.source_glob(s) for s in srcs}
    res = fused_relationship_graph(globs, with_mi=False)
    assert "regime_quality" in res and "cross_source_edges" in res
    q = res["regime_quality"]
    assert "silhouette" in q and "blurred" in q
    # every cross-source edge really spans two different sources
    for e in res["cross_source_edges"][:20]:
        assert e["a"].split("__")[0] != e["b"].split("__")[0]
