"""Functional tests for plots: each produces a valid PNG and correct metadata.
We assert the file exists + is a PNG + the returned dict is right — not pixels.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from galene.query.plots import (
    scatter, aggregated_series, timeseries, distance_series)


def _is_png(path):
    p = Path(path)
    return p.exists() and p.stat().st_size > 0 and p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_scatter(globs, has_data, tmp_path):
    g = globs("engine")
    out = tmp_path / "scatter.png"
    info = scatter(g, "EngineSpeed", "EngineFuelRate", out)
    assert _is_png(out)
    assert info["x"] == "EngineSpeed" and info["y"] == "EngineFuelRate"
    assert -1.0 <= info["pearson"] <= 1.0
    assert info["points_total"] > 0


def test_aggregated_series_trend(globs, has_data, tmp_path):
    g = globs("engine")
    out = tmp_path / "trend.png"
    info = aggregated_series(g, "EngineSpeed", out, bucket="4h", agg="avg")
    assert _is_png(out)
    assert info["signal"] == "EngineSpeed" and info["agg"] == "avg"
    assert info["bins"] >= 1


def test_aggregated_series_t_range_fewer_bins(globs, has_data, tmp_path):
    g = globs("engine")
    full = aggregated_series(g, "EngineSpeed", tmp_path / "a.png", bucket="1h")["bins"]
    import duckdb
    arr = ", ".join("'" + x + "'" for x in g)
    lo, hi = duckdb.connect().execute(
        f"SELECT min(timestamp),max(timestamp) FROM read_parquet([{arr}],union_by_name=true)"
    ).fetchone()
    q1, q3 = lo + (hi - lo) / 3, lo + 2 * (hi - lo) / 3
    sub = aggregated_series(g, "EngineSpeed", tmp_path / "b.png", bucket="1h",
                            t_range=(q1, q3))["bins"]
    assert 1 <= sub < full


def test_timeseries(globs, has_data, tmp_path):
    g = globs("engine")
    out = tmp_path / "ts.png"
    info = timeseries(g, ["EngineSpeed", "EngineFuelRate"], out)
    assert _is_png(out)
    assert info["signals"] == ["EngineSpeed", "EngineFuelRate"]
    assert info["points"] > 0


def test_distance_series(tmp_path):
    # pure rendering from pre-computed per-segment values
    segs = [{"km_start": 0, "km_end": 20, "value": 100.0},
            {"km_start": 20, "km_end": 40, "value": 120.0},
            {"km_start": 40, "km_end": 55, "value": 60.0}]
    out = tmp_path / "dist.png"
    info = distance_series(segs, out, "EngineFuelRate", unit="L", seg_km=20)
    assert _is_png(out)
    assert info["segments"] == 3 and info["seg_km"] == 20 and info["unit"] == "L"


def test_scatter_no_data_raises(globs, has_data, tmp_path):
    with pytest.raises(Exception):
        scatter(globs("engine"), "NoSuchSignalX", "NoSuchSignalY", tmp_path / "x.png")
