"""On-demand plots that VISUALIZE relationships/values in the data.

Deterministic: reads the real paired data and renders a chart (PNG). Lets a user
SEE a correlation the system reported ("RPM ~ fuel = 0.99") as an actual scatter
of point-on-point readings, not just a number.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np


def _src(globs: list[str]) -> str:
    arr = ", ".join("'" + g + "'" for g in globs)
    return f"read_parquet([{arr}], union_by_name=true)"


def scatter(globs: list[str], x_signal: str, y_signal: str, out_path: str | Path,
            max_points: int = 5000, title: str | None = None) -> dict:
    """Scatter of y vs x (each point = one timestamp's paired reading), with the
    measured Pearson r and a fitted line. Returns a small summary dict.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    con = duckdb.connect()
    df = con.execute(
        f'SELECT "{x_signal}" AS x, "{y_signal}" AS y FROM {_src(globs)} '
        f'WHERE "{x_signal}" IS NOT NULL AND "{y_signal}" IS NOT NULL'
    ).df()
    con.close()
    if df.empty:
        raise ValueError(f"no paired data for {x_signal} vs {y_signal}")

    # subsample for a readable/plottable scatter
    n_total = len(df)
    if n_total > max_points:
        df = df.sample(max_points, random_state=0)
    x = df["x"].to_numpy(float); y = df["y"].to_numpy(float)

    r = float(np.corrcoef(x, y)[0, 1]) if x.std() and y.std() else 0.0

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, s=4, alpha=0.25, color="#0366d6")
    # fitted line
    if x.std() > 0:
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, m * xs + b, color="#d73a49", lw=1.5,
                label=f"fit: y={m:.3g}x+{b:.3g}")
        ax.legend(loc="best", fontsize=9)
    ax.set_xlabel(x_signal); ax.set_ylabel(y_signal)
    ax.set_title(title or f"{y_signal} vs {x_signal}   (Pearson r={r:.2f}, "
                 f"n={n_total:,})")
    ax.grid(alpha=0.2)
    out_path = Path(out_path)
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)

    return {"x": x_signal, "y": y_signal, "pearson": round(r, 4),
            "points_plotted": len(df), "points_total": n_total,
            "path": str(out_path)}


def _bucket_seconds(bucket) -> tuple[int, str]:
    """Resolve a bucket spec to seconds + a label.

    Accepts named ('minute'/'hour'/'day') OR arbitrary intervals: an int (hours)
    or a string like '4h', '30m', '2d', '90s'.
    """
    if isinstance(bucket, (int, float)):
        return int(bucket) * 3600, f"{int(bucket)}h"
    b = str(bucket).strip().lower()
    named = {"minute": (60, "minute"), "hour": (3600, "hour"), "day": (86400, "day")}
    if b in named:
        return named[b]
    import re
    m = re.fullmatch(r"(\d+)\s*([smhd])", b)
    if m:
        n = int(m.group(1)); unit = m.group(2)
        mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
        return n * mult, f"{n}{unit}"
    return 3600, "hour"  # fallback


def aggregated_series(globs: list[str], signal: str, out_path: str | Path,
                      bucket: str = "hour", agg: str = "avg",
                      title: str | None = None,
                      t_range: tuple[float, float] | None = None) -> dict:
    """Plot an aggregate of a signal per time bucket.

    bucket: named ('hour'/'day'/'minute') or arbitrary ('4h', '30m', '2d').
    agg: avg/min/max/median/sum. E.g. "average temperature, binned every 4 hours".
    t_range: optional (start_epoch, end_epoch) to restrict to a segment/voyage.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seconds, bucket_label = _bucket_seconds(bucket)
    fn = {"avg": "avg", "min": "min", "max": "max", "median": "median",
          "sum": "sum"}.get(agg, "avg")
    where = f'"{signal}" IS NOT NULL'
    if t_range:
        lo, hi = sorted(t_range)
        where += f" AND timestamp BETWEEN {lo} AND {hi}"
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT CAST(timestamp/{seconds} AS BIGINT)*{seconds} AS bucket_t,
               {fn}("{signal}") AS v
        FROM {_src(globs)} WHERE {where}
        GROUP BY bucket_t ORDER BY bucket_t
    """).df()
    con.close()
    if df.empty:
        raise ValueError(f"no data for {signal}")

    # x axis: hours from start (readable)
    t0 = df["bucket_t"].iloc[0]
    hrs = (df["bucket_t"].to_numpy(float) - t0) / 3600.0
    v = df["v"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hrs, v, marker="o", ms=4, lw=1.2, color="#0366d6")
    ax.set_xlabel("hours from start of window")
    ax.set_ylabel(f"{agg} {signal} per {bucket_label}")
    ax.set_title(title or f"{agg} {signal} per {bucket_label}  ({len(df)} bins)")
    ax.grid(alpha=0.2)
    out_path = Path(out_path)
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)
    return {"signal": signal, "agg": agg, "bucket": bucket_label,
            "bins": len(df), "path": str(out_path)}


def timeseries(globs: list[str], signals: list[str], out_path: str | Path,
               max_points: int = 5000, title: str | None = None) -> dict:
    """Plot one or more signals over time (each on its own y-scale if multiple)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = ", ".join(f'"{s}"' for s in signals)
    con = duckdb.connect()
    df = con.execute(
        f"SELECT timestamp AS t, {cols} FROM {_src(globs)} ORDER BY timestamp"
    ).df()
    con.close()
    if df.empty:
        raise ValueError("no data for requested signals")
    if len(df) > max_points:
        step = len(df) // max_points
        df = df.iloc[::step]
    t = df["t"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, s in enumerate(signals):
        ax_i = ax if i == 0 else ax.twinx()
        ax_i.plot(t, df[s].to_numpy(float), lw=0.8, label=s,
                  color=plt.cm.tab10(i))
        ax_i.set_ylabel(s)
    ax.set_xlabel("timestamp"); ax.set_title(title or " / ".join(signals))
    ax.grid(alpha=0.2)
    out_path = Path(out_path)
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)
    return {"signals": signals, "points": len(df), "path": str(out_path)}


def distance_series(segments: list[dict], out_path: str | Path,
                    signal: str, unit: str = "", seg_km: float = 20.0,
                    title: str | None = None) -> dict:
    """Bar chart of a per-segment quantity vs DISTANCE along the track.

    `segments` = [{km_start, km_end, value}] (value already computed per segment,
    e.g. litres consumed in that distance bin). This is the spatial analog of
    aggregated_series (which buckets by TIME); here the x-axis is distance.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not segments:
        raise ValueError("no distance segments to plot")
    mids = [(s["km_start"] + s["km_end"]) / 2 for s in segments]
    vals = [s.get("value") if s.get("value") is not None else 0.0 for s in segments]
    widths = [max(s["km_end"] - s["km_start"], 0.1) for s in segments]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(mids, vals, width=[w * 0.9 for w in widths], color="#0366d6",
           align="center")
    ax.set_xlabel("distance travelled (km)")
    ylab = f"{signal} per {seg_km:g} km" + (f" ({unit})" if unit else "")
    ax.set_ylabel(ylab)
    ax.set_title(title or f"{signal} per {seg_km:g} km along the track "
                 f"({len(segments)} segments)")
    ax.grid(alpha=0.2, axis="y")
    out_path = Path(out_path)
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)
    return {"signal": signal, "seg_km": seg_km, "segments": len(segments),
            "unit": unit, "path": str(out_path)}
