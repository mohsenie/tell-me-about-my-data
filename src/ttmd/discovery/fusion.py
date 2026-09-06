"""Cross-source fusion on the common time grid — fully automatic.

Joins multiple sources that share a time window onto one 10 Hz timestamp grid,
so discovery can find CROSS-SOURCE regimes and relationships. Column selection
is AUTOMATIC: no per-source configuration. Drop a new data type into ship-data
and it is fused with zero code changes.

Column names are prefixed with the source (source__column) so edges can be
identified as within-source vs cross-source.
"""
from __future__ import annotations

import glob

import duckdb
import numpy as np

GRID_HZ = 10
# Keep at most this many columns per source in a fusion (widest-variance first),
# so a single very-wide source (e.g. 59-column nmea) can't dominate/slow fusion.
MAX_COLS_PER_SOURCE = 10
EXCLUDE = ("timestamp",)

_NUMERIC = ("DOUBLE", "FLOAT", "BIGINT", "INTEGER", "HUGEINT", "DECIMAL",
            "SMALLINT", "TINYINT")


def _numeric_columns(con, pattern) -> list[str]:
    schema = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{pattern}', union_by_name=true)"
    ).fetchall()
    return [r[0] for r in schema
            if any(r[1].upper().startswith(t) for t in _NUMERIC)
            and r[0].lower() not in EXCLUDE]


def _auto_select_columns(con, pattern, cols: list[str]) -> list[str]:
    """Pick informative columns automatically: drop constant/all-null, then keep
    the highest-variance ones (standardized) up to MAX_COLS_PER_SOURCE."""
    if not cols:
        return []
    # stddev and non-null count per column, in one pass
    aggs = ", ".join(
        f'stddev_samp("{c}") AS s_{i}, count("{c}") AS n_{i}'
        for i, c in enumerate(cols))
    row = con.execute(
        f"SELECT {aggs} FROM read_parquet('{pattern}', union_by_name=true)"
    ).fetchone()

    scored = []
    for i, c in enumerate(cols):
        std = row[2 * i]
        cnt = row[2 * i + 1]
        if std is None or std == 0 or cnt == 0:
            continue  # constant or all-null: no relationship possible
        scored.append((float(std), c))
    if len(scored) <= MAX_COLS_PER_SOURCE:
        return [c for _, c in scored]
    # keep widest-variance columns (variance here is a proxy for "informative")
    scored.sort(reverse=True)
    return [c for _, c in scored[:MAX_COLS_PER_SOURCE]]


def _grid_subquery(con, source, pattern):
    cols = _numeric_columns(con, pattern)
    cols = _auto_select_columns(con, pattern, cols)
    if not cols:
        return None, []
    tag = source.replace("-", "_")
    sel = ", ".join(f'"{c}" AS "{tag}__{c}"' for c in cols)
    sub = f"""
      SELECT CAST(round(timestamp*{GRID_HZ}) AS BIGINT) AS tg, {sel}
      FROM read_parquet('{pattern}', union_by_name=true)
      QUALIFY row_number() OVER (
          PARTITION BY CAST(round(timestamp*{GRID_HZ}) AS BIGINT)
          ORDER BY timestamp) = 1
    """
    return sub, [f"{tag}__{c}" for c in cols]


def fuse_sources(source_globs: dict[str, str]) -> tuple[list[str], np.ndarray]:
    """Inner-join the given {source: glob} onto the common grid.

    Column selection per source is automatic. Only sources with a matching
    parquet AND at least one usable (varying) column are included; the join keeps
    only timestamps present in ALL included sources (the overlap window).
    """
    con = duckdb.connect()
    subs = []
    all_cols: list[str] = []
    for src, pattern in source_globs.items():
        if not glob.glob(pattern):
            continue
        sub, cols = _grid_subquery(con, src, pattern)
        if sub:
            subs.append(sub)
            all_cols.extend(cols)

    if len(subs) < 2:
        raise ValueError("fusion needs at least 2 sources with usable columns")

    q = f"({subs[0]}) t0"
    for i, s in enumerate(subs[1:], 1):
        q += f" JOIN ({s}) t{i} USING (tg)"
    df = con.execute(f"SELECT * FROM {q}").df()
    con.close()

    cols = [c for c in df.columns if c != "tg"]
    return cols, df[cols].to_numpy(dtype=float)


def is_cross_source(edge: dict) -> bool:
    """True if an edge connects two different sources (prefix before '__')."""
    return edge["a"].split("__")[0] != edge["b"].split("__")[0]


# below this silhouette, fused regimes are too blurred to trust as operating modes
FUSED_SILHOUETTE_MIN = 0.35


def fused_relationship_graph(source_globs: dict[str, str],
                             with_mi: bool = False) -> dict:
    """Cross-source discovery done the RIGHT way for heterogeneous sources.

    Fusing dissimilar sources onto one grid BLURS regime clustering (a vibration
    'at-rest vs vibrating' split and an engine 'idle vs cruise' split don't align,
    so joint KMeans finds a low-separation compromise — measured silhouette ~0.27
    vs ~0.92 single-source). So instead of trusting FUSED regimes, this computes:
      - the fused GLOBAL relationship graph (where the value is: CROSS-SOURCE edges,
        e.g. nmea.roll ~ vibration.accel_y — two sensors seeing the same motion),
      - a regime_quality DIAGNOSTIC (the fused silhouette + a 'blurred' flag) so the
        blur is surfaced honestly rather than presented as clean modes.
    Per-source regimes stay authoritative (run per-source discovery for those).

    Returns {global_graph, regime_quality, cross_source_edges, note}."""
    from .relationships import build_graph, clean_frame
    from .regimes import segment_regimes

    cols, data = fuse_sources(source_globs)
    graph = build_graph(cols, data, "global", with_mi=with_mi).to_dict()
    ccols, clean = clean_frame(cols, data)
    quality = {"silhouette": None, "k": 0, "blurred": None}
    if ccols and len(clean) >= 50:
        reg = segment_regimes(ccols, clean)
        sil = reg.silhouette
        quality = {
            "silhouette": round(sil, 3) if sil is not None else None,
            "k": reg.k,
            "blurred": (sil is not None and sil < FUSED_SILHOUETTE_MIN),
        }
    cross = [e for e in graph.get("edges", []) if is_cross_source(e)]
    cross.sort(key=lambda e: e.get("dcor", 0), reverse=True)
    note = ("fused regimes look blurred (low separation) — use per-source operating "
            "modes; the value here is the CROSS-SOURCE relationships below."
            if quality.get("blurred") else
            "fused regimes are reasonably separated.")
    return {
        "global_graph": graph,
        "regime_quality": quality,
        "cross_source_edges": cross,
        "note": note,
    }
