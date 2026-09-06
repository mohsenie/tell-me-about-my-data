"""Load a source's numeric signals into (columns, ndarray) for discovery.

Kept separate so discovery stays IO-light and testable; uses the shared
ParquetReader connection style but returns plain numpy.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np


def load_numeric(path, where: str | None = None,
                 exclude=("timestamp",)) -> tuple[list[str], np.ndarray]:
    """Return (column_names, data) for numeric columns.

    `path` may be a single file, a glob spanning partitioned files, OR a LIST of
    globs (e.g. specific date partitions for a time-window query). DuckDB reads
    and unions them; union_by_name tolerates schema differences across days.
    """
    con = duckdb.connect()
    if isinstance(path, (list, tuple)):
        arr = ", ".join("'" + str(p) + "'" for p in path)
        src = f"read_parquet([{arr}], union_by_name=true)"
    else:
        src = f"read_parquet('{path}', union_by_name=true)"
    schema = con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()
    numeric = ("DOUBLE", "FLOAT", "BIGINT", "INTEGER", "HUGEINT", "DECIMAL",
               "SMALLINT", "TINYINT")
    cols = [r[0] for r in schema
            if any(r[1].upper().startswith(t) for t in numeric)
            and r[0].lower() not in exclude]
    sql = f"SELECT {', '.join(cols)} FROM {src}"
    if where:
        sql += f" WHERE {where}"
    df = con.execute(sql).df()
    con.close()
    return cols, df.to_numpy(dtype=float)


def load_numeric_with_time(path, where: str | None = None,
                           exclude=("timestamp",)
                           ) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Like load_numeric, but ALSO returns the per-row timestamps (epoch seconds),
    time-ordered. Additive — for detectors that need temporal order (regime
    transitions, multi-timescale, leg detection). Rows are sorted by timestamp so
    `data[i]` and `ts[i]` correspond and the sequence is chronological.
    Returns (columns, data, timestamps)."""
    con = duckdb.connect()
    if isinstance(path, (list, tuple)):
        arr = ", ".join("'" + str(p) + "'" for p in path)
        src = f"read_parquet([{arr}], union_by_name=true)"
    else:
        src = f"read_parquet('{path}', union_by_name=true)"
    schema = con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()
    numeric = ("DOUBLE", "FLOAT", "BIGINT", "INTEGER", "HUGEINT", "DECIMAL",
               "SMALLINT", "TINYINT")
    cols = [r[0] for r in schema
            if any(r[1].upper().startswith(t) for t in numeric)
            and r[0].lower() not in exclude]
    has_ts = any(r[0].lower() == "timestamp" for r in schema)
    if not has_ts:
        # no timestamp column -> behave like load_numeric, empty ts
        df = con.execute(f"SELECT {', '.join(cols)} FROM {src}"
                         + (f" WHERE {where}" if where else "")).df()
        con.close()
        return cols, df.to_numpy(dtype=float), np.empty(0)
    sql = (f"SELECT timestamp, {', '.join(cols)} FROM {src}"
           + (f" WHERE {where}" if where else "") + " ORDER BY timestamp")
    df = con.execute(sql).df()
    con.close()
    ts = df["timestamp"].to_numpy(dtype=float)
    data = df[cols].to_numpy(dtype=float)
    return cols, data, ts
