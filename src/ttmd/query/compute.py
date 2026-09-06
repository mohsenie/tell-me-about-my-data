"""Deterministic aggregation queries over the data (DuckDB computes, not the LLM).

Correctness lives here. Notably: fuel consumption is the TIME-INTEGRAL of the
fuel-rate signal (L/h over time -> litres), NOT sum/avg of the rate column.
"""
from __future__ import annotations

from dataclasses import dataclass

import duckdb

# Aggregations exposed to the query layer (enum-bounded for the LLM router).
AGGREGATIONS = ("avg", "min", "max", "sum", "median", "stddev", "count", "integral")


@dataclass
class QueryResult:
    metric: str
    signal: str
    aggregation: str
    value: float | None
    unit: str
    window: str
    n_rows: int
    note: str = ""

    def human(self) -> str:
        if self.value is None:
            return f"No data for {self.signal} in {self.window}."
        return (f"{self.metric}: {self.value:,.2f} {self.unit} "
                f"({self.aggregation} of {self.signal}, window {self.window}, "
                f"{self.n_rows:,} rows).{(' ' + self.note) if self.note else ''}")


def _src(globs: list[str]) -> str:
    arr = ", ".join("'" + g + "'" for g in globs)
    return f"read_parquet([{arr}], union_by_name=true)"


def available_metrics(globs: list[str]) -> list[str]:
    con = duckdb.connect()
    schema = con.execute(f"DESCRIBE SELECT * FROM {_src(globs)}").fetchall()
    con.close()
    return [r[0] for r in schema if r[0].lower() != "timestamp"]


def _epoch_clause(t_range: tuple[float, float] | None) -> str:
    """Optional `AND timestamp BETWEEN lo AND hi` clause for a time window given
    as (start_epoch, end_epoch). Used for voyage/segment queries."""
    if not t_range:
        return ""
    lo, hi = sorted(t_range)
    return f" AND timestamp BETWEEN {lo} AND {hi}"


def aggregate(globs: list[str], signal: str, aggregation: str, window: str,
              unit: str = "", t_range: tuple[float, float] | None = None) -> QueryResult:
    """Generic scalar aggregation of a signal over the window.

    t_range: optional (start_epoch, end_epoch) to restrict the computation to a
    time segment (e.g. a voyage window derived from the position track)."""
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"unknown aggregation '{aggregation}'")
    if aggregation == "integral":
        return _integral(globs, signal, window, unit, t_range=t_range)

    con = duckdb.connect()
    fn = {"avg": "avg", "min": "min", "max": "max", "sum": "sum",
          "median": "median", "stddev": "stddev_samp", "count": "count"}[aggregation]
    row = con.execute(
        f'SELECT {fn}("{signal}"), count("{signal}") FROM {_src(globs)} '
        f'WHERE "{signal}" IS NOT NULL{_epoch_clause(t_range)}'
    ).fetchone()
    con.close()
    val = float(row[0]) if row[0] is not None else None
    return QueryResult(f"{aggregation} {signal}", signal, aggregation, val,
                       unit, window, int(row[1] or 0))


def _integral(globs: list[str], signal: str, window: str, unit: str,
              t_range: tuple[float, float] | None = None) -> QueryResult:
    """Time-integral of a RATE signal via trapezoidal rule over timestamps.

    For a rate in units/hour and timestamps in seconds:
      total = sum( rate_i * dt_seconds / 3600 )  (units)
    This is the correct 'total consumed' from a rate — NOT sum() of the column.
    """
    con = duckdb.connect()
    row = con.execute(f"""
        WITH s AS (
          SELECT timestamp AS t, "{signal}" AS v
          FROM {_src(globs)} WHERE "{signal}" IS NOT NULL{_epoch_clause(t_range)}
          ORDER BY timestamp
        ), d AS (
          SELECT v, t - lag(t) OVER (ORDER BY t) AS dt,
                 (v + lag(v) OVER (ORDER BY t)) / 2.0 AS v_mid
          FROM s
        )
        SELECT sum(v_mid * dt / 3600.0), count(*) FROM d
        WHERE dt IS NOT NULL AND dt > 0 AND dt < 3600  -- ignore gaps > 1h
    """).fetchone()
    con.close()
    val = float(row[0]) if row[0] is not None else None
    return QueryResult(f"total {signal}", signal, "integral", val, unit, window,
                       int(row[1] or 0),
                       note="time-integral of the rate (gaps >1h excluded).")


def fuel_consumption(globs: list[str], window: str,
                     rate_signal: str, unit: str = "L",
                     t_range: tuple[float, float] | None = None) -> QueryResult:
    """Total consumed = time-integral of a RATE signal -> total.

    The rate signal + unit are RESOLVED BY THE CALLER (from field semantics),
    NOT hardcoded. Ensures 'consumption' is an integral, never a sum/avg.
    t_range optionally restricts to a voyage/segment window.
    """
    res = _integral(globs, rate_signal, window, unit=unit, t_range=t_range)
    res.metric = f"total {rate_signal}"
    res.note = (f"time-integral of {rate_signal} (a rate) over time; NOT a sum of "
                "the rate column. Gaps >1h excluded.")
    return res


def latest(globs: list[str], signals: list[str]) -> dict:
    """Most-recent reading of one or more signals (for 'current X' questions).

    Returns {signal: (value, timestamp)}. Deterministic — reads the last non-null
    value per signal by timestamp.
    """
    con = duckdb.connect()
    out = {}
    for s in signals:
        row = con.execute(
            f'SELECT "{s}", timestamp FROM {_src(globs)} WHERE "{s}" IS NOT NULL '
            f'ORDER BY timestamp DESC LIMIT 1'
        ).fetchone()
        out[s] = (float(row[0]), float(row[1])) if row else (None, None)
    con.close()
    return out
