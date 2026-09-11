"""Thin DuckDB-backed access to parquet files.

This is the only module that knows how we physically read data. Everything
upstream (profiling, catalog, future query engine) goes through here, so the
storage/engine choice (local DuckDB now, DuckDB-on-Lambda later) stays swappable.
"""
from __future__ import annotations

from pathlib import Path

import duckdb


class ParquetReader:
    """Wraps a DuckDB connection scoped to a single parquet file."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._con = duckdb.connect()

    def _src(self) -> str:
        # Parameter-safe: path is our own, but keep it centralized here.
        return f"read_parquet('{self.path}')"

    def columns(self) -> list[tuple[str, str]]:
        """Return list of (column_name, duckdb_type)."""
        rows = self._con.execute(f"DESCRIBE SELECT * FROM {self._src()}").fetchall()
        return [(r[0], r[1]) for r in rows]

    def row_count(self) -> int:
        return self._con.execute(f"SELECT COUNT(*) FROM {self._src()}").fetchone()[0]

    def scalar(self, expr: str, where: str | None = None) -> object:
        sql = f"SELECT {expr} FROM {self._src()}"
        if where:
            sql += f" WHERE {where}"
        return self._con.execute(sql).fetchone()[0]

    def query(self, sql_body: str):
        """Run an arbitrary SELECT body against this file. `{src}` is substituted."""
        return self._con.execute(sql_body.format(src=self._src())).fetchall()

    def epoch_to_utc(self, epoch_seconds: float) -> str | None:
        if epoch_seconds is None:
            return None
        return self._con.execute(
            "SELECT strftime(make_timestamp(CAST(? * 1e6 AS BIGINT)), '%Y-%m-%d %H:%M:%S')",
            [epoch_seconds],
        ).fetchone()[0]

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> "ParquetReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
