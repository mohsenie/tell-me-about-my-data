"""Data-access layer: thin wrappers over DuckDB reads of parquet."""
from .parquet_reader import ParquetReader

__all__ = ["ParquetReader"]
