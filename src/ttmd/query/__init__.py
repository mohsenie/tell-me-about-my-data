"""Deterministic data queries — compute VALUES from the data on request.

Discipline: the LLM never computes numbers. It maps a request to a typed query
here; DuckDB computes the actual value. This module owns correctness (e.g. fuel
consumption = time-integral of a rate, NOT sum/avg of the rate column).
"""
from .capabilities import list_capabilities, describe_capabilities
from .compute import (
    aggregate, fuel_consumption, latest, available_metrics, QueryResult,
)
from .spatial import (
    current_position, ships_nearby, voyage_window, track_distance_km,
    distance_segments, entity_position_at, haversine_km, signal_by_location,
    nearest_place, place_label)

__all__ = [
    "list_capabilities", "describe_capabilities",
    "aggregate", "fuel_consumption", "latest", "available_metrics", "QueryResult",
    "current_position", "ships_nearby", "voyage_window", "track_distance_km",
    "distance_segments", "entity_position_at", "haversine_km", "signal_by_location",
]
