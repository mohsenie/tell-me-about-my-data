"""Shared helpers for the CLI and the NL router.

Keeps window-resolution, knowledge-base construction, and summary printing in one
place so both the argparse commands and the natural-language router reuse them.
"""
from __future__ import annotations

import glob
from collections import Counter

import config
from galene import term
from galene.interpretation import KnowledgeBase, ActorRegistry


def kb(asset_type: str) -> KnowledgeBase:
    return KnowledgeBase(asset_type, config.ARTIFACTS_DIR / f"knowledge_{asset_type}.json")


def actor_registry() -> ActorRegistry:
    """The actors directory (global — people aren't vessel-scoped)."""
    return ActorRegistry(config.ARTIFACTS_DIR / "actors.json")


def resolve_window(source: str, vessel: str, days=None, date_from=None, date_to=None):
    """Resolve date partitions from days / from / to. Returns (globs, label).

    Empty selectors -> all available dates.
    """
    available = config.available_dates(source, vessel)
    if not available:
        return None, "none"
    selected = available
    if days:
        selected = available[-days:]
    elif date_from or date_to:
        lo = date_from or available[0]
        hi = date_to or available[-1]
        selected = [d for d in available if lo <= d <= hi]
    if not selected:
        return [], "none"
    label = selected[0] if len(selected) == 1 else f"{selected[0]}..{selected[-1]}"
    return config.source_globs_for_dates(source, selected, vessel), label


def present_globs(globs):
    """Filter to globs that actually match files on disk."""
    return [g for g in (globs or []) if glob.glob(g)]


def print_discovery_summary(source: str, result: dict) -> None:
    reg = result["regimes"]
    print("=" * 78)
    print(f"DISCOVERY {source}: {reg['k']} regimes (silhouette={reg['silhouette']})")
    if reg.get("note"):
        print(f"  note: {reg['note']}")
    for r in reg["regimes"]:
        print(f"  regime {r['label']}: {r['fraction']*100:.0f}% of data ({r['size']} rows)")
    if not result["graphs"]:
        return
    print("\nrelationship graphs (significant edges per scope):")
    for scope, g in result["graphs"].items():
        kinds = Counter(e["kind"] for e in g["edges"])
        print(f"  {scope:14s} n={g['n_samples']:5d}  edges={len(g['edges'])}  {dict(kinds)}")
