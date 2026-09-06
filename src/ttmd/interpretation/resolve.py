"""Resolve user words -> actual signal names, data-driven (no hardcoded mapping).

Uses the available column names + any confirmed/proposed field-semantics
descriptions from the knowledge base, and lets the LLM pick the best match. This
is how "fuel" -> EngineFuelRate or "rpm" -> EngineSpeed without a hardcoded dict.
"""
from __future__ import annotations

import json
import re

from .provider import LLMProvider
from .knowledge import KnowledgeBase


_RESOLVE_SYSTEM = """Map the user's informal signal reference to the SINGLE best
matching column from the available signals. Use the descriptions if given.
Reply with ONLY the exact column name, or NONE if nothing matches."""


def candidate_signals(user_term: str, available: list[str]) -> list[str]:
    """Deterministic substring candidates for a term (before asking the LLM).

    Used to detect genuine AMBIGUITY: e.g. "fuel" matches FuelLevel,
    EngineFuelRate, FuelTemperature -> 3 candidates -> the chat should ask which.
    """
    low = user_term.strip().lower()
    # normalize BOTH sides the same way (drop spaces AND underscores) so a term
    # like "frequency_x" matches a column "frequency_x" (and "boost pressure"
    # matches "BoostPressure").
    tok = low.replace(" ", "").replace("_", "")
    hits = [c for c in available if tok in c.lower().replace("_", "").replace(" ", "")]
    return hits


def resolve_signal(user_term: str, source: str, available: list[str],
                   provider: LLMProvider, kb: KnowledgeBase | None = None) -> str | None:
    """Return the column name best matching `user_term`, or None."""
    # exact / case-insensitive direct hit first (cheap, no LLM)
    low = user_term.strip().lower()
    for c in available:
        if c.lower() == low:
            return c

    # build context: column + its (proposed/confirmed) description if we have one
    sem = {}
    if kb is not None:
        for fs in kb.field_semantics(source):
            sem[fs["field"]] = fs.get("description", "")
    lines = []
    for c in available:
        d = sem.get(c)
        lines.append(f"- {c}" + (f": {d}" if d else ""))
    prompt = (f"User reference: \"{user_term}\"\n\nAvailable signals:\n"
              + "\n".join(lines) + "\n\nBest matching column name:")

    ans = provider.complete(_RESOLVE_SYSTEM, prompt).strip()
    ans = re.sub(r"[^A-Za-z0-9_]", "", ans)  # strip stray punctuation/quotes
    if ans in available:
        return ans
    # fuzzy fallback: substring match
    for c in available:
        if low in c.lower() or c.lower() in low:
            return c
    return None


def resolve_aggregation(signal: str, source: str, kb: KnowledgeBase | None,
                        default: str = "avg") -> tuple[str, str]:
    """Return (aggregation, unit) for a signal from confirmed/proposed field
    semantics — e.g. a RATE -> 'integral'. Falls back to (default, "")."""
    if kb is not None:
        for fs in kb.field_semantics(source):
            if fs["field"] == signal:
                return fs.get("aggregation", default), fs.get("unit", "")
    return default, ""
