"""Domain-scope gate: bound the assistant to the asset's data context.

Rejects off-domain questions (films, general knowledge, coding, chit-chat) BEFORE
any expensive reasoning call. Two layers:
  1. Fast deterministic pre-filter (obvious off-domain / obvious in-domain).
  2. LLM classifier for the ambiguous middle (cheap yes/no call).

Use this at the entry of any free-form user input (the future chat layer). The
existing `interpret`/`correct` commands don't take free-form questions, so they
are already bounded by construction; this gate protects the chat surface.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .provider import LLMProvider

REFUSAL = ("That's outside what I can help with — I only analyze this asset's "
           "sensor data (relationships, operating modes, and anomalies).")

# In-domain vocabulary: asset/telemetry terms + how users INTERACT with the tool
# (meta/capability/query verbs). Extend per vertical.
_IN_DOMAIN = {
    # telemetry / asset terms
    "signal", "signals", "sensor", "relationship", "relationships", "correlation",
    "correlate", "correlated", "regime", "regimes", "anomaly", "anomalies",
    "engine", "fuel", "rpm", "speed", "pressure", "temperature", "temp",
    "vibration", "boost", "oil", "coolant", "nmea", "ais", "reading", "readings",
    "trend", "drift", "operating", "mode", "modes", "telemetry", "asset", "ship",
    "vessel", "vessels", "relate", "related", "ships", "boat", "boats",
    "nearby", "near", "around", "position", "location", "where", "located",
    "distance", "close", "proximity", "latitude", "longitude", "gps", "mmsi",
    # interaction / meta / query verbs (how users ask the tool for things)
    "ask", "query", "queries", "field", "fields", "data", "report", "reports",
    "help", "show", "list", "plot", "graph", "chart", "average", "avg", "mean",
    "max", "maximum", "min", "minimum", "total", "sum", "value", "values",
    "measure", "metric", "metrics", "analyze", "analyse", "describe", "capable",
    "capabilities", "what can", "why", "when", "how much", "how many",
    # time words
    "yesterday", "today", "days", "day", "hour", "hours", "week", "month",
    "window", "past", "last", "over",
}
# Obvious off-domain markers.
_OFF_DOMAIN = {
    "film", "movie", "watch", "recipe", "cook", "weather forecast", "joke",
    "poem", "stock", "crypto", "dating", "song", "sport", "score", "president",
    "capital of", "translate", "write code", "python script",
}

_WORD = re.compile(r"[a-z0-9]+")

_CLASSIFY_SYSTEM = (
    "You are a scope filter for an asset-telemetry diagnostic assistant. "
    "IN scope = anything about THIS asset's sensor data OR about using the tool: "
    "signals/fields, values, plots, correlations, operating modes, anomalies, "
    "reports, AND meta questions like 'what can I ask?', 'what fields are there?', "
    "'help', 'what can you do?'. "
    "OUT of scope = general knowledge, entertainment, opinions, coding, world "
    "facts, personal advice. "
    "When unsure, lean IN (it's a data tool; users ask about their data). "
    "Answer with exactly one word: IN or OUT.")


@dataclass
class ScopeDecision:
    in_scope: bool
    reason: str            # "prefilter_in" | "prefilter_out" | "llm_in" | "llm_out"


def _tokens(s: str) -> set[str]:
    return set(_WORD.findall(s.lower()))


def check_scope(text: str, provider: LLMProvider | None = None,
                context: str | None = None) -> ScopeDecision:
    """Decide if a message is in-domain.

    `context` (recent conversation) lets follow-ups like "is that a problem?" be
    judged in context rather than in isolation. Off-domain KEYWORDS still hard-
    block regardless of context, so injection/off-topic can't sneak in mid-chat.
    """
    low = text.lower()
    toks = _tokens(text)

    # Layer 1: fast deterministic pre-filter (hard block wins even in context).
    if any(m in low for m in _OFF_DOMAIN):
        return ScopeDecision(False, "prefilter_out")
    # single-token OR multi-word phrase (e.g. "what can", "how much") match
    if (toks & _IN_DOMAIN) or any(" " in m and m in low for m in _IN_DOMAIN):
        return ScopeDecision(True, "prefilter_in")

    # In a conversation, a short context-free follow-up (pronouns like "that",
    # "it", "should I worry") is almost certainly about the ongoing topic.
    if context:
        verdict = provider.complete(
            _CLASSIFY_SYSTEM,
            f"Ongoing conversation about this asset's telemetry:\n{context}\n\n"
            f"Latest message: {text}\n"
            f"Is the latest message part of THIS telemetry conversation?"
        ).strip().upper() if provider else "IN"
        return ScopeDecision(verdict.startswith("IN"),
                             "llm_in_ctx" if verdict.startswith("IN") else "llm_out_ctx")

    # Layer 2: ambiguous, no context -> cheap LLM classification.
    if provider is not None:
        verdict = provider.complete(_CLASSIFY_SYSTEM, text).strip().upper()
        if verdict.startswith("IN"):
            return ScopeDecision(True, "llm_in")
        return ScopeDecision(False, "llm_out")

    # No provider and no in-domain signal -> refuse to be safe.
    return ScopeDecision(False, "prefilter_out")
