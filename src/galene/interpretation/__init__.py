"""LLM interpretation layer: suggests (hypothesizes) root causes for the
measured relationships/regimes, given the asset type, and learns from user
corrections.

Discipline: the deterministic report states FACTS (what was measured). This layer
adds HYPOTHESES (why it might be) — always labeled as suggestions, never asserted.
User corrections become attributed KNOWLEDGE that overrides guesses and informs
future ones (retrieval-augmented, not fine-tuning).
"""
from .provider import LLMProvider, StubProvider, get_provider
from .knowledge import KnowledgeBase
from .interpret import interpret_report, apply_correction
from .scope import check_scope, ScopeDecision, REFUSAL
from .actors import ActorRegistry, AmbiguousActor

__all__ = [
    "LLMProvider", "StubProvider", "get_provider",
    "KnowledgeBase", "interpret_report", "apply_correction",
    "check_scope", "ScopeDecision", "REFUSAL",
    "ActorRegistry", "AmbiguousActor",
]
