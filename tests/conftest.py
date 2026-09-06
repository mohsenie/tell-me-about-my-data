"""Shared pytest fixtures for functional tests.

Design: tests assert OUTCOMES (files created, numeric results, structure) on the
real vessel data — deterministically, with NO live LLM. Two provider options:
  - StubProvider (the real offline stub) for pure/deterministic paths.
  - FakeProvider (here) for orchestrator DISPATCH tests: returns canned intent /
    param / resolve / scope responses so the routing+dispatch machinery runs
    without Bedrock. It answers based on WHICH system-prompt it was handed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import config  # noqa: E402
from ttmd.interpretation.provider import LLMProvider  # noqa: E402


# ---- fake, deterministic LLM provider for dispatch tests -------------------
class FakeProvider(LLMProvider):
    """Deterministic provider. You seed it with an `intent` and `params` for the
    turn under test; it returns them for the intent/param calls, echoes signal
    resolution, and passes scope. For the persona-framing call it returns the
    answer unchanged (so tests can assert the underlying content). No network."""

    def __init__(self, intent: str = "smalltalk", params: dict | None = None,
                 resolve_to: str | None = None):
        self.intent = intent
        self.params = params or {}
        self.resolve_to = resolve_to
        self.calls = []

    def complete(self, system: str, user: str, max_tokens=None) -> str:
        self.calls.append(system[:40])
        s = system.lower()
        if s.startswith("you classify"):
            return self.intent
        if s.startswith("you extract structured query parameters"):
            return json.dumps(self.params)
        if s.startswith("map the user's informal"):
            # echo the requested resolution, else the first available token
            return self.resolve_to or "NONE"
        if "scope filter" in s:
            return "IN"
        if s.startswith("you are continuing a conversation"):
            # persona framing: return the analyst answer unchanged so tests can
            # assert on the real content (framing is exercised elsewhere)
            m = re.search(r"re-express.*?:\n(.*)\n\nYour reply:", user, re.DOTALL)
            return m.group(1).strip() if m else user
        # summarize / reason / anything else: passthrough marker
        return "(fake) synthesized answer"


@pytest.fixture
def vessel():
    return config.DEFAULT_VESSEL


@pytest.fixture
def sources(vessel):
    return config.discover_sources(vessel)


@pytest.fixture
def has_data(sources):
    """Skip a test cleanly if the sample vessel data isn't present."""
    if not sources:
        pytest.skip("no vessel data on disk")
    return True


@pytest.fixture
def globs():
    """Callable -> present parquet globs for a source (all dates)."""
    import glob as _glob

    def _get(source, v=None):
        pat = config.source_glob(source, v or config.DEFAULT_VESSEL)
        return [g for g in _glob.glob(pat)]
    return _get


@pytest.fixture
def stub_provider():
    from ttmd.interpretation.provider import StubProvider
    return StubProvider()


@pytest.fixture
def kb_ship():
    from ttmd.cli_helpers import kb
    return kb("ship")


@pytest.fixture
def deps(stub_provider, kb_ship, vessel):
    """ChatDeps wired with the offline stub — for testing the DETERMINISTIC
    capability methods (routing, resolution, spatial, anomaly, per_ratio...)."""
    from ttmd.chat_deps import ChatDeps
    return ChatDeps("ship", stub_provider, kb_ship, vessel)


@pytest.fixture
def fake():
    """Factory for a seeded FakeProvider: fake(intent='value', params={...})."""
    def _make(**kw):
        return FakeProvider(**kw)
    return _make


@pytest.fixture
def pos_source(deps, has_data):
    """The single-entity position source + its lat/lon columns, or skip."""
    src = deps._own_position_source()
    if not src:
        pytest.skip("no position source")
    lat, lon = deps._latlon_cols(src)
    if not (lat and lon):
        pytest.skip("no lat/lon on position source")
    return src, lat, lon
