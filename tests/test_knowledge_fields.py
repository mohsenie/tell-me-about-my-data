"""Functional tests for the deterministic parts of the knowledge base + the
field-semantics JSON parser (no LLM)."""
from __future__ import annotations

import json

import pytest

from ttmd.interpretation.knowledge import KnowledgeBase
from ttmd.interpretation.fields import _parse_json_array


# ---------------- KnowledgeBase ----------------
def test_kb_correction_roundtrip(tmp_path):
    p = tmp_path / "kb.json"
    kb = KnowledgeBase("ship", p)
    kb.add_correction("EngineSpeed", "FuelTemperature", "fuel line near turbo",
                      general_fact="fuel heated by nearby components")
    # persisted + reloadable
    assert p.exists()
    kb2 = KnowledgeBase("ship", p)
    hit = kb2.confirmed("EngineSpeed", "FuelTemperature")
    assert hit and hit["explanation"] == "fuel line near turbo"
    assert hit["source"] == "user_confirmed"
    # the general fact generalized into asset_facts
    facts = json.loads(p.read_text())["asset_facts"]
    assert any("heated" in f["fact"] for f in facts)


def test_kb_key_order_independent(tmp_path):
    kb = KnowledgeBase("ship", tmp_path / "kb.json")
    kb.add_correction("A", "B", "because")
    # lookup works regardless of argument order (undirected edge)
    assert kb.confirmed("A", "B") is not None
    assert kb.confirmed("B", "A") is not None


def test_kb_field_semantics_store_and_promote(tmp_path):
    kb = KnowledgeBase("ship", tmp_path / "kb.json")
    kb.set_field_semantics("engine", [
        {"field": "EngineSpeed", "description": "rpm", "unit": "rpm",
         "aggregation": "avg", "role": "none", "confidence": "high"},
        {"field": "EngineFuelRate", "description": "fuel rate", "unit": "L/h",
         "aggregation": "integral", "role": "none", "confidence": "high"},
    ])
    fs = kb.field_semantics("engine")
    assert len(fs) == 2
    # status starts as proposed
    assert all(f["status"] == "proposed" for f in fs)
    # promote makes it confirmed
    kb.promote_field("engine", "EngineSpeed")
    assert kb.confirmed_field("engine", "EngineSpeed") is not None
    assert kb.confirmed_field("engine", "EngineFuelRate") is None  # still proposed


def test_kb_field_semantics_replace_idempotent(tmp_path):
    kb = KnowledgeBase("ship", tmp_path / "kb.json")
    props = [{"field": "X", "description": "d", "unit": "u", "aggregation": "avg",
              "role": "none", "confidence": "low"}]
    kb.set_field_semantics("engine", props)
    kb.set_field_semantics("engine", props)   # re-run
    assert len(kb.field_semantics("engine")) == 1   # not duplicated


# ---------------- fields JSON parser ----------------
def test_parse_clean_array():
    raw = '```json\n[{"field":"a","role":"none"},{"field":"b","role":"latitude"}]\n```'
    out = _parse_json_array(raw)
    assert len(out) == 2 and out[1]["role"] == "latitude"


def test_parse_truncated_array_salvages_complete_objects():
    # simulate an LLM response cut off mid-array (no closing ]) — the parser
    # should still recover the complete objects before the cut.
    raw = ('[{"field":"a","description":"one"},'
           '{"field":"b","description":"two"},'
           '{"field":"c","descrip')   # truncated
    out = _parse_json_array(raw)
    assert len(out) == 2
    assert [o["field"] for o in out] == ["a", "b"]


def test_parse_garbage_returns_empty():
    assert _parse_json_array("no json here at all") == []
