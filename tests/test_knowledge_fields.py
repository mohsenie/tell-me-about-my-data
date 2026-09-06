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


# ---------------- expert review: promote / reject extracted facts ----------------
def test_promote_extracted_fact(tmp_path):
    kb = KnowledgeBase("ship", tmp_path / "kb.json")
    kb.add_document_facts([{"fact": "Oil pressure drops at low idle.",
                            "citation": "manual p.12"}], doc_source="m.pdf")
    ext = [f for f in kb.asset_facts() if f["source"] == "document_extracted"]
    assert len(ext) == 1
    fid = ext[0]["id"]
    kb.promote_fact(fid)
    # now it's expert-confirmed, no longer an unverified extraction
    facts = kb.asset_facts()
    assert all(f["source"] != "document_extracted" for f in facts)
    assert any(f["id"] == fid and f["source"] == "user_confirmed" for f in facts)


def test_reject_extracted_fact_removes_it(tmp_path):
    kb = KnowledgeBase("ship", tmp_path / "kb.json")
    kb.add_document_facts([{"fact": "Wrong claim.", "citation": "m p.1"}],
                          doc_source="m.pdf")
    fid = kb.asset_facts()[0]["id"]
    kb.reject_fact(fid)
    assert kb.asset_facts() == []


def test_promote_unknown_id_raises(tmp_path):
    kb = KnowledgeBase("ship", tmp_path / "kb.json")
    with pytest.raises(KeyError):
        kb.promote_fact("deadbeef")


# ---------------- LLM usage metering (token/cost logging) ----------------
def test_usage_meter_counts_and_estimates():
    from ttmd.interpretation.provider import UsageMeter, StubProvider
    m = UsageMeter(StubProvider())
    m.complete("system prompt here", "user message one")
    m.complete("system two", "user message two longer")
    s = m.summary()
    assert s["calls"] == 2
    assert s["total_tokens"] == s["input_tokens"] + s["output_tokens"] > 0
    assert s["estimated_cost_usd"] >= 0
    assert s["tokens_estimated"] is True          # stub -> estimated from length
    assert "call(s)" in m.summary_line()


def test_get_provider_meter_flag():
    from ttmd.interpretation.provider import get_provider, UsageMeter
    assert isinstance(get_provider(meter=True), UsageMeter)
    assert not isinstance(get_provider(meter=False), UsageMeter)


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
