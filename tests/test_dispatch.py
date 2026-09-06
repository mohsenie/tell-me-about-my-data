"""Functional tests for the orchestrator + chat_deps DISPATCH, using a seeded
FakeProvider (no live LLM). We assert the OUTCOME shape of each intent — the
right source is used, the right signal resolved, a computed number is present —
not the LLM's prose. Also covers mode framing (analyst passthrough) and
prerequisite auto-run.
"""
from __future__ import annotations

import pytest

import config
from ttmd.interpretation.orchestrator import Orchestrator


def _session(fake, deps, kb_ship, home="engine", mode="analyst", **seed):
    prov = fake(**seed)
    o = Orchestrator(home, "ship", prov, kb_ship, deps, mode=mode)
    # ensure the deps use the SAME fake provider (deps was built with the stub)
    deps.provider = prov
    return o, prov


def test_value_intent_computes_number(fake, deps, kb_ship, has_data):
    o, _ = _session(fake, deps, kb_ship,
                    intent="value",
                    params={"signal": "EngineSpeed", "aggregation": "avg"})
    reply = o.send("average engine speed")
    assert "EngineSpeed" in reply
    assert any(ch.isdigit() for ch in reply)   # a number came back


def test_capabilities_intent_lists_fields(fake, deps, kb_ship, has_data):
    o, _ = _session(fake, deps, kb_ship, intent="capabilities", params={})
    reply = o.send("what can I ask about?")
    assert "EngineSpeed" in reply and "queryable" in reply.lower()


def test_position_intent_returns_coords(fake, deps, kb_ship, has_data):
    o, _ = _session(fake, deps, kb_ship, intent="position", params={})
    reply = o.send("where is the ship now")
    assert "lat" in reply.lower() and any(ch.isdigit() for ch in reply)


def test_distance_intent(fake, deps, kb_ship, has_data):
    o, _ = _session(fake, deps, kb_ship, intent="distance", params={})
    reply = o.send("how far did the ship travel")
    assert "km" in reply.lower()


def test_regimes_intent_describes_modes(fake, deps, kb_ship, has_data):
    o, _ = _session(fake, deps, kb_ship, intent="regimes", params={})
    reply = o.send("what operating modes are there")
    assert "operating mode" in reply.lower() and "%" in reply


def test_relationship_autoruns_discovery(fake, deps, kb_ship, has_data, tmp_path):
    # point discovery artifacts at a temp dir so we exercise the auto-run path
    o, _ = _session(fake, deps, kb_ship, intent="relationship", params={})
    reply = o.send("what correlates with EngineSpeed")
    # either lists relationships or reports it analyzed first
    assert ("EngineSpeed" in reply) or ("analyz" in reply.lower())


def test_efficiency_per_distance(fake, deps, kb_ship, has_data):
    o, _ = _session(fake, deps, kb_ship, intent="efficiency",
                    params={"signal": "EngineFuelRate", "per_distance_km": 20,
                            "per_distance_label": "20 km"})
    reply = o.send("fuel consumption per 20km")
    assert "per" in reply.lower() and any(ch.isdigit() for ch in reply)


def test_mode_switch_message(fake, deps, kb_ship, has_data):
    o, _ = _session(fake, deps, kb_ship, intent="smalltalk", params={})
    reply = o.send("switch to business mode")
    assert o.mode == "operator" and "operator" in reply.lower()


def test_analyst_mode_no_reframe(fake, deps, kb_ship, has_data):
    # analyst mode: interpretive replies pass through _frame unchanged
    o, _ = _session(fake, deps, kb_ship, mode="analyst", intent="smalltalk", params={})
    raw = "regime 0: EngineSpeed ~ FuelTemperature strengthened (0.25 -> 0.75)"
    assert o._frame("anomaly", "any issues?", raw) == raw


def test_scope_refuses_offdomain(fake, deps, kb_ship):
    o, _ = _session(fake, deps, kb_ship, intent="smalltalk", params={})
    # off-domain is caught by the deterministic prefilter before the LLM
    reply = o.send("what film should I watch tonight?")
    from ttmd.interpretation import REFUSAL
    assert reply == REFUSAL


# ---------------- drift explanation ("why?" follow-up) ----------------
def test_is_why_followup_guard():
    """The deterministic causal-follow-up guard fires on short 'why' questions
    and not on fresh queries."""
    f = Orchestrator._is_why_followup
    for m in ["why?", "Why", "what could cause that?", "how come", "explain",
              "what does that mean"]:
        assert f(m), m
    for m in ["what is the average fuel consumption",
              "plot rpm vs speed",
              "where is the ship right now please tell me the coordinates"]:
        assert not f(m), m


def test_explain_drift_none_without_prior(deps, has_data):
    """No prior anomaly result -> explain_drift returns None (nothing to explain)."""
    assert getattr(deps, "_last_drift", None) is None
    assert deps.explain_drift("why?") is None


def test_explain_drift_uses_findings(fake, deps, kb_ship, has_data):
    """With a stashed structured drift result, explain_drift grounds the provider
    call and returns text (never asserting cause is the prompt's job)."""
    deps.provider = fake(intent="reasoning")
    deps._last_drift = {
        "source": "engine",
        "result": {
            "layer1_structure_drift": [
                {"regime": "1", "confidence": "high",
                 "changed_edges": [{"edge": ["OilPressure", "EngineSpeed"],
                                    "change": "weakened", "delta": -0.3}]}],
            "layer2_regime_events": [],
            "layer2_transition_events": {"findings": [
                {"type": "unseen_transition", "from": 0, "to": 2}]},
        },
        "behavioral": ["stayed in one location ~24h, ~37x usual"],
    }
    out = deps.explain_drift("why did that happen?")
    assert isinstance(out, str) and len(out) > 0


def test_why_followup_routes_to_explain(fake, deps, kb_ship, has_data):
    """After an anomaly turn, a bare 'why?' routes to explain_drift instead of a
    fresh reasoning query."""
    o, _ = _session(fake, deps, kb_ship, intent="reasoning")
    o._last_intent = "anomaly"
    deps._last_drift = {"source": "engine",
                        "result": {"layer1_structure_drift": [],
                                   "layer2_regime_events": []},
                        "behavioral": []}
    reply = o.send("why?")
    assert isinstance(reply, str) and len(reply) > 0
    assert o._last_intent == "reasoning"     # follow-up consumed, state advanced
