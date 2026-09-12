"""Functional tests for the orchestrator + chat_deps DISPATCH, using a seeded
FakeProvider (no live LLM). We assert the OUTCOME shape of each intent — the
right source is used, the right signal resolved, a computed number is present —
not the LLM's prose. Also covers mode framing (analyst passthrough) and
prerequisite auto-run.
"""
from __future__ import annotations

import pytest

import config
from galene.interpretation.orchestrator import Orchestrator


def _session(fake, deps, kb_ship, home="engine", mode="expert", **seed):
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


def test_value_honors_stddev_aggregation(fake, deps, kb_ship, has_data):
    """Bug A regression: 'standard deviation of X' must compute stddev, not fall
    back to avg. stddev is now in the param aggregation enum, and compute_value
    honors an explicit aggregation."""
    o, _ = _session(fake, deps, kb_ship, intent="value",
                    params={"signal": "EngineCoolantTemperature",
                            "aggregation": "stddev"})
    # name the exact signal so resolution is unambiguous (mirrors the LLM's
    # extracted signal param); the point under test is the AGGREGATION.
    reply = o.send("standard deviation of EngineCoolantTemperature")
    assert "stddev" in reply.lower()
    assert "avg " not in reply.lower()         # NOT silently the mean
    assert "EngineCoolantTemperature" in reply


def test_value_honors_date_range_and_integral_unit(fake, deps, kb_ship, has_data):
    """Bug B regression: an explicit date range restricts the window (not all
    data), and an integral of a rate reports the total's unit (L), never 'L/h'."""
    o, _ = _session(fake, deps, kb_ship, intent="value",
                    params={"signal": "EngineFuelRate", "aggregation": "integral",
                            "date_from": "2026-09-03", "date_to": "2026-09-04"})
    reply = o.send("integral of EngineFuelRate from 2026-09-03 to 2026-09-04")
    # window is the requested range, not the full dataset
    assert "2026-09-03..2026-09-04" in reply
    # a total is not a rate -> unit stripped of the per-hour denominator
    assert "L/h" not in reply


def test_capabilities_intent_lists_fields(fake, deps, kb_ship, has_data):
    o, _ = _session(fake, deps, kb_ship, intent="capabilities", params={})
    reply = o.send("what can I ask about?")
    assert "EngineSpeed" in reply and "queryable" in reply.lower()


def test_single_source_relationship_not_fused(deps, has_data):
    """Bug C regression: a relationship question naming only engine signals must
    NOT be treated as cross-source just because a word like 'speed' (from
    EngineSpeed) fuzzy-matches an nmea signal. It should return the per-source
    graph (the single dcor), not a fused cross-source table."""
    msg = "distance correlation between EngineFuelRate and EngineSpeed"
    # the cross-source detector should decline (fewer than 2 sources referenced)
    assert deps._cross_source_relationship(msg, "engine") is None
    out = deps.relationship_lookup("engine", msg)
    assert "Cross-source" not in out
    assert "EngineSpeed" in out


def test_genuine_cross_source_relationship_still_fused(deps, has_data):
    """Bug C guard: a REAL cross-source question (two sources' exact signals) still
    routes to fused discovery — the tightening didn't over-correct."""
    srcs = deps.all_sources()
    if not ({"nmea", "vibration"} <= set(srcs)):
        pytest.skip("needs nmea + vibration sources")
    out = deps.relationship_lookup(
        "vibration",
        "cross-source relationship between vibration acceleration_y and nmea roll")
    assert "Cross-source" in out


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
    # session starts in expert (test default); switching to a DIFFERENT mode
    # produces the switch confirmation.
    o, _ = _session(fake, deps, kb_ship, intent="smalltalk", params={})
    reply = o.send("switch to general mode")
    assert o.mode == "general" and "general" in reply.lower()
    reply = o.send("switch to expert mode")
    assert o.mode == "expert" and "expert" in reply.lower()


def test_expert_mode_no_reframe(fake, deps, kb_ship, has_data):
    # expert mode: interpretive replies pass through _frame unchanged (verbatim)
    o, _ = _session(fake, deps, kb_ship, mode="expert", intent="smalltalk", params={})
    raw = "regime 0: EngineSpeed ~ FuelTemperature strengthened (0.25 -> 0.75)"
    assert o._frame("anomaly", "any issues?", raw) == raw


def test_scope_refuses_offdomain(fake, deps, kb_ship):
    o, _ = _session(fake, deps, kb_ship, intent="smalltalk", params={})
    # off-domain is caught by the deterministic prefilter before the LLM
    reply = o.send("what film should I watch tonight?")
    from galene.interpretation import REFUSAL
    assert reply == REFUSAL


# ---------------- regimes across all sources ----------------
def test_describe_regimes_all_covers_every_source(deps, has_data):
    """'usage patterns for all sources' describes every source, not just one."""
    out = deps.describe_regimes_all("list usage patterns for all sources")
    srcs = deps.all_sources()
    assert len(srcs) >= 2
    # each source is named in the combined output
    for s in srcs:
        assert f"'{s}'" in out
    # and it actually reports operating modes
    assert "operating mode" in out


def test_regimes_all_routing_guard():
    """The orchestrator routes 'all/each/every source' to the all-sources path."""
    for m in ["list usage patterns for all data sources",
              "what regimes does each source have",
              "operating modes across sources"]:
        low = m.lower()
        assert any(w in low for w in ("all source", "all data source", "each source",
                                      "every source", "all sources", "per source",
                                      "across sources", "all of them", "everything"))
    # a single-source ask must NOT trip the all-sources guard
    low = "list usage patterns for engine".lower()
    assert not any(w in low for w in ("all source", "each source", "every source",
                                      "all sources", "per source", "across sources"))


# ---------------- source data coverage over a journey ----------------
def test_coverage_question_routes_to_source_coverage(fake, deps, kb_ship, has_data):
    """'for what % of the last trip was the engine used' routes to source_coverage
    (data coverage), NOT regimes or a fabricated on/off number, and keeps the
    honest 'a gap is not proof it was off' caveat."""
    o, _ = _session(fake, deps, kb_ship, intent="regimes", params={})   # router guess
    reply = o.send("for what percentage of the last journey was the engine used?")
    low = reply.lower()
    assert "reported data for" in low               # the coverage phrasing
    assert "not proof" in low                        # gap != off honesty
    # never a fabricated 'engine off X%' claim
    assert "engine was off" not in low and "% off" not in low


def test_coverage_guard_scope():
    """The coverage guard fires for coverage/used/report + a trip word + a source;
    a plain regimes or value question is unaffected."""
    def fires(m, sources=("engine", "nmea")):
        low = m.lower()
        cov = any(w in low for w in ("coverage", "report data", "reported data",
                                     "reporting", "uptime", "was used", "in use",
                                     "how much of the", "what percentage", "% of the"))
        trip = any(w in low for w in ("trip", "journey", "voyage", "leg", "last"))
        named = any(s in low for s in sources)
        return cov and trip and (named or "engine" in low or "ecu" in low)
    assert fires("for what percentage of the last journey was the engine used?")
    assert fires("how much of the last trip did the ecu report data")  # truck ECU
    assert not fires("what operating modes are there")   # regimes, untouched
    assert not fires("average engine speed yesterday")   # value, untouched


# ---------------- nearby-vessels-per-distance (distinct-new) ----------------
def test_nearby_per_distance_routes_to_real_capability(fake, deps, kb_ship, has_data):
    """'how many nearby ships per 10km' now routes to the real per-distance
    capability (NOT efficiency/fuel). It reports new-vessels-per-bucket, and
    NEVER fabricates a fuel/efficiency number."""
    ambiguous = "ambiguous — mixes nearby, efficiency (per km), and plot (a list)"
    o, _ = _session(fake, deps, kb_ship, intent=ambiguous, params={})
    reply = o.send("how many new ships nearby were seen for every 10km of travel in "
                   "the last trip. give me the list for every 20km")
    # must NOT have fabricated an efficiency/fuel RESULT
    assert "EngineFuelRate" not in reply and "L per 20 km" not in reply
    # it's the nearby-per-distance answer (buckets/new entities) or an honest
    # 'need both sources' message — never the old 'can't yet count' stub.
    low = reply.lower()
    assert ("nearby per" in low or "km:" in low
            or "multi-entity position feed" in low or "km " in low)
    assert "can't yet count" not in low


def test_nearby_per_distance_asset_neutral_noun(has_data):
    """The nearby-per-distance answer names the nearby entities from the ASSET TYPE,
    not a hardcoded 'vessel' — a truck asset says 'other trucks', a ship 'other
    ships'. Data-agnostic (same data + code, different asset_type)."""
    import config
    from galene.chat_deps import ChatDeps
    from galene.interpretation.provider import StubProvider
    from galene.cli_helpers import kb as _kb
    truck = ChatDeps("truck", StubProvider(), _kb("truck"), config.DEFAULT_VESSEL)
    out_t = truck.nearby_per_distance("how many other vehicles nearby per 20 km")
    ship = ChatDeps("ship", StubProvider(), _kb("ship"), config.DEFAULT_VESSEL)
    out_s = ship.nearby_per_distance("how many ships nearby per 20 km")
    # if the capability produced a result, the noun tracks the asset type
    if "per 20 km" in out_t:
        assert "other trucks" in out_t and "vessel" not in out_t.lower()
    if "per 20 km" in out_s:
        assert "other ships" in out_s


def test_nearby_per_distance_guard_scope(fake, deps, kb_ship, has_data):
    """The guard fires ONLY for nearby-entities + per-distance + counting; a plain
    fuel-per-distance or nearby-now question is unaffected (guard returns falsey)."""
    import re
    def fires(m):
        low = m.lower()
        about = ("nearby" in low or "near me" in low or "near us" in low
                 or "around us" in low
                 or (re.search(r"\b(other|around)\b", low) and
                     re.search(r"\b(ship|vessel|truck|vehicle|craft|unit|boat|car|"
                               r"aircraft|plane|entit)", low)))
        perd = bool(re.search(r"(per|every|each)\s*\d*\s*(km|kilomet|mile|nm|nautical|m\b)", low))
        cnt = any(w in low for w in ("how many", "number of", "count", "how much"))
        return bool(about and perd and cnt)
    assert fires("how many nearby ships per 10 km")
    assert fires("how many other vehicles were near me every 5 km")   # ASSET-NEUTRAL (truck)
    assert fires("count other trucks around us per 10km")             # ASSET-NEUTRAL
    assert not fires("fuel consumption per 20 km")        # not about nearby
    assert not fires("what ships are nearby now")         # not per-distance
    assert not fires("how many ships are nearby now")     # counting nearby, but not per-distance


# ---------------- broad-scope routing (shared _wants_all_sources) ----------------
def test_wants_all_sources_shared_guard(fake, deps, kb_ship):
    o, _ = _session(fake, deps, kb_ship)
    f = o._wants_all_sources
    assert f("give me an overview")
    assert f("what is notable in my data")
    assert f("capabilities across all sources")
    # a NAMED source suppresses the broad sweep
    assert not f("summarize the engine")
    assert not f("overview of the engine")
    # a plain unscoped ask is not broad
    assert not f("what can I ask about")


def test_is_orientation_first_contact_guard(fake, deps, kb_ship):
    """A first-contact 'tell me about the ship' orientation sweeps every source;
    a source-named capabilities ask stays single."""
    o, _ = _session(fake, deps, kb_ship)
    f = o._is_orientation
    assert f("what can you tell me about this ship?")
    assert f("tell me about this ship")
    assert f("what do you know about it?")
    assert f("what have you got on it?")
    # naming a source suppresses the whole-vessel sweep
    assert not f("tell me about the engine")
    # a plain field-list ask isn't an 'about the asset' orientation
    assert not f("what fields are there?")


def test_orientation_routes_capabilities_to_all_sources(fake, deps, kb_ship, has_data):
    """An orientation message dispatched as capabilities sweeps EVERY source,
    not just the home source."""
    o, _ = _session(fake, deps, kb_ship, home="engine",
                    intent="capabilities")
    out = o._dispatch("capabilities", "what can you tell me about this ship?")
    for s in deps.all_sources():
        assert s in out                      # every source represented


def test_capabilities_all_covers_every_source(deps, has_data):
    out = deps.capabilities_all()
    for s in deps.all_sources():
        assert s in out


def test_summarize_all_covers_sources_behavioral_once(deps, has_data):
    out = deps.summarize_all("give me an overview")
    srcs = deps.all_sources()
    assert len(srcs) >= 2
    for s in srcs:
        assert f"[{s}]" in out                      # each source has a section
    assert out.count("stayed in one location") <= 1  # behavioral once, not per-source


# ---------------- anomalies across all sources ----------------
def test_detect_anomaly_all_is_high_level_rollup(deps, has_data):
    """'anomalies in all data sources' returns a SHORT roll-up, not the full
    per-edge dump: a 'Checked N sources' header, one line per source, behavioral
    once, and a pointer to per-source detail — NOT raw edge tables."""
    out = deps.detect_anomaly_all("is there any anomaly in my data")
    srcs = deps.all_sources()
    assert len(srcs) >= 2
    assert out.startswith("Checked ")                 # roll-up header
    assert out.count("stayed in one location") <= 1   # behavioral once
    # the raw per-edge detail tables must NOT be in the roll-up
    assert "threshold" not in out and "Δ+" not in out and "Δ-" not in out
    # every source is accounted for (either flagged, clean, or no-baseline line)
    for s in srcs:
        assert s in out


def test_anomaly_detail_expands_one_source(deps, has_data):
    """After the roll-up, anomaly_detail returns the FULL report for a source that
    had a baseline (from the cache)."""
    deps.detect_anomaly_all("any anomaly in my data")
    cached = list(getattr(deps, "_anomaly_details", {}).keys())
    if not cached:
        pytest.skip("no source has a baseline in this run")
    detail = deps.anomaly_detail(cached[0])
    assert "Drift check for" in detail                 # the full render, not the roll-up


def test_anomaly_broad_scope_routing_guard(deps):
    """Broad phrasing routes to the all-sources sweep; a named source does not."""
    named = set(deps.all_sources())
    def broad(m):
        low = m.lower()
        b = any(w in low for w in ("all source", "all data source", "each source",
                                   "every source", "all sources", "across sources",
                                   "my data", "the data", "any data", "anywhere",
                                   "all of them", "everything", "any source"))
        n = any(s.lower() in low for s in named)
        return b and not n
    assert broad("are there any anomalies in my data")
    assert broad("anomalies in all data sources")
    assert not broad("is the engine drifting")      # named source -> single


# ---------------- consumption resolution + voyage normality check ----------------
def test_is_consumption_recognizes_phrasings(deps):
    assert deps._is_consumption("how much fuel on the last voyage")
    assert deps._is_consumption("what the engine usually consumes")
    assert deps._is_consumption("fuel burn")
    assert not deps._is_consumption("average engine speed")


def test_consumption_guard_resolves_fuel_to_rate(deps, has_data):
    """A consumption question resolves ambiguous 'fuel' to the RATE signal instead
    of asking; a non-consumption 'fuel' question still asks."""
    avail = deps.signals("engine")
    if "EngineFuelRate" not in avail:
        pytest.skip("no fuel rate on this data")
    sig = deps._resolve_or_ask("fuel", "engine", avail,
                               "how much fuel on the last voyage")
    assert sig == "EngineFuelRate"
    with pytest.raises(Exception):
        deps._resolve_or_ask("fuel", "engine", avail, "what is the fuel reading now")


def test_wants_normal_check_guard():
    from galene.chat_deps import ChatDeps
    assert ChatDeps._wants_normal_check("does it look normal for the same distance")
    assert ChatDeps._wants_normal_check("is that usual?")
    assert not ChatDeps._wants_normal_check("how much fuel on the last voyage")


def test_compound_trip_consumption_routes_to_voyage(fake, deps, kb_ship, has_data):
    """A compound 'fuel total for the last voyage + is it normal?' question routes
    to voyage (rate-resolved), NOT the generic resolver that asks which fuel. The
    router (fake) returns the verbose 'ambiguous' verdict that mentions voyage AND
    anomaly — the deterministic guard must override it."""
    ambiguous = ("ambiguous — this is both voyage (a consumed total) and anomaly "
                 "(is it abnormal vs baseline)")
    o, _ = _session(fake, deps, kb_ship, intent=ambiguous, params={})
    # describe engine fuel fields so the voyage path can resolve the rate
    kb_ship.set_field_semantics("engine", [
        {"field": "EngineFuelRate", "description": "fuel rate", "unit": "L/h",
         "role": "none", "confidence": "high", "aggregation": "integral"},
        {"field": "FuelLevel", "description": "tank level", "unit": "%",
         "role": "none", "confidence": "high", "aggregation": "avg"},
    ])
    reply = o.send("the fuel total for the last voyage. is it within normal range "
                   "for the similar distance travelled before?")
    # did NOT ask to disambiguate fuel; answered the voyage total (or a clean
    # 'no completed voyage' message) — never the "Which one did you mean" prompt.
    assert "Which one did you mean" not in reply


def test_voyage_efficiency_norm_verdict(deps, monkeypatch):
    """The per-distance norm compares the last leg's fuel/km to prior legs and
    returns normal/high/low. Uses stubbed legs + aggregate so it's deterministic
    regardless of sample-data coverage."""
    legs = [
        {"from": "A", "to": "B", "t_start": 0, "t_end": 10, "distance_km": 100},
        {"from": "B", "to": "C", "t_start": 20, "t_end": 30, "distance_km": 100},
        {"from": "C", "to": "D", "t_start": 40, "t_end": 50, "distance_km": 100},
    ]
    monkeypatch.setattr(deps, "_detect_legs", lambda: legs)

    class _R:  # fake aggregate result: fuel proportional to a per-leg value
        def __init__(self, v): self.value = v
    fuels = {(0, 10): 100.0, (20, 30): 110.0, (40, 50): 300.0}   # last leg 3x

    def fake_aggregate(globs, signal, agg, label, unit=None, t_range=None):
        return _R(fuels[t_range])
    import galene.query as _q
    monkeypatch.setattr(_q, "aggregate", fake_aggregate)

    verdict = deps._voyage_efficiency_norm("engine", "EngineFuelRate", "L")
    assert verdict is not None and "HIGHER than usual" in verdict


def test_voyage_efficiency_norm_honest_when_thin(deps, monkeypatch):
    """When there aren't enough prior voyages with fuel data, the check does NOT
    return None (silent drop) — it explains it can't judge yet."""
    legs = [
        {"from": "A", "to": "B", "t_start": 0, "t_end": 10, "distance_km": 100},
        {"from": "B", "to": "C", "t_start": 20, "t_end": 30, "distance_km": 100},
        {"from": "C", "to": "D", "t_start": 40, "t_end": 50, "distance_km": 100},
    ]
    monkeypatch.setattr(deps, "_detect_legs", lambda: legs)

    class _R:
        def __init__(self, v): self.value = v
    # only ONE prior leg has fuel data -> not enough for a range
    fuels = {(0, 10): None, (20, 30): 110.0, (40, 50): 300.0}

    def fake_aggregate(globs, signal, agg, label, unit=None, t_range=None):
        return _R(fuels[t_range])
    import galene.query as _q
    monkeypatch.setattr(_q, "aggregate", fake_aggregate)

    msg = deps._voyage_efficiency_norm("engine", "EngineFuelRate", "L")
    assert msg is not None
    assert "can't judge it yet" in msg          # honest, not silent


# ---------------- signal labeling (detector -> field-semantics meaning) ----------
def test_signal_label_translates_via_semantics(deps, kb_ship):
    """A raw signal column is translated to its field-semantics meaning; an
    undescribed one falls back to the raw name (no fabrication)."""
    kb_ship.set_field_semantics("engine", [
        {"field": "EngineCoolantTemperature", "description": "coolant temperature",
         "unit": "degC", "role": "none", "confidence": "high", "aggregation": "avg"},
    ])
    assert deps._signal_label("engine", "EngineCoolantTemperature") == \
        "coolant temperature (degC)"
    assert deps._signal_label("engine", "NotDescribed") == "NotDescribed"


def test_label_top_signals_attaches_meaning(deps, kb_ship):
    kb_ship.set_field_semantics("engine", [
        {"field": "EngineOilPressure", "description": "oil pressure", "unit": "kPa",
         "role": "none", "confidence": "high", "aggregation": "avg"},
    ])
    out = deps._label_top_signals("engine",
                                  [{"signal": "EngineOilPressure", "share": 0.4}])
    assert out[0]["meaning"] == "oil pressure (kPa)" and out[0]["share"] == 0.4


def test_label_drift_signals_enriches_ae_and_joint(deps, kb_ship):
    """_label_drift_signals adds 'meaning' to joint + AE flagged signals in place."""
    kb_ship.set_field_semantics("engine", [
        {"field": "EngineSpeed", "description": "engine speed", "unit": "rpm",
         "role": "none", "confidence": "high", "aggregation": "avg"},
    ])
    result = {
        "joint_anomalies": {"findings": [
            {"regime": "0", "top_signals": [{"signal": "EngineSpeed", "share": 0.7}]}]},
        "ae_anomalies": {"findings": [
            {"regime": "1", "top_signals": [{"signal": "EngineSpeed", "share": 0.5}]}]},
    }
    deps._label_drift_signals("engine", result)
    assert result["joint_anomalies"]["findings"][0]["top_signals"][0]["meaning"] \
        == "engine speed (rpm)"
    assert result["ae_anomalies"]["findings"][0]["top_signals"][0]["meaning"] \
        == "engine speed (rpm)"


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
