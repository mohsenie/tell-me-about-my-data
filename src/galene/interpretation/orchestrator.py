"""Chat orchestrator — the single conversational surface.

Each user turn:
  1. scope-gate (refuse off-domain),
  2. classify intent (LLM),
  3. check PREREQUISITES for that intent; if missing, tell the user clearly and
     offer to run the missing phase,
  4. resolve parameters (signals via field-semantics, window) — data-driven,
  5. dispatch to a deterministic function or LLM reasoning,
  6. reply, keep history.

The LLM routes + interprets + resolves; deterministic code computes. No hardcoded
signal names — "fuel"/"rpm" are resolved from the data + field semantics.
"""
from __future__ import annotations

import json

from .provider import LLMProvider
from .knowledge import KnowledgeBase
from .scope import check_scope, REFUSAL
from .resolve import resolve_signal, resolve_aggregation


INTENTS = ["capabilities", "describe_fields", "value", "trend", "plot",
           "relationship", "reasoning", "correction", "position", "nearby",
           "voyage", "efficiency", "distance", "between", "geo", "anomaly",
           "summarize", "regimes", "actors", "smalltalk"]


class _NeedsClarification(Exception):
    """Raised by a capability when a term is ambiguous; carries the question."""
    def __init__(self, question):
        super().__init__(question)
        self.question = question


_INTENT_INSTRUCTION = "You classify user messages for an asset-telemetry assistant."
_PARAM_INSTRUCTION = "You extract structured query parameters as JSON."

_INTENT_SYSTEM = """Classify the user's message about asset telemetry into ONE
intent label (reply with ONLY the label). Read carefully — a signal NAME appearing
does NOT make it a relationship question.

- capabilities: what fields/signals exist, what they can query/ask, or general
    "help" / "what can you do" requests. ALSO the first-contact ORIENTATION
    question — "what can you tell me about this <asset>?", "what do you know about
    it?", "tell me about this ship", "what have you got on it?" — i.e. "what's
    here / what can I ask", with NO notability or concern framing. (If they ask
    what's NOTABLE / interesting / worth attention / a summary of findings, that
    is summarize, not this.)
- describe_fields: asking what the fields MEAN — their definitions/units. e.g.
    "what do the fields mean?", "what does X measure?", "what units is X in?".
    (Merely asking WHICH fields EXIST is capabilities, not describe_fields.) NOT
    when the user wants an actual NUMBER for a specific signal (that is value),
    even if they use the word "normal"/"typical".
- value: wants a NUMBER for a specific signal — total/average/max/min, OR the
    typical/normal/current level of a NAMED signal. e.g. "average fuel rate over 2
    days", "max coolant temp", "total fuel last week", "what's the normal/typical
    frequency_x", "current engine speed". If a specific signal is named and a
    quantity (even "normal"/"typical"/"usual" value) is asked for, it's value, not
    describe_fields. A plain time window ("over/for the past N days") is STILL
    value — one number for that window. Default aggregation is the mean.
- trend: asking whether/how a signal CHANGED over time — "has X gone up/down?",
    "is X higher than before?", "how has X changed since ...", direction over time.
- plot: ONLY if they (a) explicitly ask for a chart/graph/plot/visual, OR (b) ask
    for a SCATTER of two signals ("scatter of X vs Y" is ALWAYS plot, even though a
    scatter shows a correlation — the word scatter/chart means they want the
    picture, not the number), OR (c) ask to BUCKET/BIN/GROUP a value by a TIME
    interval — a VALUE PER TIME INTERVAL, like "average X EVERY 4 hours", "X PER
    hour", "bucket X hourly", "hourly average". Without an explicit chart word or a
    per-TIME-interval phrase, it is value, not plot. If the bucketing is per
    DISTANCE ("every 20km", "per mile") it's efficiency, not plot.
- relationship: asking what CORRELATES with what / the correlation/link BETWEEN
    signals. Must be about a relationship between signals, not one signal's value.
- reasoning: asking WHY signals relate / an explanation or root cause.
- correction: telling you the real cause/reason for a relationship (disagreeing).
- position: asking WHERE the asset/ship is now or at some time ("where is the
    ship", "current position/location", "where was it at ...").
- nearby: asking what OTHER vessels/objects were near/around at a time ("what
    ships were nearby at 13:00", "any vessels around us", "who was close").
- distance: asking HOW FAR the asset travelled / total distance covered over a
    period — "how many km did the ship travel", "distance travelled last 2 days",
    "how far did we go". This is the LENGTH OF THE TRACK, not a signal value.
- geo: asking how a signal VARIES BY LOCATION / whether it differs across places
    — "is vibration higher in some locations", "how does X vary by location/area/
    region", "relationship between <signal> and location", "where is it roughest".
    A signal (or a concept like 'vibration' = all its axes) aggregated over
    geographic areas. The tell is 'by location/area/region/where' or '<signal> and
    location'. NOT a per-distance metric (that's efficiency).
- between: asking the DISTANCE BETWEEN TWO NAMED OTHER entities/vessels at a time
    — "distance between JORO and HARRIS", "how far apart were X and Y at 15:00".
    Two named vessels + "between/apart". NOT a signal correlation (that's
    relationship), NOT proximity to us (that's nearby).
- efficiency: asking for "X PER Y" where Y is a DISTANCE or another QUANTITY
    (not time). Covers: (a) consumption per distance — "fuel per km", "litres per
    20km", "fuel economy"; (b) any signal averaged per distance — "velocity_z per
    10km", "plot vibration every 10km"; (c) a ratio per another quantity — "fuel
    per operating hour", "fuel per MWh", "X per revolution". The defining feature
    is "per <distance-or-quantity>"/"every <distance>", INCLUDING when a chart is
    requested. If the bucketing is per <TIME> (per hour / every 4 hours) it's
    plot/trend, NOT efficiency.
- regimes: asking about the OPERATING MODES / usage patterns the asset runs in —
    "list usage patterns", "what operating modes are there", "what regimes",
    "how does the engine operate", "what states does it run in", "usage
    breakdown". Wants the discovered operating regimes + how much time in each,
    NOT the list of fields (capabilities) and NOT a per-signal value.
- summarize: an OPEN-ENDED request for what is NOTABLE / worth attention —
    "what is notable", "summarize my data", "what should I pay attention to",
    "give me an overview", "anything interesting", "what stands out". Wants a
    short prioritized summary of FINDINGS (behavior/regimes/drift), NOT a single
    metric and NOT a specific drift check. The tell is notability/attention
    framing. (A neutral "what can you tell me about the ship / what data is here"
    with NO notability framing is capabilities, not this. If they ask
    specifically "has X drifted / is it normal", that's anomaly, not this.)
- anomaly: asking whether anything has DRIFTED / changed / is abnormal / wrong /
    degrading vs normal — "has anything changed?", "is the engine behaving
    normally?", "any anomalies?", "check for drift", "what's different from the
    baseline?". About deviation from normal, not a single value or a correlation.
- voyage: asking HOW MUCH of a quantity was CONSUMED/USED going between two
    points — a single total for the trip. e.g. "how much fuel from A to B", "fuel
    used going from <lat,lon> to <lat,lon>". Defining features: (a) a FROM and a
    TO, AND (b) a consumed TOTAL ("how much ... used/consumed"). If the user asks
    to PLOT/BUCKET a signal between two points (a chart, not a single total),
    that is plot — the from/to just restrict the window, it's not a voyage total.
- actors: managing the directory of PEOPLE/actors — adding, listing, or removing
    a person and their role/description. e.g. "add Andrew as the engine room
    technician", "who is the captain?", "list the crew / people / actors",
    "remove Andrew", "delete the actor Maria". About WHO (people to notify), NOT
    about telemetry signals, values, or relationships between signals.
- smalltalk: greetings or meta questions about the assistant itself.

If the request is genuinely ambiguous between two intents (e.g. you cannot tell
if they want a single number vs a chart, or a value vs a correlation), reply with
"ambiguous" instead of guessing.

Ongoing conversation:
{history}
Message: {message}
Intent:"""

MODES = ("general", "analyst", "expert")


def _norm_mode(m):
    """Normalize a mode name/synonym to one of MODES. Default: general.

    Three modes:
      general  — plain-language, business/operations framing, no jargon (DEFAULT).
      analyst  — concise + concrete: which signals/couplings changed, what to check.
      expert   — full raw detail, verbatim (no LLM reframing; the numbers as-is).
    """
    m = (m or "").strip().lower()
    alias = {
        # general (plain / business / operations)
        "business": "general", "operations": "general", "ops": "general",
        "manager": "general", "captain": "general", "exec": "general",
        "operator": "general", "plain": "general", "simple": "general",
        "basic": "general",
        # analyst (technician / engineer / mid-detail)
        "technician": "analyst", "tech": "analyst", "engineer": "analyst",
        "maintenance": "analyst", "mechanic": "analyst", "analysis": "analyst",
        # expert (full raw detail)
        "data": "expert", "detailed": "expert", "detail": "expert",
        "raw": "expert", "full": "expert", "everything": "expert",
    }
    if m in MODES:
        return m
    return alias.get(m, "general")


def _detect_mode_switch(message):
    """If the message asks to switch presentation mode ('switch to business mode',
    'use technician view', 'operator mode'), return the new mode, else None."""
    low = (message or "").lower()
    if not any(w in low for w in ("mode", "view", "as a", "for a", "switch",
                                  "talk to me", "explain")):
        return None
    import re as _re
    for token in _re.findall(r"[a-z]+", low):
        m = _norm_mode(token)
        # only accept if the token itself was a known mode/alias (not the default
        # fallback firing on an unrelated word)
        if token in MODES or token in {
                "business", "operations", "ops", "manager", "captain", "exec",
                "operator", "plain", "simple", "basic",
                "technician", "tech", "engineer", "maintenance", "mechanic", "analysis",
                "data", "detailed", "detail", "raw", "full", "everything"}:
            return m
    return None


_MODE_PERSONA = {
    "general": (
        "a GENERAL business/operations user who does NOT know sensor/statistics "
        "jargon. Be SHORT (1-3 sentences). Translate technical findings into "
        "operational meaning: an operating-mode/regime change -> how the asset is "
        "being USED (e.g. 'spending more time stationary', 'taking a route it "
        "hasn't before', 'staying in one place longer than usual'); a relationship/"
        "structure drift -> 'something in how it's behaving changed, worth a look'. "
        "NEVER use the words regime, dcor, correlation, edge, coefficient, "
        "threshold. Lead with whether there's anything to worry about."),
    "analyst": (
        "a TECHNICAL ANALYST / ENGINEER. Be concise but concrete: name the specific "
        "signals and couplings that changed and in which operating mode, and suggest "
        "what to CHECK. Keep the actionable detail, drop raw statistics tables."),
    # 'expert' has no persona entry: it's the verbatim full-detail passthrough.
}
_FRAME_SYSTEM = (
    "You are continuing a CONVERSATION with {persona}\n"
    "You are given the recent conversation and a fresh analyst answer. Re-express "
    "the answer for this person AS THE NEXT TURN in the conversation.\n"
    "STRICT RULES:\n"
    "- Use ONLY the facts in the analyst answer — never add, invent, or change a "
    "number, signal, or finding. Rewording for the audience, not re-analyzing.\n"
    "- NEVER invent a NEW percentage or statistic by combining/deriving from the "
    "given numbers (e.g. do NOT sum regime fractions into a 'percent of time the "
    "engine runs'). Only restate numbers that are literally present.\n"
    "- NEVER give an operating mode / regime an invented MEANING or name. The "
    "analyst answer says modes are distinguished by signals and that their "
    "meaning (idle, cruise, load, startup, ...) is the USER's to assign — DO NOT "
    "assign it yourself. Say 'operating mode 1', or describe it by the signals "
    "given ('the higher-oil-pressure mode'), never 'the load/idle/startup mode'.\n"
    "- Do NOT reinterpret WHAT a number measures. 'X% of the time in mode N' means "
    "the split of the RECORDED data across modes — it is NOT 'X% of the time the "
    "engine is on/running/used'. Keep the original meaning exactly.\n"
    "- ANSWER THE SPECIFIC QUESTION asked. If they asked about fuel, lead with the "
    "fuel-relevant part; if about the engine's health, the engine part. Don't dump "
    "the whole picture when they asked something narrow.\n"
    "- DON'T REPEAT what you already told them earlier in the conversation. If a "
    "fact (e.g. the long stop) was already stated, refer to it briefly ('as noted, "
    "the extended stop...') or omit it — do NOT restate it in full each time.\n"
    "- Sound like a continuing dialogue, not a fresh standalone report. Be concise.\n"
    "- Keep the 'observed change, not a stated cause' honesty. If the answer is "
    "already a simple number/value, return it unchanged.")
_FRAME_USER = ("Recent conversation:\n{history}\n\nUser's new question: {question}\n\n"
               "Fresh analyst answer to re-express (for THIS question, without "
               "repeating what was already said):\n{answer}\n\nYour reply:")

_CLARIFY_SYSTEM = "You write ONE short clarifying question for a telemetry assistant."
_CLARIFY_USER = """The user's request is ambiguous: "{message}"
Ask ONE short question to disambiguate what they want (e.g. a single value vs a
chart over time, or which of two signals). One sentence, no preamble."""

_PARAM_SYSTEM = """Extract query parameters from the user's request as JSON.
Available signals: {signals}
Return ONLY JSON with keys:
  "signal": best-matching signal for the main quantity (or ""),
  "signal_x","signal_y": for a scatter/correlation of two signals (or ""),
  "aggregation": one of avg/min/max/sum/median/stddev/count/integral (or ""),
  "bucket": time bin like "1h","4h","1d" for trends (or ""),
  "days": integer number of days back if stated (or null),
  "date_from","date_to": explicit calendar range as ISO dates "YYYY-MM-DD" if the
      user gave one ("over 2026-09-03 to 2026-09-04", "between Sep 3 and Sep 4",
      "on 2026-09-04"), else "". For a single day, set both to that date.
  "plot_kind": "trend" or "scatter" if a plot (or ""),
  "from_lat","from_lon": start-point coordinates for a voyage, as numbers, if the
      user gave coordinates (else null),
  "to_lat","to_lon": end-point coordinates for a voyage, as numbers, if given
      (else null),
  "from_place","to_place": start/end PLACE NAMES for a voyage if the user named
      places instead of coordinates (e.g. "Gdynia", "Scotland"), else "".
  "per_distance_km": for an efficiency question ("per 20km", "per mile"), the
      distance in KILOMETRES as a number (per km -> 1; per 20km -> 20; per mile ->
      1.60934; per nautical mile -> 1.852), else null.
  "per_distance_label": the distance unit as the USER said it, for display
      ("20km" -> "20 km"; "mile" -> "mile"; "nautical mile" -> "nautical mile"),
      else "".
  "per_signal": for an "X per Y" ratio where Y is a SIGNAL/quantity that is NOT a
      distance (e.g. "fuel per operating hour" -> "operating hour"; "fuel per MWh"
      -> "MWh"; "X per revolution" -> "revolution"). Empty "" if the denominator
      is distance or there is no per-Y.
Message: {message}"""

_ACTOR_SYSTEM = """Extract an ACTOR-directory operation from the user's message as
JSON. The directory holds PEOPLE (name + a role/description). Return ONLY JSON:
  "op": one of "add" / "list" / "delete" (or "" if unclear),
  "name": the person's name for add/delete (or ""),
  "description": their role/description for add, e.g. "engine room technician"
      or "captain" (or ""),
  "contact": an email/handle if the user gave one (or "").
Examples:
  "add Andrew as the engine room technician" -> {{"op":"add","name":"Andrew","description":"engine room technician","contact":""}}
  "who is the captain / list the crew / show actors" -> {{"op":"list","name":"","description":"","contact":""}}
  "remove Andrew / delete the actor Maria" -> {{"op":"delete","name":"Andrew","description":"","contact":""}}
Message: {message}"""


class Orchestrator:
    """Holds conversation state + injected capability callbacks so it stays
    testable and decoupled from the CLI."""

    def __init__(self, source: str, asset_type: str, provider: LLMProvider,
                 kb: KnowledgeBase, deps, mode: str = "general"):
        self.source = source      # default/home source the chat started on
        self._active = source     # source used for the CURRENT turn (routing)
        self.asset_type = asset_type
        self.provider = provider
        self.kb = kb
        self.deps = deps          # capability object (see CLI wiring)
        self.mode = _norm_mode(mode)  # presentation mode (operator/technician/analyst)
        self.history: list[dict] = []
        self._described_attempted = set()  # sources we've auto-described (once each)
        self._clarifying = None   # original message awaiting a disambiguation reply
        self._last_intent = None  # intent of the previous turn (for 'why?' follow-up)

    def _hist(self) -> str:
        return "\n".join(f"{h['role']}: {h['text']}" for h in self.history[-6:])

    def _wants_all_sources(self, message: str) -> bool:
        """True if a broad, whole-vessel question ('my data', 'all/each/every
        source', 'across sources', 'anywhere') that should sweep EVERY source
        rather than the single active one — UNLESS a specific source is named.
        Shared by summarize / capabilities / regimes / anomaly for consistency."""
        low = (message or "").lower()
        broad = any(w in low for w in (
            "all source", "all data source", "each source", "every source",
            "all sources", "across sources", "per source", "my data", "the data",
            "any data", "anywhere", "all of them", "everything", "any source",
            "whole vessel", "entire vessel", "overall", "overview"))
        if not broad:
            return False
        named = any(s.lower() in low for s in self.deps.all_sources())
        return not named

    def _is_orientation(self, message: str) -> bool:
        """True for a first-contact 'tell me about this asset / what do you know
        about it / what can you tell me about the ship' — a whole-vessel ORIENTATION
        that should sweep EVERY source's fields, not just the home source. Distinct
        from _wants_all_sources (which keys off 'my data'/'all sources' wording);
        this catches the natural 'about the <asset>' framing. UNLESS a specific
        source is named (then keep it single)."""
        low = (message or "").lower()
        orient = any(p in low for p in (
            "about this", "about the", "about it", "about our", "about your",
            "know about", "tell me about", "what have you got",
            "what do you have on"))
        if not orient:
            return False
        named = any(s.lower() in low for s in self.deps.all_sources())
        return not named

    @staticmethod
    def _is_why_followup(message: str) -> bool:
        """A short causal follow-up ('why?', 'what could cause that', 'how come',
        'what does that mean') that should EXPLAIN the previous anomaly answer
        rather than start a fresh discovery-reasoning query. Deterministic guard —
        the LLM otherwise routes a bare 'why?' to generic reasoning and loses the
        drift context."""
        low = (message or "").strip().lower().rstrip("?.! ")
        if not low or len(low.split()) > 8:
            return False
        triggers = ("why", "how come", "what could cause", "what causes",
                    "what caused", "what might cause", "what does that mean",
                    "what does this mean", "explain", "elaborate", "cause")
        return any(low == t or low.startswith(t + " ") or t in low
                   for t in triggers)

    # quantity words that signal "give me a NUMBER" (not a field-meaning request).
    # NOTE: "mean" is deliberately excluded — it's overloaded ("what does X mean")
    # and "average"/"avg" cover the statistical sense.
    _VALUE_WORDS = ("average", "avg", "maximum", "minimum", "total", "median",
                    "normal", "typical", "usual", "current", "how much",
                    "how many", "reading", "readings", "max", "min", "sum")

    def _looks_like_value(self, message: str) -> bool:
        """True if the message NAMES an exact signal column AND asks for a
        quantity — a deterministic 'this is a value request' signal used to
        override a borderline LLM intent. Column names come from the data (all
        sources), so nothing is hardcoded to a specific signal."""
        import re as _re
        low = (message or "").lower()
        # whole-word match so "mean"/"min" don't match inside other words
        if not any(_re.search(rf"\b{_re.escape(w)}\b", low) for w in self._VALUE_WORDS):
            return False
        try:
            for s in self.deps.all_sources():
                for col in self.deps.signals(s):
                    if col.lower() in low:      # exact column name mentioned
                        return True
        except Exception:
            pass
        return False

    def send(self, message: str) -> str:
        # Presentation-mode switch ("switch to business mode") — handled before
        # anything else; changes ONLY how answers are framed, never the data.
        sw = _detect_mode_switch(message)
        if sw and sw != self.mode:
            self.mode = sw
            _desc = {"general": "short, plain, business-focused",
                     "analyst": "concise + concrete: which signals changed, what to check",
                     "expert": "full detail with the raw numbers"}[sw]
            reply = (f"Switched to {sw} mode — I'll tailor how I explain things "
                     f"({_desc}). The underlying analysis is unchanged.")
            self.history.append({"role": "user", "text": message})
            self.history.append({"role": "assistant", "text": reply})
            return reply

        # "show <source> detail" FOLLOW-UP: after the all-sources anomaly roll-up,
        # a request to expand one source's full breakdown. Deterministic.
        if getattr(self, "_last_intent", None) == "anomaly":
            low = message.lower()
            if any(w in low for w in ("detail", "details", "breakdown", "full",
                                      "expand", "more on", "show me")):
                named = next((s for s in self.deps.all_sources() if s.lower() in low),
                             None)
                if named or "detail" in low or "breakdown" in low:
                    target = named or self.source
                    reply = self._frame("anomaly", message,
                                        self.deps.anomaly_detail(target))
                    self.history.append({"role": "user", "text": message})
                    self.history.append({"role": "assistant", "text": reply})
                    self._last_intent = "anomaly"
                    return reply

        # "why?" FOLLOW-UP: if the previous answer was an anomaly/summary result
        # and the user asks a short causal follow-up, EXPLAIN the detected change
        # (grounded in the structured findings + KB + docs) instead of starting a
        # fresh reasoning query that would lose the drift context.
        if (getattr(self, "_last_intent", None) in ("anomaly", "summarize")
                and self._is_why_followup(message)):
            explanation = self.deps.explain_drift(message)
            if explanation:
                reply = self._frame("reasoning", message, explanation)
                self.history.append({"role": "user", "text": message})
                self.history.append({"role": "assistant", "text": reply})
                self._last_intent = "reasoning"
                return reply

        if not check_scope(message, self.provider, context=self._hist() or None).in_scope:
            return REFUSAL

        raw_intent = self.provider.complete(
            _INTENT_INSTRUCTION,
            _INTENT_SYSTEM.format(history=self._hist() or "(none)",
                                  message=message)).strip().lower()

        # Ambiguous -> ask a clarifying question instead of guessing. Remember the
        # original message so the user's next reply is interpreted with it.
        if "ambiguous" in raw_intent and not any(
                i in raw_intent for i in INTENTS if i != "smalltalk"):
            q = self.provider.complete(
                _CLARIFY_SYSTEM, _CLARIFY_USER.format(message=message)).strip()
            self._clarifying = message
            self.history.append({"role": "user", "text": message})
            self.history.append({"role": "assistant", "text": q})
            return q

        intent = next((i for i in INTENTS if i in raw_intent), "reasoning")

        # DETERMINISTIC GUARD (source data coverage): "for what % of the last trip
        # was the engine USED / did it REPORT data / what's the ECU's uptime" is a
        # DATA-COVERAGE question, not on/off and not a regime split. Route it to
        # source_coverage (measures reporting coverage + annotates no-data gaps
        # moving/stationary, never asserts 'off'). Asset-neutral: keys off
        # coverage/used/report/uptime + a trip word, on a NAMED source.
        _lowc = message.lower()
        _cov_word = any(w in _lowc for w in ("coverage", "report data", "reported data",
                                             "reporting", "uptime", "was used", "in use",
                                             "was the engine used", "did the engine",
                                             "how much of the", "what percentage",
                                             "what % ", "% of the"))
        _trip_word = any(w in _lowc for w in ("trip", "journey", "voyage", "leg",
                                              "last"))
        _named_src = next((s for s in self.deps.all_sources()
                           if s.lower() in _lowc), None)
        # also match an engine-ish word -> the source that owns rate/engine signals
        if _cov_word and _trip_word and (_named_src or "engine" in _lowc
                                         or "ecu" in _lowc or "motor" in _lowc):
            cov_src = _named_src or self.deps.route_source(
                "value", message, {}, self.source)
            reply = self._frame("coverage", message,
                                self.deps.source_coverage(cov_src, message))
            self.history.append({"role": "user", "text": message})
            self.history.append({"role": "assistant", "text": reply})
            self._last_intent = "coverage"
            return reply

        # DETERMINISTIC GUARD: "how many nearby <entities> per <distance>" asks to
        # COUNT nearby entities bucketed along the track — routed to the real
        # per-distance capability. ASSET-NEUTRAL: the trigger is proximity
        # ("nearby"/"around"/"near me") + per-distance + a count, NOT a hardcoded
        # 'ship'/'vessel' word (a truck fleet asks 'other trucks/vehicles nearby').
        # Without this the "per km" phrasing pulls it into efficiency and fabricates
        # a fuel answer.
        import re as _re0
        _low0 = message.lower()
        _about_nearby = ("nearby" in _low0 or "near me" in _low0
                         or "near us" in _low0 or "around us" in _low0
                         or _re0.search(r"\b(other|around)\b", _low0) and
                         _re0.search(r"\b(ship|vessel|truck|vehicle|craft|unit|"
                                     r"boat|car|aircraft|plane|entit)", _low0))
        _per_distance = bool(_re0.search(
            r"(per|every|each)\s*\d*\s*(km|kilomet|mile|nm|nautical|m\b)", _low0))
        _counting = any(w in _low0 for w in ("how many", "number of", "count", "how much"))
        if _about_nearby and _per_distance and _counting:
            reply = self._frame("nearby", message,
                                self.deps.nearby_per_distance(message))
            self.history.append({"role": "user", "text": message})
            self.history.append({"role": "assistant", "text": reply})
            self._last_intent = "nearby"
            return reply

        # DETERMINISTIC GUARD (compound trip+consumption): a question that pairs a
        # TRIP word ("voyage/trip/leg") with a fuel/consumption/total word is a
        # voyage-consumption question — even when the LLM calls it "ambiguous"
        # because it also asks "is that normal?". Force the voyage intent: that path
        # resolves fuel->the RATE signal (no level/temp ambiguity) AND appends the
        # per-distance normality verdict. Fixes the compound question routing to the
        # generic resolver (which asks "which fuel?"). Data-agnostic phrasing check.
        _low = message.lower()
        if (any(w in _low for w in ("voyage", "trip", "leg", "journey", "passage"))
                and any(w in _low for w in ("fuel", "consum", "burn", "total",
                                            "gas", "diesel", "how much"))):
            intent = "voyage"

        # DETERMINISTIC GUARD: the value/describe_fields boundary on phrasing like
        # "normal frequency_x" is genuinely borderline and the LLM flips between
        # runs. If the message NAMES an exact signal column AND asks for a
        # quantity, it's a value request — override a describe_fields/reasoning
        # guess. Data-driven: checks the actual column names, nothing hardcoded.
        if intent in ("describe_fields", "reasoning") and self._looks_like_value(message):
            intent = "value"

        # DETERMINISTIC GUARD: an explicit chart word means a chart, never a
        # relationship/reasoning TEXT answer ("scatter of X vs Y" flips otherwise).
        # Leave per-distance charts (efficiency) alone; only rescue relationship/
        # reasoning -> plot.
        _low = message.lower()
        if (intent in ("relationship", "reasoning")
                and any(w in _low for w in ("scatter", "plot", "graph", "chart",
                                            "draw", "visual"))):
            intent = "plot"

        # If we just asked a clarifying question, the user's reply is the
        # disambiguation. Prepend it so it takes PRIORITY (e.g. reply
        # "EngineFuelRate" overrides the ambiguous "fuel" in the original).
        if getattr(self, "_clarifying", None):
            message = f"{message} (for: {self._clarifying})"
            self._clarifying = None

        # VESSEL-SCOPED ROUTING: pick the source that can answer this question
        # rather than staying locked to the source the chat started on. For
        # value/plot/trend we peek at params (which signal) to find the source
        # that actually has that signal.
        params = None
        if intent in ("value", "plot", "trend", "voyage", "efficiency", "distance",
                      "geo"):
            params = self._extract_params(message)
        self._active = self.deps.route_source(
            intent, message, params or {}, self.source)

        try:
            reply = self._dispatch(intent, message, params)
        except _NeedsClarification as e:
            # deterministic signal-ambiguity -> ask, remember the original request
            self._clarifying = message
            self.history.append({"role": "user", "text": message})
            self.history.append({"role": "assistant", "text": e.question})
            return e.question
        # PRESENTATION MODE: reframe INTERPRETIVE answers for the audience
        # (operator/technician). Analyst = unchanged. Never alters the numbers.
        reply = self._frame(intent, message, reply)
        self.history.append({"role": "user", "text": message})
        self.history.append({"role": "assistant", "text": reply})
        self._last_intent = intent
        return reply

    # intents whose answers EXPLAIN (worth reframing per audience). Deterministic
    # value/plot/etc. are the same in every mode, so they're left untouched.
    _INTERPRETIVE = ("summarize", "anomaly", "regimes", "reasoning", "relationship")

    def _frame(self, intent, message, reply):
        """Reframe an interpretive answer for the current mode. operator = short,
        plain, business/behavior wording (no jargon like regime/dcor/edge);
        technician = which signals/couplings + likely component/action; analyst =
        unchanged. Presentation only — the input facts/numbers are not altered,
        the LLM just re-expresses them for the audience."""
        if self.mode == "expert" or intent not in self._INTERPRETIVE:
            return reply
        if not reply or reply.strip().startswith("I don't have a known-good"):
            return reply   # don't reframe the baseline-setup prompt
        persona = _MODE_PERSONA[self.mode]
        try:
            out = self.provider.complete(
                _FRAME_SYSTEM.format(persona=persona),
                _FRAME_USER.format(history=self._hist() or "(start of conversation)",
                                   question=message, answer=reply),
                max_tokens=500).strip()
            return out or reply
        except Exception:
            return reply

    # --- dispatch with prerequisite checks ---
    def _dispatch(self, intent: str, message: str, params: dict | None = None) -> str:
        d = self.deps
        src = self._active   # routed source for this turn (may differ from home)
        if intent == "capabilities":
            # broad "my data / all sources" OR a first-contact "tell me about the
            # ship" orientation -> sweep every source; a named source stays single.
            if self._wants_all_sources(message) or self._is_orientation(message):
                return d.capabilities_all()
            return d.capabilities(src)

        if intent == "describe_fields":
            notice = ""
            if not d.has_field_semantics(src):
                # auto-run: the user asked for field meanings, so just produce them
                notice = self._auto_describe_fields(src)
            return notice + d.field_descriptions(src)

        if intent == "actors":
            return self._dispatch_actors(message)

        if intent in ("relationship", "reasoning", "correction"):
            # correction doesn't need discovery; relationship/reasoning do -> auto-run
            notice = ""
            if intent != "correction" and not d.has_discovery(src):
                notice = self._auto_discover(src)
            if intent == "relationship":
                return notice + d.relationship_lookup(src, message)
            if intent == "correction":
                return d.capture_correction(src, message)
            return notice + d.reason(src, message, self.history)

        if intent in ("position", "nearby", "between"):
            # Spatial queries resolve columns by ROLE (lat/lon/identifier/name)
            # from field-semantics. Auto-describe this source's fields once so the
            # roles exist — makes it fully data-agnostic (no reliance on column
            # NAME conventions). Falls back to name hints if description fails.
            notice = ""
            if (not d.has_field_semantics(src)
                    and src not in self._described_attempted):
                self._described_attempted.add(src)
                notice = self._auto_describe_fields(src)
            if intent == "position":
                return notice + d.position(src, message)
            if intent == "between":
                return notice + d.distance_between(src, message)
            return notice + d.nearby(src, message)

        if intent == "voyage":
            # A voyage query spans sources: the WINDOW comes from the position
            # track, the QUANTITY (fuel/consumption) is integrated on the source
            # that owns that signal (src, routed by signal ownership). Auto-
            # describe so the rate's unit/aggregation are known.
            notice = ""
            if (not d.has_field_semantics(src)
                    and src not in self._described_attempted):
                self._described_attempted.add(src)
                notice = self._auto_describe_fields(src)
            if params is None:
                params = self._extract_params(message)
            # LIST voyages ("what trips has it made", "list voyages/legs") — a
            # trip-listing question, not a consumption total. Deterministic guard.
            low = message.lower()
            if (any(w in low for w in ("voyage", "trip", "leg", "journey", "passage"))
                    and any(w in low for w in ("list", "what ", "which", "how many",
                                               "show", "detected"))
                    and not any(w in low for w in ("fuel", "consum", "used", "burn"))):
                return notice + d.list_voyages(src, message)
            return notice + d.voyage(src, message, params)

        if intent == "efficiency":
            # consumption per distance: rate integral / track distance. Needs
            # field-semantics for the rate + a position source for distance.
            notice = ""
            if (not d.has_field_semantics(src)
                    and src not in self._described_attempted):
                self._described_attempted.add(src)
                notice = self._auto_describe_fields(src)
            if params is None:
                params = self._extract_params(message)
            # "X per Y" where Y is another SIGNAL (not distance) -> general ratio;
            # otherwise the per-distance path (value or chart).
            if (params.get("per_signal") or "").strip():
                return notice + d.per_ratio(src, message, params)
            return notice + d.efficiency(src, message, params)

        if intent == "distance":
            if params is None:
                params = self._extract_params(message)
            return d.distance_travelled(src, message, params)

        if intent == "geo":
            # signal-by-location. Concept resolution ("vibration" = its axes) needs
            # field-semantics, so auto-describe the source once.
            notice = ""
            if (not d.has_field_semantics(src)
                    and src not in self._described_attempted):
                self._described_attempted.add(src)
                notice = self._auto_describe_fields(src)
            if params is None:
                params = self._extract_params(message)
            return notice + d.signal_by_location(src, message, params)

        if intent == "anomaly":
            # Drift detection needs a KNOWN-GOOD baseline, which only the operator
            # can designate (which period was healthy). So unlike discovery, we
            # can't auto-run it — we ask for the window if none is set.
            # BROAD scope ("my data", "all/each/every source", "anywhere") -> sweep
            # every source; a specifically-named source stays single.
            if self._wants_all_sources(message):
                return d.detect_anomaly_all(message)
            return d.detect_anomaly(src, message)

        if intent == "summarize":
            # Open-ended overview. Auto-runs discovery inside _notable_facts;
            # folds in drift only if a baseline exists. No baseline required.
            # Broad "overview of my data / all sources" -> summarize EVERY source.
            if self._wants_all_sources(message):
                return d.summarize_all(message)
            notice = ""
            if (not d.has_field_semantics(src)
                    and src not in self._described_attempted):
                self._described_attempted.add(src)
                notice = self._auto_describe_fields(src)
            return notice + d.summarize(src, message)

        if intent == "regimes":
            # "for ALL sources / each / every source" -> describe every one.
            if self._wants_all_sources(message):
                return d.describe_regimes_all(message)
            # Operating modes come from discovery -> auto-run it if missing.
            notice = ""
            if not d.has_discovery(src):
                notice = self._auto_discover(src)
            return notice + d.describe_regimes(src, message)

        if intent in ("value", "plot", "trend"):
            # Auto-describe fields once (so values carry UNITS and the correct
            # aggregation, e.g. rate -> integral). Runs inline, no confirmation.
            notice = ""
            if (not d.has_field_semantics(src)
                    and src not in self._described_attempted):
                self._described_attempted.add(src)
                notice = self._auto_describe_fields(src)
            if params is None:
                params = self._extract_params(message)
            if intent == "value":
                return notice + d.compute_value(src, message, params)
            if intent == "trend":
                return notice + d.compare_trend(src, message, params)
            return notice + d.make_plot(src, message, params)

        # smalltalk / fallback
        return ("I analyze this asset's telemetry — ask about fields, values, "
                "plots, correlations, or why signals relate.")

    # --- auto-run prerequisite phases (no confirmation; inline notice) ---
    def _auto_discover(self, src: str) -> str:
        """Run discovery for `src` right now and return a short notice to prepend
        to the answer. The question that triggered this needs the relationship
        graph, so we just build it rather than asking permission."""
        self.deps.run_discovery(src)
        return f"(first analyzed '{src}' to find its relationships)\n\n"

    def _auto_describe_fields(self, src: str) -> str:
        """Describe fields for `src` inline so values carry units/aggregation.
        Returns a short notice (or a soft note if it couldn't run) to prepend."""
        try:
            msg = self.deps.run_describe_fields(src)
        except Exception:
            return ""   # never block the actual answer on field description
        return f"({msg})\n\n"

    def _extract_params(self, message: str) -> dict:
        # Use signals across ALL sources so a term can be extracted regardless of
        # which source ends up answering (vessel-scoped routing decides that next).
        available = []
        try:
            for s in self.deps.all_sources():
                for sig in self.deps.signals(s):
                    if sig not in available:
                        available.append(sig)
        except Exception:
            available = self.deps.signals(self.source)
        raw = self.provider.complete(
            _PARAM_INSTRUCTION,
            _PARAM_SYSTEM.format(signals=", ".join(available), message=message))
        try:
            import re
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            return json.loads(m.group(0)) if m else {}
        except Exception:
            return {}

    def _dispatch_actors(self, message: str) -> str:
        """Actor-directory CRUD (add/list/delete people). The LLM extracts the
        operation + fields; deterministic code performs it. Delete/resolve ask a
        clarifying question when a name is ambiguous (via NeedsClarification)."""
        raw = self.provider.complete(_PARAM_INSTRUCTION,
                                     _ACTOR_SYSTEM.format(message=message))
        try:
            import re
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            params = json.loads(m.group(0)) if m else {}
        except Exception:
            params = {}
        op = (params.get("op") or "").lower()
        if op == "add":
            return self.deps.add_actor(params.get("name", ""),
                                       description=params.get("description", ""),
                                       contact_label=params.get("contact") or None)
        if op == "delete":
            return self.deps.delete_actor(params.get("name", ""))
        # default / "list" (also the safe fallback for an unclear op)
        return self.deps.list_actors()
