"""Capability bridge for the chat orchestrator.

Implements the actions the orchestrator dispatches to (capabilities, values,
plots, relationship lookups, reasoning, corrections), plus PREREQUISITE checks
and the ability to run a missing phase on demand. Signal resolution is
data-driven (field semantics), not hardcoded.
"""
from __future__ import annotations

import glob
import json

import config
from galene import term
from galene.cli_helpers import resolve_window, present_globs
from galene.discovery.loader import load_numeric
from galene.discovery.relationships import build_per_regime_graphs
from galene.query import list_capabilities, describe_capabilities, aggregate, fuel_consumption
from galene.query.timeparse import parse_instant
from galene.query.plots import scatter, aggregated_series
from galene.interpretation.resolve import (
    resolve_signal, resolve_aggregation, candidate_signals)
from galene.interpretation.orchestrator import _NeedsClarification as NeedsClarification

import re
# words to ignore when scanning the user message for signal terms
_STOP = {"show", "me", "the", "a", "an", "plot", "chart", "graph", "of", "over",
         "time", "for", "past", "last", "days", "day", "hours", "hour", "week",
         "average", "avg", "mean", "max", "min", "total", "sum", "per", "every",
         "bucket", "and", "vs", "versus", "give", "get", "what", "was", "is",
         "how", "much", "many", "since", "ago", "it", "engine"}  # 'engine' too generic here


def _content_words(message: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]+", (message or "").lower())
            if w not in _STOP and len(w) > 2]


def _exact_signal(term: str, avail: list[str]) -> bool:
    t = term.strip().lower().replace(" ", "").replace("_", "")
    return any(c.lower().replace("_", "") == t for c in avail)


def _named_exact_signal(message: str, avail: list[str]) -> str | None:
    """Return a signal whose exact name appears as a token in the message."""
    low = (message or "").lower()
    for c in avail:
        if c.lower() in low:   # exact column name mentioned
            return c
    return None
_SUMMARY_SYSTEM = (
    "You write a SHORT 'what's notable' summary of an asset's telemetry for an "
    "operator. You are given already-COMPUTED facts (operating modes, strongest "
    "relationships, notable fields, and drift vs a known-good baseline if present). "
    "Rules: use ONLY the given facts — never invent signals, numbers, or causes. "
    "Prioritize: if 'behavioral_flags' are present, lead with them (they are the "
    "most important — e.g. stayed somewhere longer than usual) and ALWAYS include "
    "them; then drift/anomalies, then operating-mode usage, then the strongest "
    "structure. 3-6 short bullets, plain language, specific "
    "(name the signals/regimes). Report the OBSERVED change, NEVER assert a cause "
    "(a change can be a fault OR a legitimate operational change). End with one "
    "short caveat line to that effect.")
_SUMMARY_USER = "Source: {source}\nComputed facts (JSON):\n{facts}\n\nSummary:"

_EXPLAIN_SYSTEM = (
    "You explain an ALREADY-DETECTED change in an asset's telemetry to help a "
    "human decide what to check. You are given: the structured detector findings "
    "(which sensor relationships changed and in which operating mode, any change in "
    "the ORDER modes occur, and whether the change was sudden or gradual), plus "
    "expert-confirmed facts about this asset and any manual excerpts. "
    "RULES: (1) NEVER assert the cause — a change can be a developing fault OR a "
    "legitimate operational change (route, load, weather, deliberate operation); "
    "offer the plausible directions to CHECK, not a verdict. (2) Ground every "
    "statement in the given findings/facts — do not invent sensors or numbers. "
    "(3) When a manual/asset fact is relevant, use it and say so. (4) Be concise "
    "and concrete: name the specific signals/couplings that moved and suggest what "
    "to inspect. End with a one-line reminder that this is an observed change, not "
    "a confirmed cause.")
_EXPLAIN_USER = ("Asset type: {asset_type}\nDetector findings (JSON):\n{findings}\n\n"
                 "Expert-confirmed asset facts:\n{facts}\n\n"
                 "Relevant manual excerpts:\n{docs}\n\n"
                 "Question: {question}\n\nExplanation:")

from galene.interpretation.fields import describe_fields
from galene.interpretation.documents import DocumentIndex
from galene.interpretation.interpret import interpret_report
from galene.interpretation import apply_correction
from galene.reporting import describe_discovery


class ChatDeps:
    def __init__(self, asset_type, provider, kb, vessel=config.DEFAULT_VESSEL):
        self.asset_type = asset_type
        self.provider = provider
        self.kb = kb
        self.vessel = vessel

    # --- prerequisite checks ---
    def _discovery_path(self, source):
        return config.ARTIFACTS_DIR / f"discovery_{source}.json"

    def has_discovery(self, source):
        return self._discovery_path(source).exists()

    def has_field_semantics(self, source):
        return len(self.kb.field_semantics(source)) > 0

    def _discovery(self, source):
        return json.loads(self._discovery_path(source).read_text())

    def signals(self, source):
        globs = present_globs([config.source_glob(source, self.vessel)])
        caps = list_capabilities({source: config.source_glob(source, self.vessel)})
        return [s["name"] for s in caps.get(source, {}).get("signals", [])]

    def columns(self, source):
        """ALL column names for a source (any type), from the parquet schema.
        Needed for spatial queries: lat/lon/id are numeric but NAME/callsign are
        text, so they never appear in signals() (numeric-only)."""
        import duckdb
        globs = present_globs([config.source_glob(source, self.vessel)])
        if not globs:
            return []
        rel = f"read_parquet('{config.source_glob(source, self.vessel)}', union_by_name=true)"
        try:
            schema = duckdb.connect().execute(f"DESCRIBE SELECT * FROM {rel}").fetchall()
            return [r[0] for r in schema]
        except Exception:
            return []

    def _resolve_or_ask(self, term, source, avail, message=""):
        """Resolve a term to ONE signal, or raise NeedsClarification if ambiguous.

        Checks the ORIGINAL user message for ambiguous words (e.g. "fuel" matching
        3 signals) even if the LLM param-extractor already narrowed to one — the
        ambiguity must be judged on what the USER said, not the LLM's pick.
        """
        # If the message names an EXACT signal (e.g. a clarification reply
        # "EngineFuelRate"), that wins — skip ambiguity scanning entirely.
        exact = _named_exact_signal(message, avail)
        if exact:
            return exact
        # 1. deterministic ambiguity on the user's own words. Before declaring a
        # broad word ambiguous, see if the REST of the message narrows it: e.g.
        # "frequency" -> {frequency_x,_y,_z}, but "x axis" in the message picks
        # frequency_x. Only ask if it's still >1 after that narrowing.
        for word in _content_words(message):
            cands = candidate_signals(word, avail)
            if len(cands) > 1 and not _exact_signal(word, avail):
                narrowed = self._narrow_by_message(cands, message, word)
                if len(narrowed) == 1:
                    return narrowed[0]
                # CONSUMPTION guard: for a consumption/quantity question ("how much
                # fuel", "what it consumes"), an ambiguous namesake like "fuel"
                # (level/temperature/rate) means the RATE — a total-over-time only
                # makes sense for the rate. Prefer the rate among the candidates
                # instead of asking. Data-driven: rate set from field-semantics.
                if self._is_consumption(message):
                    rates = self._rate_signals(source)
                    rate_hits = [c for c in (narrowed or cands) if c in rates]
                    if len(rate_hits) == 1:
                        return rate_hits[0]
                raise NeedsClarification(
                    f"Which one did you mean — {', '.join(narrowed or cands)}?")
        # 2. resolve the (possibly LLM-narrowed) term
        if not term:
            raise NeedsClarification("Which signal do you mean?")
        cands = candidate_signals(term, avail)
        if len(cands) > 1 and not _exact_signal(term, avail):
            raise NeedsClarification(f"Which one did you mean — {', '.join(cands)}?")
        sig = resolve_signal(term, source, avail, self.provider, self.kb)
        if not sig:
            raise NeedsClarification(
                f"I couldn't match '{term}' to a field. Available: {', '.join(avail)}")
        return sig

    def _narrow_by_message(self, candidates, message, matched_word):
        """Given several candidate columns that all matched `matched_word`, keep
        only those whose DISTINGUISHING parts (the parts of the column name beyond
        the matched word, e.g. the '_x' in 'frequency_x') are mentioned in the
        message. So "frequency ... x axis" narrows {frequency_x,_y,_z} to
        frequency_x. Data-driven: the distinguishing parts come from the actual
        column names, not a hardcoded axis list."""
        msg_tokens = set(re.findall(r"[a-z0-9]+", (message or "").lower()))
        # word-parts of the matched term (compared as whole parts, NOT substrings,
        # so a discriminator like "y" isn't dropped just because it's a letter
        # inside "frequency").
        mw_parts = set(re.split(r"[_\W]+", matched_word.strip().lower()))
        kept = []
        for c in candidates:
            parts = re.split(r"[_\W]+", c.lower())
            # discriminators = the name-parts that are NOT part of the matched word
            discriminators = [p for p in parts if p and p not in mw_parts]
            if discriminators and any(d in msg_tokens for d in discriminators):
                kept.append(c)
        return kept

    # --- actions ---
    def capabilities(self, source):
        caps = list_capabilities({source: config.source_glob(source, self.vessel)})
        return describe_capabilities(caps)

    def capabilities_all(self):
        """What you can query across EVERY source on the vessel (for 'what's in all
        my data' / 'what can I ask about across sources'). list_capabilities already
        accepts a multi-source map, so this just passes all of them."""
        sources = self.all_sources()
        if not sources:
            return "No data sources found for this vessel."
        caps = list_capabilities({s: config.source_glob(s, self.vessel)
                                  for s in sources})
        return describe_capabilities(caps)

    def run_discovery(self, source):
        globs = present_globs([config.source_glob(source, self.vessel)])
        if not globs:
            return f"No data found for '{source}'."
        cols, data = load_numeric(globs)
        result = build_per_regime_graphs(cols, data, with_mi=False)
        self._discovery_path(source).write_text(json.dumps(result, indent=2, default=str))
        reg = result["regimes"]
        return (f"Done — discovered {reg['k']} operating modes and "
                f"{len(result['graphs'].get('global',{}).get('edges',[]))} relationships "
                f"in '{source}'. You can now ask about correlations or reasoning.")

    def has_protocol(self, source):
        """Whether the user declared a data protocol for this source (sources.yaml)."""
        return bool(config.source_metadata(self.vessel).get(source, {}).get("protocol"))

    def protocol_hint(self, source):
        """Message telling the user how to declare the protocol for better results."""
        path = config.SHIP_DATA_DIR / self.vessel / "sources.yaml"
        return (f"Note: no data protocol is declared for '{source}'. Field meanings "
                f"will be guessed from names only (lower confidence). For much better "
                f"results, declare it (e.g. J1939 / \"NMEA 0183\" / AIS) in:\n"
                f"  {path}\n"
                f"  sources:\n    {source}:\n      protocol: J1939")

    def run_describe_fields(self, source):
        caps = list_capabilities({source: config.source_glob(source, self.vessel)})
        cols = list(caps.get(source, {}).get("signals", []))   # numeric + stats
        # Also describe NON-numeric columns (e.g. name/callsign/destination) so the
        # LLM can tag their semantic ROLE (identifier / entity_name). Spatial and
        # identity queries resolve columns by role, not by hardcoded names.
        described = {c["name"] for c in cols}
        for c in self.columns(source):
            if c not in described and c.lower() != "timestamp":
                cols.append({"name": c})   # no min/max for text columns
        meta = config.source_metadata(self.vessel).get(source, {})
        protocol = meta.get("protocol")
        proposals = describe_fields(source, cols, self.asset_type, self.provider,
                                    protocol=protocol,
                                    source_description=meta.get("description"))
        self.kb.set_field_semantics(source, proposals)
        msg = f"Described {len(proposals)} fields for '{source}'"
        msg += f" (protocol: {protocol})." if protocol else " (no protocol declared)."
        return msg

    def field_descriptions(self, source):
        fs = self.kb.field_semantics(source)
        if not fs:
            return "No field descriptions yet."
        lines = ["Field meanings (proposed unless confirmed):"]
        for f in fs:
            role = f.get("role", "none")
            role_tag = f", role={role}" if role and role != "none" else ""
            lines.append(f"  - {f['field']}: {f.get('description','?')} "
                         f"[{f.get('unit') or '?'}, {f.get('confidence','?')}{role_tag}]")
        return "\n".join(lines)

    def _semantics_map(self, source):
        """{field: {description, unit, role, confidence}} from the field-semantics
        JSON. The bridge that turns a raw column the detector points at into a
        human meaning. Empty if fields haven't been described yet."""
        return {f["field"]: f for f in self.kb.field_semantics(source) if f.get("field")}

    def _signal_label(self, source, signal, sem=None):
        """Human label for a raw signal column via field-semantics: e.g.
        'EngineCoolantTemperature' -> 'coolant temperature (degC)'. Falls back to
        the raw name if the field isn't described. Data-driven — nothing hardcoded."""
        sem = sem if sem is not None else self._semantics_map(source)
        e = sem.get(signal)
        if not e:
            return signal
        desc = e.get("description") or e.get("meaning") or signal
        unit = e.get("unit")
        return f"{desc} ({unit})" if unit and unit not in ("?", "none") else desc

    def _label_top_signals(self, source, top_signals, sem=None):
        """Attach the field-semantics meaning to each detector-flagged signal so an
        anomaly (from AE or Mahalanobis) POINTS AT the raw column AND says what it
        means. Returns [{signal, meaning, share}]."""
        sem = sem if sem is not None else self._semantics_map(source)
        out = []
        for s in top_signals:
            name = s.get("signal") if isinstance(s, dict) else s
            item = {"signal": name, "meaning": self._signal_label(source, name, sem)}
            if isinstance(s, dict) and "share" in s:
                item["share"] = s["share"]
            out.append(item)
        return out

    def _window(self, source, days):
        globs, label = resolve_window(source, self.vessel, days=days)
        return present_globs(globs), label

    # (module helper _percentile_local defined at end of file)
    # words that mean "consumed over time" -> the RATE signal (integral), not the
    # ambiguous set of same-word namesakes (level/temperature).
    _CONSUMPTION_WORDS = ("consumption", "consumed", "consume", "consumes",
                          "consuming", "used", "using", "burnt", "burned", "burn",
                          "burns", "burning", "usage")

    def _is_consumption(self, message):
        low = (message or "").lower()
        if any(re.search(rf"\b{w}\b", low) for w in self._CONSUMPTION_WORDS):
            return True
        # "how much fuel/gas/diesel ..." is a quantity total over a window, which
        # for a rate signal means consumption — treat it as such so it resolves to
        # the rate rather than asking level-vs-temperature. Data-agnostic: keys off
        # the "how much <X>" quantity phrasing, not a hardcoded signal name.
        return bool(re.search(r"\bhow much\b", low))

    def compute_value(self, source, message, params):
        days = params.get("days")
        present, label = self._window(source, days)
        if not present:
            return f"No data for '{source}' in that window."
        # "fuel CONSUMPTION" means the fuel RATE (integrated) — resolve within the
        # rate signals so it isn't ambiguous with level/temperature.
        if self._is_consumption(message) and self._rate_signals(source):
            signal = self._resolve_rate(params.get("signal"), source,
                                        self._rate_signals(source), message)
            unit = self._integrated_unit(
                resolve_aggregation(signal, source, self.kb, default="integral")[1])
            res = aggregate(present, signal, "integral", label, unit=unit)
            return res.human()
        signal = self._resolve_or_ask(params.get("signal") or "", source,
                                      self.signals(source), message)
        agg, unit = resolve_aggregation(signal, source, self.kb,
                                        default=params.get("aggregation") or "avg")
        # honor explicit aggregation if the user asked for one
        if params.get("aggregation"):
            agg = params["aggregation"]
        res = aggregate(present, signal, agg, label, unit=unit)
        return res.human()

    def make_plot(self, source, message, params):
        days = params.get("days")
        present, label = self._window(source, days)
        if not present:
            return f"No data for '{source}' in that window."
        kind = params.get("plot_kind") or ("scatter" if params.get("signal_y") else "trend")
        avail = self.signals(source)
        if kind == "scatter":
            x = resolve_signal(params.get("signal_x") or "", source, avail, self.provider, self.kb)
            y = resolve_signal(params.get("signal_y") or "", source, avail, self.provider, self.kb)
            if not (x and y):
                return "For a scatter I need two signals; I couldn't match both."
            out = config.ARTIFACTS_DIR / f"plot_{source}_{x}_vs_{y}.png"
            info = scatter(present, x, y, out)
            return (f"Scatter of {y} vs {x}: Pearson r={info['pearson']} "
                    f"({info['points_total']:,} points). Saved: {out.name}")
        # trend. "fuel CONSUMPTION" plot -> the rate signal (no level/temp ambiguity).
        if self._is_consumption(message) and self._rate_signals(source):
            sig = self._resolve_rate(params.get("signal"), source,
                                     self._rate_signals(source), message)
        else:
            sig = self._resolve_or_ask(params.get("signal") or "", source, avail, message)
        agg = params.get("aggregation") or resolve_aggregation(sig, source, self.kb)[0]
        bucket = params.get("bucket") or "1h"
        # optional SPATIAL window: "plot X from <coord> to <coord>" restricts the
        # trend to the voyage window derived from the position track.
        t_range, win_note = self._voyage_t_range(params)
        out = config.ARTIFACTS_DIR / f"plot_{source}_{sig}_{agg}_{bucket}.png"
        info = aggregated_series(present, sig, out, bucket=bucket, agg=agg,
                                 t_range=t_range)
        span = win_note or f"over {label}"
        return (f"Trend of {agg} {sig} per {info['bucket']} {span} "
                f"({info['bins']} bins). Saved: {out.name}")

    def _voyage_t_range(self, params):
        """If the params carry from/to coordinates, derive the voyage time window
        from the position track and return ((t_start,t_end), note). Else (None,'')."""
        pts = self._voyage_points(params)
        if pts is None:
            return None, ""
        (fla, flo), (tla, tlo) = pts
        pos_src = self._own_position_source()
        plat, plon = self._latlon_cols(pos_src) if pos_src else (None, None)
        if not (pos_src and plat and plon):
            return None, ""
        from galene.query import voyage_window
        pos_globs = present_globs([config.source_glob(pos_src, self.vessel)])
        win = voyage_window(pos_globs, plat, plon, fla, flo, tla, tlo)
        if not win:
            return None, ""
        import datetime as _dt
        f = _dt.datetime.utcfromtimestamp(win["t_start"]).strftime("%m-%d %H:%M")
        t = _dt.datetime.utcfromtimestamp(win["t_end"]).strftime("%m-%d %H:%M")
        return (win["t_start"], win["t_end"]), f"over the leg {f}->{t} UTC"

    def compare_trend(self, source, message, params):
        """Answer 'has X gone up/down since ...' by comparing the recent window
        to the equivalent prior window (deterministic)."""
        avail = self.signals(source)
        signal = self._resolve_or_ask(params.get("signal") or "", source, avail, message)
        n = params.get("days") or 2
        all_dates = config.available_dates(source, self.vessel)
        if len(all_dates) < 2:
            return f"Not enough history to compare {signal} over time."
        # recent window = last n dates; prior window = the n dates before that
        recent = all_dates[-n:]
        prior = all_dates[-2 * n:-n] or all_dates[:-n]
        agg, unit = resolve_aggregation(signal, source, self.kb, default="avg")
        rg = present_globs(config.source_globs_for_dates(source, recent, self.vessel))
        pg = present_globs(config.source_globs_for_dates(source, prior, self.vessel))
        if not rg or not pg:
            return f"Not enough data in both periods to compare {signal}."
        r_now = aggregate(rg, signal, agg, "recent", unit=unit).value
        r_prev = aggregate(pg, signal, agg, "prior", unit=unit).value
        if r_now is None or r_prev is None:
            return f"Couldn't compute {signal} for both periods."
        delta = r_now - r_prev
        pct = (delta / r_prev * 100) if r_prev else 0
        direction = "up" if delta > 0 else ("down" if delta < 0 else "unchanged")
        u = f" {unit}" if unit else ""
        return (f"{signal} ({agg}) has gone {direction}: "
                f"{r_prev:.2f}{u} (prior {len(prior)}d) -> {r_now:.2f}{u} (recent "
                f"{len(recent)}d), a change of {delta:+.2f}{u} ({pct:+.1f}%).")

    def _time_range(self, globs):
        import duckdb
        arr = ", ".join("'" + g + "'" for g in globs)
        row = duckdb.connect().execute(
            f"SELECT min(timestamp), max(timestamp) FROM read_parquet([{arr}], union_by_name=true)"
        ).fetchone()
        return float(row[0]), float(row[1])

    def signal_family(self, source, term):
        """Resolve a CONCEPT term (e.g. 'vibration') to the SET of related signals
        on `source` — data-driven, by matching the term against each field's NAME
        and its field-semantics DESCRIPTION. For a magnitude we prefer a coherent
        UNIT sub-group (the most common unit among the matches), so we don't mix
        e.g. acceleration + frequency + temperature. Returns (family, magnitude):
          family    = all matching columns,
          magnitude = the coherent-unit subset best suited for an aggregate.
        """
        from collections import Counter
        tok = term.strip().lower().replace(" ", "").replace("_", "")
        avail = set(self.signals(source))
        sem = {f["field"]: f for f in self.kb.field_semantics(source)}
        family = []
        for c in self.signals(source):
            name_hit = tok in c.lower().replace("_", "")
            desc = (sem.get(c, {}).get("description") or "").lower()
            desc_hit = term.strip().lower() in desc
            if name_hit or desc_hit:
                family.append(c)
        if not family:
            return [], []
        # magnitude subset: the most common unit among the family (coherent group)
        units = Counter((sem.get(c, {}).get("unit") or "").strip() for c in family)
        # ignore empty-unit when a real unit exists
        real = [(u, n) for u, n in units.items() if u]
        best_unit = max(real, key=lambda x: x[1])[0] if real else ""
        magnitude = [c for c in family
                     if (sem.get(c, {}).get("unit") or "").strip() == best_unit] or family
        return family, magnitude

    def _position_at(self, globs, lat_col, lon_col, at_epoch, tolerance_s=600):
        """Position closest to a given instant (within tolerance)."""
        import duckdb
        arr = ", ".join("'" + g + "'" for g in globs)
        row = duckdb.connect().execute(
            f'SELECT "{lat_col}", "{lon_col}" FROM read_parquet([{arr}], union_by_name=true) '
            f'WHERE "{lat_col}" IS NOT NULL AND "{lon_col}" IS NOT NULL '
            f'AND abs(timestamp - {at_epoch}) <= {tolerance_s} '
            f'ORDER BY abs(timestamp - {at_epoch}) LIMIT 1'
        ).fetchone()
        return {"lat": float(row[0]), "lon": float(row[1])} if row else None

    def all_sources(self):
        return config.discover_sources(self.vessel)

    def _source_has_signal(self, source, term):
        """True if `source` has a signal matching `term`."""
        from galene.interpretation.resolve import candidate_signals
        return bool(candidate_signals(term, self.signals(source)))

    def _rate_signals(self, source):
        """Signals in `source` that field-semantics marks as a RATE (aggregation
        == integral) — i.e. consumption-like. Data-driven, no hardcoded names."""
        avail = set(self.signals(source))
        return [f["field"] for f in self.kb.field_semantics(source)
                if f.get("aggregation") == "integral" and f.get("field") in avail]

    def _has_rate_signal(self, source):
        return bool(self._rate_signals(source))

    def route_source(self, intent, message, params, current):
        """Pick the source that can answer this question (vessel-scoped routing).

        - position/nearby -> a source with lat/lon (+ id for nearby).
        - value/plot/trend -> the source that actually HAS the referenced signal.
        - relationship/reasoning/correction -> the current source (discovery-bound).
        Falls back to `current` if nothing better is found.
        """
        srcs = self.all_sources()

        # meta questions about a SOURCE ("what data from ais-own", "describe the
        # vibration fields") route to the source named in the message.
        if intent in ("capabilities", "describe_fields"):
            return self._named_source(message) or current

        if intent in ("position", "distance"):
            # "where is THE asset" wants a single-entity position source. Data-
            # driven distinction from a multi-vessel source: prefer a lat/lon
            # source that does NOT carry an entity identifier (a multi-vessel feed
            # like AIS-of-others does). Fall back to any lat/lon source.
            latlon = [s for s in srcs if all(self._latlon_cols(s))]
            single = [s for s in latlon if not self._id_col(s)]
            if single:
                return single[0]
            if latlon:
                return latlon[0]
            return current

        if intent in ("nearby", "between"):
            # a multi-vessel source: lat/lon + an entity identifier (by role)
            for s in srcs:
                lat, lon = self._latlon_cols(s)
                if lat and lon and self._id_col(s):
                    return s
            return current

        if intent in ("value", "plot", "trend", "voyage", "efficiency", "geo"):
            # route to the source that HAS the referenced quantity. For voyage the
            # quantity is the consumption/fuel-rate signal; if none is named, fall
            # back to a source that has a rate-like signal (aggregation=integral).
            term = (params.get("signal") or params.get("signal_x")
                    or params.get("signal_y") or "")
            if term:
                if self._source_has_signal(current, term):
                    return current
                for s in srcs:
                    if self._source_has_signal(s, term):
                        return s
            if intent == "geo":
                # a CONCEPT term (e.g. "vibration") may match a source by field
                # DESCRIPTIONS (needs the source described) -> family resolver;
                # else the term is often a SOURCE NAME ("vibration") -> route there;
                # else a source that has the named signal by column name.
                if term and self.signal_family(current, term)[0]:
                    return current
                if term:
                    for s in srcs:
                        if self.signal_family(s, term)[0]:
                            return s
                named = self._named_source(message)
                if named:
                    return named
            if intent in ("voyage", "efficiency"):
                # no signal named: prefer a source that has a RATE signal (its
                # field-semantics say aggregation == integral), i.e. a consumption.
                if self._has_rate_signal(current):
                    return current
                for s in srcs:
                    if self._has_rate_signal(s):
                        return s
            return current

        if intent in ("anomaly", "summarize"):
            # Drift/summary questions name a SOURCE/subsystem ("the vibration
            # sensor", "the engine"), not usually a signal. Route to a source
            # named in the message first; else a source owning a named signal;
            # else stay home.
            named = self._named_source(message)
            if named:
                return named
            term = self._named_signal_term(message)
            if term and not self._source_has_signal(current, term):
                for s in srcs:
                    if self._source_has_signal(s, term):
                        return s
            return current

        if intent == "regimes":
            # operating modes live in DISCOVERED structure; route to a named
            # source, else the source owning a named signal, else home.
            named = self._named_source(message)
            if named:
                return named
            term = self._named_signal_term(message)
            if term and not self._source_has_signal(current, term):
                for s in srcs:
                    if self._source_has_signal(s, term):
                        return s
            return current

        if intent in ("relationship", "reasoning"):
            # These operate on a source's DISCOVERED structure. If the user names
            # a signal that lives in a specific source, route to that source so we
            # look at (and, if needed, discover) the right one. Prefer a source
            # that both HAS the signal and is already discovered; else any source
            # with the signal; else stay home.
            term = self._named_signal_term(message)
            if term:
                if self._source_has_signal(current, term):
                    return current
                discovered = [s for s in srcs if self.has_discovery(s)
                              and self._source_has_signal(s, term)]
                if discovered:
                    return discovered[0]
                for s in srcs:
                    if self._source_has_signal(s, term):
                        return s
            return current

        # correction operates on the home source's knowledge
        return current

    def _named_source(self, message):
        """A source the user named in the message, if any (data-driven: matches
        against the ACTUAL source names on disk, tolerant of hyphen/space/case).
        E.g. 'the vibration sensor' -> 'vibration'; 'ais other feed' -> 'ais-other'.
        Longer names matched first so 'ais-other' wins over a bare 'ais'."""
        low = (message or "").lower()
        srcs = sorted(self.all_sources(), key=len, reverse=True)
        for s in srcs:
            variants = {s.lower(), s.lower().replace("-", " "),
                        s.lower().replace("-", ""), s.lower().replace("_", " ")}
            if any(v and v in low for v in variants):
                return s
        return None

    def _named_signal_term(self, message):
        """Pull the most likely signal term the user named (any source's columns).
        Used to route relationship/reasoning to the source that owns the signal."""
        from galene.interpretation.resolve import candidate_signals
        avail = []
        for s in self.all_sources():
            for sig in self.signals(s):
                if sig not in avail:
                    avail.append(sig)
        # exact column name mentioned wins
        low = (message or "").lower()
        for c in avail:
            if c.lower() in low:
                return c
        # else first content word that matches some signal
        _stop = {"what", "correlates", "with", "does", "relate", "to", "the", "is",
                 "why", "related", "and", "of", "show", "me", "about", "how"}
        for w in re.findall(r"[a-z]+", low):
            if w in _stop or len(w) <= 2:
                continue
            if candidate_signals(w, avail):
                return w
        return ""

    def _cols_by_role(self, source, role):
        """Columns tagged with a semantic ROLE by field-semantics (LLM-proposed,
        expert-reviewable). This is the DATA-DRIVEN path: no hardcoded column
        names. Returns the field names carrying that role, most-confident first."""
        _rank = {"high": 0, "medium": 1, "low": 2}
        fs = [f for f in self.kb.field_semantics(source)
              if f.get("role") == role and f.get("field") in set(self.columns(source))]
        fs.sort(key=lambda f: _rank.get(f.get("confidence", "low"), 3))
        return [f["field"] for f in fs]

    def _role_col(self, source, role, name_fallback=None):
        """One column for a role: prefer field-semantics; if none are described
        yet, fall back to a caller-supplied NAME-PATTERN predicate so spatial
        queries still work before describe-fields has run. The fallback is a hint,
        not the mechanism — once fields are described, roles win."""
        by_role = self._cols_by_role(source, role)
        if by_role:
            return by_role[0]
        if name_fallback is not None:
            return next((c for c in self.columns(source) if name_fallback(c)), None)
        return None

    def _latlon_cols(self, source):
        """Latitude/longitude columns, resolved by ROLE (field-semantics), with a
        name-pattern fallback for before-describe-fields."""
        lat = self._role_col(source, "latitude",
                             lambda c: "lat" in c.lower())
        lon = self._role_col(source, "longitude",
                             lambda c: "lon" in c.lower() or "lng" in c.lower())
        return lat, lon

    def _id_col(self, source):
        """The MULTI-ENTITY identifier column, by ROLE (field-semantics), with a
        name-pattern fallback. Crucially, an identifier for 'nearby' must vary
        across rows (many entities) — a CONSTANT column (e.g. an ingestion asset
        tag present in every source) is NOT an entity identifier. So the fallback
        additionally requires the candidate to be non-constant."""
        by_role = self._cols_by_role(source, "identifier")
        # a role-tagged identifier still must actually distinguish entities
        for c in by_role:
            if self._distinct_count(source, c) > 1:
                return c
        # fallback: a plausibly-named column that is NOT constant
        for c in self.columns(source):
            if (c.lower() in ("mmsi", "vessel_id", "id", "device_id", "user_id")
                    and self._distinct_count(source, c) > 1):
                return c
        return None

    def _distinct_count(self, source, col):
        """Approx distinct values of a column (cached). Used to reject constant
        columns as entity identifiers."""
        self._distinct_cache = getattr(self, "_distinct_cache", {})
        key = (source, col)
        if key in self._distinct_cache:
            return self._distinct_cache[key]
        import duckdb
        rel = f"read_parquet('{config.source_glob(source, self.vessel)}', union_by_name=true)"
        try:
            n = duckdb.connect().execute(
                f'SELECT approx_count_distinct("{col}") FROM {rel}').fetchone()[0]
        except Exception:
            n = 0
        self._distinct_cache[key] = n
        return n

    def _name_col(self, source):
        """The human-readable entity-name column, by ROLE, with a name fallback."""
        return self._role_col(
            source, "entity_name",
            lambda c: c.lower() in ("name", "callsign", "shipname", "vessel_name"))

    def position(self, source, message):
        from galene.query import current_position
        lat, lon = self._latlon_cols(source)
        if not (lat and lon):
            return (f"'{source}' has no latitude/longitude fields, so I can't give "
                    "a position. Ask about a source that reports geographic "
                    "coordinates.")
        globs = present_globs([config.source_glob(source, self.vessel)])
        if not globs:
            return "No position data available."
        import datetime as _dt
        # If the message names a TIME ("yesterday at 15:00", "2 hours ago"), give
        # the position AT that instant; otherwise the most recent fix.
        tmin, tmax = self._time_range(globs)
        from galene.query.timeparse import parse_instant_explained
        at, anchor_note = parse_instant_explained(message, tmin, tmax)
        from galene.query.spatial import place_label
        places = config.known_places(self.vessel)
        if at is not None:
            fix = self._position_at(globs, lat, lon, at, tolerance_s=1800)
            when = _dt.datetime.utcfromtimestamp(at).strftime("%Y-%m-%d %H:%M UTC")
            suffix = (" " + anchor_note) if anchor_note else ""
            if not fix:
                return (f"No position fix near {when} (no data within 30 min of "
                        "that time)." + suffix)
            return (f"Position at {when}: "
                    f"{place_label(fix['lat'], fix['lon'], places)} (lat, lon)." + suffix)
        pos = current_position(globs, lat, lon)
        if not pos:
            return "No position data available."
        when = _dt.datetime.utcfromtimestamp(pos["timestamp"]).strftime("%Y-%m-%d %H:%M UTC")
        return (f"Most recent position ({when}): "
                f"{place_label(pos['lat'], pos['lon'], places)} (lat, lon).")

    def distance_travelled(self, source, message, params):
        """How far the asset travelled over a window = length of the position
        track (great-circle sum of hops), NOT an integral of a speed signal.
        Uses the position source; window from params (default all data)."""
        from galene.query import track_distance_km
        pos_src = source if all(self._latlon_cols(source)) else self._own_position_source()
        plat, plon = self._latlon_cols(pos_src) if pos_src else (None, None)
        if not (pos_src and plat and plon):
            return ("I can't measure distance travelled — no source with "
                    "latitude/longitude to trace the track.")
        days = params.get("days")
        globs, label = resolve_window(pos_src, self.vessel, days=days)
        globs = present_globs(globs)
        if not globs:
            return f"No position data for '{pos_src}' in that window."
        tmin, tmax = self._time_range(globs)
        km = track_distance_km(globs, plat, plon, t_range=(tmin, tmax))
        nm = km / 1.852
        return (f"Distance travelled over {label}: {km:,.1f} km ({nm:,.1f} nautical "
                f"miles), traced from the '{pos_src}' position track.")

    def signal_by_location(self, source, message, params):
        """How a signal (or a CONCEPT like 'vibration' = all its axes) varies BY
        LOCATION: aggregate it over geographic cells of the track and report the
        high/low regions. Cross-source (values on `source`, position from the
        position source). Serves e.g. 'is vibration higher in some locations' —
        vibration magnitude by area as a sea-state proxy. Data-driven throughout.
        """
        from galene.query import signal_by_location as _sbl
        term = (params.get("signal") or "").strip()
        # a source name in the message ("vibration") isn't a signal term
        if term and term.lower() in {s.lower() for s in self.all_sources()}:
            term = ""
        # resolve the concept -> a family; use the coherent-unit MAGNITUDE subset.
        if term:
            family, mag = self.signal_family(source, term)
        else:
            # term was a source name / absent: use the whole source's family via a
            # generic descriptor (its own name), else all numeric signals.
            family, mag = self.signal_family(source, source.split("-")[0])
            if not family:
                family = mag = self.signals(source)
        if not mag:
            # fall back to a single named signal resolved normally
            sig = None
            try:
                sig = self._resolve_or_ask(term, source, self.signals(source), message)
            except Exception:
                sig = None
            if not sig:
                return (f"I couldn't find a signal on '{source}' matching "
                        f"'{term or source}' to map by location.")
            mag = [sig]

        pos_src = self._own_position_source()
        plat, plon = self._latlon_cols(pos_src) if pos_src else (None, None)
        if not (pos_src and plat and plon):
            return ("I can't map by location — no position track (a source with "
                    "latitude/longitude) to place the readings.")
        days = params.get("days")
        present, label = self._window(source, days)
        pos_globs = present_globs([config.source_glob(pos_src, self.vessel)])
        val_globs = present_globs([config.source_glob(source, self.vessel)])
        if not (pos_globs and val_globs):
            return "Missing position or value data for a by-location map."
        tmin, tmax = self._time_range(pos_globs)
        cells, is_mag = _sbl(pos_globs, plat, plon, val_globs, mag,
                             cell_deg=0.1, t_range=(tmin, tmax))
        if not cells:
            return (f"No overlapping position + '{source}' data to map by location "
                    f"over {label}.")
        what = (f"{term or source} magnitude" if is_mag else mag[0])
        hi, lo = cells[0], cells[-1]
        _, unit = resolve_aggregation(mag[0], source, self.kb, default="")
        u = f" {unit}" if unit and not is_mag else ""
        spread = (hi["value"] - lo["value"])
        lines = [f"{what} by location over {label} "
                 f"({len(cells)} ~11 km cells, from '{source}' + '{pos_src}' track):"]
        lines.append(f"  highest: {hi['value']:.3g}{u} near {hi['lat']:.2f}, "
                     f"{hi['lon']:.2f} (n={hi['n']:,})")
        lines.append(f"  lowest:  {lo['value']:.3g}{u} near {lo['lat']:.2f}, "
                     f"{lo['lon']:.2f} (n={lo['n']:,})")
        rel = (spread / lo["value"] * 100) if lo["value"] else 0
        if abs(rel) < 5:
            lines.append(f"  -> roughly UNIFORM across locations (only {rel:.0f}% "
                         "high-to-low); no strong spatial pattern.")
        else:
            lines.append(f"  -> varies ~{rel:.0f}% high-to-low across locations — a "
                         "real spatial pattern (could reflect conditions there, "
                         "e.g. sea-state; the data shows the WHERE, not the cause).")
        return "\n".join(lines)

    def _own_position_source(self):
        """The source that gives OUR OWN position (single-entity lat/lon). Data-
        driven: a lat/lon source with NO entity identifier (a multi-vessel feed
        has one). No source-name hardcoding."""
        latlon = [s for s in self.all_sources() if all(self._latlon_cols(s))]
        single = [s for s in latlon if not self._id_col(s)]
        return (single or latlon or [None])[0]

    def nearby(self, source, message):
        from galene.query import ships_nearby, current_position
        # nearby needs OTHER-entity positions + an identifier + own position.
        # All columns resolved by ROLE (field-semantics), not hardcoded names.
        lat, lon = self._latlon_cols(source)
        id_col = self._id_col(source)
        if not (lat and lon and id_col):
            return (f"'{source}' doesn't look like multi-entity position data "
                    "(needs latitude/longitude + an identifier). Ask about a "
                    "source that tracks multiple entities.")
        # own position: the single-entity position source (data-driven), else
        # this source's own latest fix as a last resort.
        own_src = self._own_position_source()
        own_globs = present_globs([config.source_glob(own_src, self.vessel)]) if own_src else []
        olat, olon = self._latlon_cols(own_src) if own_globs else (lat, lon)
        own = current_position(own_globs, olat, olon) if own_globs else None
        if not own:
            return ("I need our own position to judge what's nearby, but no single-"
                    "entity position source is available. Provide one to answer "
                    "proximity questions.")
        globs = present_globs([config.source_glob(source, self.vessel)])
        # resolve the target instant from the message (e.g. "at 13:00 yesterday");
        # default to the latest own position time if no time phrase is given.
        tmin, tmax = self._time_range(globs)
        at = parse_instant(message, tmin, tmax) or own["timestamp"]
        own_at = self._position_at(own_globs, olat, olon, at) or own
        name_col = self._name_col(source)
        found = ships_nearby(globs, at, own_at["lat"], own_at["lon"], lat, lon,
                             id_col, radius_km=50.0, name_col=name_col)
        import datetime as _dt
        when = _dt.datetime.utcfromtimestamp(at).strftime("%Y-%m-%d %H:%M UTC")
        if not found:
            return f"No other entities found within 50 km at {when}."
        lines = [f"{len(found)} within 50 km of our position at {when}:"]
        for f in found[:15]:
            nm = f" ({f['name']})" if f.get("name") else ""
            lines.append(f"  - {f['id']}{nm}: {f['distance_km']} km")
        return "\n".join(lines)

    def distance_between(self, source, message):
        """Distance between TWO named other entities (e.g. 'JORO' and 'HARRIS') at
        a time. Resolves each name in the multi-entity source, gets each one's
        position at the instant, returns the haversine between them. Column names
        role-resolved; entity names parsed from the message by the LLM."""
        from galene.query import entity_position_at, haversine_km
        lat, lon = self._latlon_cols(source)
        id_col = self._id_col(source)
        name_col = self._name_col(source)
        if not (lat and lon and id_col):
            return (f"'{source}' isn't multi-entity position data (needs "
                    "lat/lon + an identifier), so I can't measure between-entity "
                    "distance.")
        names = self._extract_two_entities(message)
        if not names or len(names) < 2:
            return ("Name the two entities, e.g. 'distance between JORO and "
                    "HARRIS at 15:00'.")
        globs = present_globs([config.source_glob(source, self.vessel)])
        tmin, tmax = self._time_range(globs)
        at = parse_instant(message, tmin, tmax) or tmax
        a = entity_position_at(globs, lat, lon, id_col, name_col, names[0], at)
        b = entity_position_at(globs, lat, lon, id_col, name_col, names[1], at)
        import datetime as _dt
        when = _dt.datetime.utcfromtimestamp(at).strftime("%Y-%m-%d %H:%M UTC")
        missing = [n for n, p in ((names[0], a), (names[1], b)) if not p]
        if missing:
            return (f"Couldn't find {' and '.join(repr(m) for m in missing)} near "
                    f"{when} in '{source}'.")
        km = haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
        na = a.get("name") or a["id"]
        nb = b.get("name") or b["id"]
        return (f"Distance between {na} and {nb} at {when}: {km:,.2f} km "
                f"({km/1.852:,.2f} nautical miles).")

    def _extract_two_entities(self, message):
        """Pull two entity names/ids the user named for a between-distance query."""
        raw = self.provider.complete(
            "Extract exactly the TWO entity/vessel names or ids the user is "
            "comparing. Reply as JSON list of two strings, e.g. [\"JORO\",\"HARRIS\"]. "
            "Names only, no extra words.",
            f"Message: {message}\nJSON:")
        try:
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            arr = json.loads(m.group(0)) if m else []
            return [str(x).strip() for x in arr if str(x).strip()][:2]
        except Exception:
            return []

    def voyage(self, source, message, params):
        """How much of a quantity (fuel/consumption) was used going from a START
        point to an END point. Cross-source: the WINDOW is derived from the
        position track; the INTEGRAL is computed on `source` (which owns the rate
        signal). Everything resolved by ROLE / field-semantics — no hardcoding.
        """
        from galene.query import voyage_window, aggregate

        # 0. TRIP ABSTRACTION: "the last voyage / trip / leg" needs no coordinates —
        # auto-detect legs from the track and use the most recent one's window.
        pts = self._voyage_points(params)
        if pts is None and self._is_last_trip(message):
            leg = self._last_leg()
            if leg is None:
                return ("I couldn't identify a completed voyage from the track yet "
                        "(need at least two port-calls/stops). Give me start and end "
                        "coordinates and I'll compute it directly.")
            return self._voyage_over_window(source, message, params, leg,
                                            win_label=f"the last voyage "
                                            f"({leg['from']} -> {leg['to']})")

        # 1. start/end points. Coordinates only for now; place names need a
        # geocoder (a separate, opt-in follow-up) — say so rather than guess.
        if pts is None:
            if params.get("from_place") or params.get("to_place"):
                return ("I can compute voyage fuel between two COORDINATES today "
                        "(e.g. 'from 54.5,18.5 to 57.5,-4.2'). Turning place names "
                        f"like '{params.get('from_place') or params.get('to_place')}'"
                        " into coordinates needs a geocoder, which isn't enabled "
                        "yet. Give me the lat/lon of each end and I'll do it.")
            return ("For a voyage I need a START and an END point as coordinates, "
                    "e.g. 'how much fuel from 54.5,18.5 to 57.5,-4.2'.")
        (from_lat, from_lon), (to_lat, to_lon) = pts

        # 2. the position track (single-entity position source), role-resolved.
        pos_src = self._own_position_source()
        plat, plon = self._latlon_cols(pos_src) if pos_src else (None, None)
        if not (pos_src and plat and plon):
            return ("I can't find a position track to derive the voyage window "
                    "from (no source with latitude/longitude). ")
        pos_globs = present_globs([config.source_glob(pos_src, self.vessel)])
        win = voyage_window(pos_globs, plat, plon,
                            from_lat, from_lon, to_lat, to_lon)
        if not win:
            return ("Couldn't match those points to the position track — the "
                    "vessel's track may not pass near them.")

        # 3. the rate signal to integrate, on `source` (routed by ownership).
        # A voyage asks how much was CONSUMED — that only makes sense for a RATE
        # signal (aggregation=integral). So resolve the user's term against the
        # RATE signals only: "fuel" then means EngineFuelRate, not FuelLevel (%)
        # or FuelTemperature (degC) — no spurious ambiguity. Data-driven: the rate
        # set comes from field-semantics, not a hardcoded name.
        rates = self._rate_signals(source)
        if not rates:
            return (f"I don't see a consumption/rate signal on '{source}' to total "
                    "over the voyage (no field is marked as a rate). Describe the "
                    "fields first, or name the rate signal.")
        # resolve within the rate subset; a non-rate namesake ("fuel" hitting
        # FuelLevel) falls back to the rate signal rather than erroring.
        signal = self._resolve_rate(params.get("signal"), source, rates, message)
        agg, rate_unit = resolve_aggregation(signal, source, self.kb, default="integral")
        # a voyage 'consumption' is a time-integral of a rate
        if agg != "integral":
            agg = "integral"
        # integrating a RATE over time changes the unit: e.g. L/h -> L, kg/h -> kg.
        # Derive the integrated unit by stripping a per-hour denominator.
        unit = self._integrated_unit(rate_unit)

        # 4. integrate the rate over the voyage window (cross-source: window from
        # pos_src, integral on `source`).
        globs = present_globs([config.source_glob(source, self.vessel)])
        t_range = (win["t_start"], win["t_end"])
        res = aggregate(globs, signal, "integral", "voyage", unit=unit, t_range=t_range)

        import datetime as _dt
        f = _dt.datetime.utcfromtimestamp(win["t_start"]).strftime("%Y-%m-%d %H:%M")
        t = _dt.datetime.utcfromtimestamp(win["t_end"]).strftime("%Y-%m-%d %H:%M")
        dur_h = (win["t_end"] - win["t_start"]) / 3600.0
        val = f"{res.value:,.1f} {unit}".strip() if res.value is not None else "n/a"
        note = ""
        if win["start_dist_km"] > 5 or win["end_dist_km"] > 5:
            note = (f" (nearest track approach: {win['start_dist_km']} km to start, "
                    f"{win['end_dist_km']} km to end)")
        out = (f"{signal} used from ({from_lat:.4f}, {from_lon:.4f}) to "
               f"({to_lat:.4f}, {to_lon:.4f}): {val}. "
               f"Window {f} → {t} UTC ({dur_h:.1f} h), track ~{win['track_km']:.0f} km"
               f" [from '{pos_src}' position, integrated on '{source}'].{note}")
        # compound "...and is that normal for the distance?" -> always answer it
        # (verdict, or an honest can't-judge-yet reason), never silently drop it.
        if self._wants_normal_check(message) and res.value is not None:
            out += "\n" + self._voyage_efficiency_norm(source, signal, unit)
        return out

    def _integrated_unit(self, rate_unit):
        """Unit of the time-INTEGRAL of a rate: strip a per-hour denominator.
        'L/h' -> 'L', 'kg/hr' -> 'kg', 'L/hour' -> 'L'. If it's not a per-hour
        rate, return it unchanged (best-effort; unit is informational)."""
        u = (rate_unit or "").strip()
        for suffix in ("/h", "/hr", "/hour", " per hour", "/hr.", "/ h"):
            if u.lower().endswith(suffix):
                return u[: len(u) - len(suffix)].strip()
        return u

    def _resolve_rate(self, term, source, rates, message):
        """Pick the rate signal for a consumption question. Resolve `term` within
        the RATE subset; but if the term was a non-rate namesake (e.g. the
        extractor guessed FuelLevel) or doesn't resolve, fall back to the sole
        rate signal, or ask among the rates if there are several. Never errors on
        a non-rate namesake — 'fuel' for a consumption query means the fuel RATE."""
        if len(rates) == 1:
            return rates[0]                      # only one consumable rate -> it
        if term:
            try:
                sig = self._resolve_or_ask(term, source, rates, message)
                if sig in rates:
                    return sig
            except NeedsClarification:
                raise                            # genuinely ambiguous among rates
            except Exception:
                pass
        # several rates, none clearly named -> ask which one
        raise NeedsClarification(
            f"Which consumption do you mean — {', '.join(rates)}?")

    def _is_cumulative(self, source, signal):
        """True if a signal looks like a CUMULATIVE COUNTER (monotonic non-
        decreasing over time) — e.g. total-hours, odometer. For such a signal the
        'total over a window' is last-first (a delta), not an integral or sum.
        Data-driven: judged from the actual values, no hardcoded field names."""
        import duckdb
        rel = f"read_parquet('{config.source_glob(source, self.vessel)}', union_by_name=true)"
        try:
            row = duckdb.connect().execute(f"""
                WITH s AS (SELECT "{signal}" AS v FROM {rel}
                           WHERE "{signal}" IS NOT NULL ORDER BY timestamp),
                d AS (SELECT v - lag(v) OVER () AS dv FROM s)
                SELECT count(*) FILTER (WHERE dv < -1e-9),
                       count(*) FILTER (WHERE dv > 1e-9),
                       count(*) FILTER (WHERE dv IS NOT NULL)
                FROM d
            """).fetchone()
        except Exception:
            return False
        decreases, increases, total = row[0] or 0, row[1] or 0, row[2] or 0
        # A true counter essentially NEVER decreases over the whole dataset (an
        # odometer/hour-meter only goes up), and increases at least sometimes.
        # (A temperature that warms then cools WILL show decreases over the full
        # data, so this rejects it — even if a short warming window looked flat.)
        return total > 0 and increases > 0 and decreases / total < 0.001

    def _total_over(self, source, globs, signal, t_range):
        """The correct 'total' of `signal` over the window, returned as
        (value, unit, how). Rate -> time-integral; cumulative counter -> delta
        (last-first); otherwise None (a plain 'total' isn't meaningful)."""
        agg, unit = resolve_aggregation(signal, source, self.kb, default="avg")
        if signal in self._rate_signals(source):
            r = aggregate(globs, signal, "integral", "win", unit=unit, t_range=t_range)
            return r.value, self._integrated_unit(unit), "integral"
        if self._is_cumulative(source, signal):
            import duckdb
            lo, hi = sorted(t_range)
            rel = f"read_parquet('{config.source_glob(source, self.vessel)}', union_by_name=true)"
            row = duckdb.connect().execute(f"""
                SELECT max("{signal}") - min("{signal}") FROM {rel}
                WHERE "{signal}" IS NOT NULL AND timestamp BETWEEN {lo} AND {hi}
            """).fetchone()
            return (float(row[0]) if row and row[0] is not None else None), unit, "delta"
        return None, unit, "none"

    def per_ratio(self, source, message, params):
        """General 'X per Y' over a window: total-X / total-Y, where each 'total'
        is computed correctly for its kind (rate -> integral; cumulative counter ->
        delta; distance -> track length). Fully data-driven: X and Y are resolved
        from field-semantics, no hardcoded signal names. Distance is just one Y.
        """
        from galene.query import track_distance_km
        avail = self.signals(source)
        x_term = params.get("signal") or ""
        y_term = params.get("per_signal") or ""

        # X: numerator signal (rate-aware for consumption phrasing)
        rates = self._rate_signals(source)
        x = None
        if rates and (self._is_consumption(message)
                      or any(candidate_signals(w, rates) for w in _content_words(message))):
            x = self._resolve_rate(x_term or None, source, rates, message)
        if x is None and x_term:
            try:
                x = self._resolve_or_ask(x_term, source, avail, message)
            except NeedsClarification:
                raise
            except Exception:
                x = None
        if x is None:
            return ("For an 'X per Y' I need the numerator signal, e.g. "
                    "'fuel per operating hour'.")

        days = params.get("days")
        present, label = self._window(source, days)
        if not present:
            return f"No data for '{source}' in that window."
        tmin, tmax = self._time_range(present)
        t_range = (tmin, tmax)

        x_val, x_unit, x_how = self._total_over(source, present, x, t_range)
        if x_val is None:
            return (f"'{x}' isn't a rate or a cumulative total, so 'total {x}' "
                    "isn't well-defined. For an average-per-Y ask me to PLOT it.")

        # Y: denominator. Distance is special (geometric from the track); else a
        # signal on this source resolved like X.
        y_is_distance = any(k in (y_term + " " + message).lower()
                            for k in ("distance", "km", "mile", "nm", "travel"))
        if y_is_distance and not y_term:
            pos_src = self._own_position_source()
            plat, plon = self._latlon_cols(pos_src) if pos_src else (None, None)
            if not (pos_src and plat and plon):
                return "No position track to measure distance for the 'per Y'."
            pos_globs = present_globs([config.source_glob(pos_src, self.vessel)])
            y_val = track_distance_km(pos_globs, plat, plon, t_range=t_range)
            y_unit, y_name = "km", "distance"
        else:
            # resolve the denominator directly (it's usually an exact column name
            # from the extractor); avoid _resolve_or_ask's whole-message ambiguity
            # scan, which can trip on unrelated words.
            y = resolve_signal(y_term, source, avail, self.provider, self.kb) if y_term else None
            if y is None:
                return ("I couldn't identify the 'per Y' quantity. Name it, e.g. "
                        "'fuel per operating hour' or 'fuel per km'.")
            y_val, y_unit, y_how = self._total_over(source, present, y, t_range)
            y_name = y
            if y_val is None:
                return (f"'{y}' isn't a rate or a cumulative total, so 'per {y}' "
                        "isn't well-defined as a denominator.")
        if not y_val or y_val == 0:
            return (f"The 'per {y_name}' quantity is ~0 over {label}, so the ratio "
                    "isn't meaningful for that window.")

        ratio = x_val / y_val
        u = f"{x_unit}/{y_unit}".strip("/")
        return (f"{x} per {y_name}: {ratio:,.4g} {u} "
                f"({x_val:,.2f} {x_unit} / {y_val:,.2f} {y_unit}, window {label}).")

    def efficiency(self, source, message, params):
        """PER-DISTANCE metric. Two flavors, both bucketing along the track:
          - a RATE signal (fuel): INTEGRATE per distance -> consumption per km.
          - any OTHER signal (velocity_z): AGGREGATE (avg) per distance bin.
        The distance segmentation + plotting is shared; only the per-bin math
        differs. Cross-source (distance from the position source), data-driven.
        """
        from galene.query import aggregate, track_distance_km

        # resolve the signal. Prefer a rate when the phrasing is about consumption
        # or a rate is clearly named; otherwise take whatever signal was named
        # (velocity_z, vibration, ...) and AGGREGATE it per distance.
        rates = self._rate_signals(source)
        avail = self.signals(source)
        term = params.get("signal") or ""
        signal = None
        # A per-distance metric is USUALLY consumption. Prefer the RATE signals
        # when the MESSAGE words point at a rate — judged on the user's own words,
        # not the LLM extractor's single (often wrong) pick. E.g. "fuel per 20km":
        # the word "fuel" matches the fuel RATE, so use it, ignoring an extractor
        # guess of FuelLevel. (Also fires on consumption phrasing / nothing named.)
        msg_hits_rate = any(candidate_signals(w, rates)
                            for w in _content_words(message))
        if rates and (self._is_consumption(message) or not term or msg_hits_rate):
            signal = self._resolve_rate(term or None, source, rates, message)
        # otherwise resolve the named signal against ALL signals (velocity_z, ...)
        if signal is None and term:
            try:
                signal = self._resolve_or_ask(term, source, avail, message)
            except NeedsClarification:
                raise
            except Exception:
                signal = None
        if signal is None:
            return (f"I couldn't match a signal on '{source}' to chart per "
                    "distance. Name the signal (e.g. velocity_z, or a fuel rate).")
        is_rate = signal in rates

        if is_rate:
            _, rate_unit = resolve_aggregation(signal, source, self.kb, default="integral")
            unit = self._integrated_unit(rate_unit)   # L/h -> L
            bin_agg = "integral"
        else:
            _, unit = resolve_aggregation(signal, source, self.kb, default="avg")
            bin_agg = params.get("aggregation") or "avg"

        # time window (default: all data, or last N days if stated)
        days = params.get("days")
        present, label = self._window(source, days)
        if not present:
            return f"No data for '{source}' in that window."
        tmin, tmax = self._time_range(present)
        t_range = (tmin, tmax)

        seg_km = float(params.get("per_distance_km") or 20.0)

        # PLOT path: a chart of <signal> per <seg_km> ALONG the track.
        if self._wants_plot(message, params):
            return self._efficiency_plot(source, signal, unit, seg_km, t_range,
                                         label, bin_agg=bin_agg)

        # NON-rate single value per distance isn't meaningful as one number
        # (it's an aggregate that varies along the route) -> steer to the chart.
        if not is_rate:
            return (f"'{signal}' isn't a consumption rate, so a single per-{seg_km:g}km "
                    "number isn't meaningful — ask me to PLOT it per distance to see "
                    "how it varies along the track.")

        # fuel consumed over the window (integral of the rate)
        res = aggregate(present, signal, "integral", label, unit=unit, t_range=t_range)
        if res.value is None:
            return f"Couldn't compute {signal} over {label}."

        # distance travelled over the same window, from the position source
        pos_src = self._own_position_source()
        plat, plon = self._latlon_cols(pos_src) if pos_src else (None, None)
        if not (pos_src and plat and plon):
            return ("I have the fuel total but can't find a position track to get "
                    "distance travelled (no source with latitude/longitude), so I "
                    "can't compute per-distance consumption.")
        pos_globs = present_globs([config.source_glob(pos_src, self.vessel)])
        dist_km = track_distance_km(pos_globs, plat, plon, t_range=t_range)
        if not dist_km or dist_km <= 0:
            return (f"The vessel barely moved over {label} (track ~{dist_km} km), so "
                    "consumption-per-distance isn't meaningful for that window.")

        per_km = res.value / dist_km
        scale = params.get("per_distance_km") or 1.0
        try:
            scale = float(scale)
        except (TypeError, ValueError):
            scale = 1.0
        per_scaled = per_km * scale
        dist_label = (params.get("per_distance_label") or "").strip() \
            or (f"{scale:g} km" if scale != 1 else "km")
        return (f"{signal}: {per_scaled:,.2f} {unit} per {dist_label} "
                f"({res.value:,.1f} {unit} over {dist_km:,.0f} km travelled, "
                f"window {label}) [fuel on '{source}', distance from '{pos_src}'].")

    def _wants_plot(self, message, params):
        """True if the user asked for a chart (explicit plot_kind, or a
        draw/plot/graph/chart verb in the message)."""
        if params.get("plot_kind"):
            return True
        low = (message or "").lower()
        return any(w in low for w in ("plot", "graph", "chart", "draw", "visual"))

    def _efficiency_plot(self, source, signal, unit, seg_km, t_range, label,
                         bin_agg="integral"):
        """Chart of `signal` per `seg_km` ALONG the track. For a RATE the per-
        segment value is the INTEGRAL (consumption in that bin); for any other
        signal it's an AGGREGATE (avg) in that bin. Cross-source: segments from
        the position source, values on `source`."""
        from galene.query import aggregate, distance_segments
        from galene.query.plots import distance_series

        pos_src = self._own_position_source()
        plat, plon = self._latlon_cols(pos_src) if pos_src else (None, None)
        if not (pos_src and plat and plon):
            return ("I can't chart per-distance — no position track "
                    "(no source with latitude/longitude) to segment by distance.")
        pos_globs = present_globs([config.source_glob(pos_src, self.vessel)])
        segs = distance_segments(pos_globs, plat, plon, seg_km, t_range=t_range)
        if not segs:
            return (f"Not enough track movement over {label} to segment by "
                    f"{seg_km:g} km.")
        val_globs = present_globs([config.source_glob(source, self.vessel)])
        for s in segs:
            r = aggregate(val_globs, signal, bin_agg, "seg", unit=unit,
                          t_range=(s["t_start"], s["t_end"]))
            s["value"] = r.value if r.value is not None else 0.0
        out = config.ARTIFACTS_DIR / f"plot_{source}_{signal}_{bin_agg}_per_{seg_km:g}km.png"
        info = distance_series(segs, out, signal, unit=unit, seg_km=seg_km)
        if bin_agg == "integral":
            total = sum(s["value"] for s in segs)
            detail = f"{total:,.1f} {unit} total"
            what = "total"
        else:
            vals = [s["value"] for s in segs if s["value"] is not None]
            mean = sum(vals) / len(vals) if vals else 0.0
            detail = f"~{mean:,.2f} {unit} overall".strip()
            what = bin_agg
        return (f"Chart of {what} {signal} per {seg_km:g} km along the track over "
                f"{label}: {info['segments']} segments, {detail} "
                f"[values on '{source}', distance from '{pos_src}']. Saved: {out.name}")

    def _voyage_points(self, params):
        """Extract ((from_lat,from_lon),(to_lat,to_lon)) from params, or None if
        coordinates aren't both present."""
        def num(x):
            try:
                return float(x)
            except (TypeError, ValueError):
                return None
        fl, fo = num(params.get("from_lat")), num(params.get("from_lon"))
        tl, to = num(params.get("to_lat")), num(params.get("to_lon"))
        if None in (fl, fo, tl, to):
            return None
        return (fl, fo), (tl, to)

    @staticmethod
    def _is_last_trip(message):
        """True if the message references the most recent voyage/leg/trip without
        giving coordinates ('the last voyage', 'most recent trip', 'this leg')."""
        low = (message or "").lower()
        has_trip = any(w in low for w in ("voyage", "trip", "leg", "journey", "passage"))
        has_recent = any(w in low for w in ("last", "latest", "recent", "this",
                                            "current", "most recent"))
        return has_trip and has_recent

    def _detect_legs(self):
        """Auto-detected voyage legs on the own-position source (place-named via
        the offline port list). Empty if no position track / <2 stops."""
        from galene.query import detect_legs
        pos_src = self._own_position_source()
        if not pos_src:
            return []
        lat, lon = self._latlon_cols(pos_src)
        if not (lat and lon):
            return []
        globs = present_globs([config.source_glob(pos_src, self.vessel)])
        if not globs:
            return []
        return detect_legs(globs, lat, lon, places=config.known_places(self.vessel))

    def _last_leg(self):
        legs = self._detect_legs()
        return legs[-1] if legs else None

    def list_voyages(self, source, message):
        """List the auto-detected voyages/legs (port-call to port-call), most
        recent last, with place names + distance + duration. No endpoints needed."""
        legs = self._detect_legs()
        if not legs:
            return ("I couldn't identify distinct voyages from the track yet — that "
                    "needs at least two port-calls (near-stationary stops). The "
                    "asset may have stayed in one area over the available data.")
        import datetime as _dt
        lines = [f"{len(legs)} voyage leg(s) detected from the position track "
                 "(most recent last):"]
        for lg in legs:
            f = _dt.datetime.utcfromtimestamp(lg["t_start"]).strftime("%m-%d %H:%M")
            t = _dt.datetime.utcfromtimestamp(lg["t_end"]).strftime("%m-%d %H:%M")
            lines.append(f"  - {lg['from']} -> {lg['to']}: {lg['distance_km']:.0f} km, "
                         f"{lg['duration_h']:.1f} h ({f} -> {t} UTC)")
        return "\n".join(lines)

    def _voyage_over_window(self, source, message, params, leg, win_label):
        """Integrate the rate signal over an ALREADY-KNOWN leg window (from trip
        auto-detection). Mirrors voyage()'s integral step but skips the coordinate
        -> window match (the leg already carries t_start/t_end)."""
        from galene.query import aggregate
        rates = self._rate_signals(source)
        if not rates:
            return (f"Found {win_label}, but '{source}' has no consumption/rate "
                    "signal to total over it. Describe the fields or name the rate.")
        signal = self._resolve_rate(params.get("signal"), source, rates, message)
        _, rate_unit = resolve_aggregation(signal, source, self.kb, default="integral")
        unit = self._integrated_unit(rate_unit)
        globs = present_globs([config.source_glob(source, self.vessel)])
        t_range = (leg["t_start"], leg["t_end"])
        res = aggregate(globs, signal, "integral", "voyage", unit=unit, t_range=t_range)
        val = f"{res.value:,.1f} {unit}".strip() if res.value is not None else "n/a"
        import datetime as _dt
        f = _dt.datetime.utcfromtimestamp(leg["t_start"]).strftime("%Y-%m-%d %H:%M")
        t = _dt.datetime.utcfromtimestamp(leg["t_end"]).strftime("%Y-%m-%d %H:%M")
        out = (f"{signal} used on {win_label}: {val}. "
               f"Window {f} -> {t} UTC ({leg['duration_h']:.1f} h), "
               f"track ~{leg['distance_km']:.0f} km.")
        # If the question ALSO asked whether that's normal for the distance, ALWAYS
        # respond to it — with the per-distance verdict when we have the history, or
        # an honest "can't judge yet, here's why" otherwise (never silently drop the
        # part of the question the user asked). No baseline needed — the norm is the
        # asset's own past voyages.
        if self._wants_normal_check(message) and res.value is not None:
            out += "\n" + self._voyage_efficiency_norm(source, signal, unit)
        return out

    @staticmethod
    def _wants_normal_check(message):
        """True if the message asks whether a result is TYPICAL/normal/expected
        (e.g. 'does it look normal', 'is that usual for the distance')."""
        low = (message or "").lower()
        return any(w in low for w in ("normal", "usual", "typical", "expected",
                                      "as always", "compared to", "vs usual",
                                      "abnormal", "unusual"))

    def _voyage_efficiency_norm(self, source, signal, unit):
        """Compare the MOST RECENT leg's per-distance consumption to the asset's
        own prior legs. ALWAYS returns a message: the verdict when there's enough
        history, else an honest 'can't judge yet, here's why' (so the caller never
        silently drops the 'is it normal?' part). Data-agnostic: fuel/km = (rate
        integrated over the leg) / (leg distance); norm = median/p90 of prior legs.
        Observed comparison, never the cause; confidence by number of prior legs."""
        import statistics
        from galene.query import aggregate
        _cant = ("Whether that's within the expected range: I can't judge it yet — "
                 "that compares this trip's fuel-per-distance to enough PRIOR "
                 "voyages, and ")
        legs = self._detect_legs()
        if len(legs) < 3:                     # need a couple of priors + the current
            return _cant + (f"I've only detected {len(legs)} voyage(s) so far "
                            "(need at least a few port-to-port trips to form a norm).")
        globs = present_globs([config.source_glob(source, self.vessel)])
        per_km = []
        for lg in legs:
            if lg["distance_km"] <= 0:
                per_km.append(None); continue
            r = aggregate(globs, signal, "integral", "leg", unit=unit,
                          t_range=(lg["t_start"], lg["t_end"]))
            per_km.append((r.value / lg["distance_km"]) if r.value is not None else None)
        cur = per_km[-1]
        hist = [v for v in per_km[:-1] if v is not None]
        if cur is None:
            return _cant + "this voyage has no usable fuel data over its window."
        if len(hist) < 2:
            return _cant + (f"only {len(hist)} earlier voyage(s) have fuel data over "
                            f"their window on '{source}' (the others fall outside its "
                            "available dates), so there isn't a reliable range yet.")
        med = statistics.median(hist)
        p90 = _percentile_local(hist, 90)
        iunit = f"{unit}/km" if unit else "per km"
        conf = "high" if len(hist) >= 10 else ("medium" if len(hist) >= 5 else "low")
        ratio = cur / med if med else float("inf")
        if cur > p90 and cur >= 1.25 * med:
            verdict = (f"That's HIGHER than usual for the distance: {cur:.2f} {iunit} "
                       f"vs a usual ~{med:.2f} (about {ratio:.1f}x the median of "
                       f"{len(hist)} prior voyage(s)) — worth a look, though it could "
                       f"be load/weather/route, not a fault.")
        elif cur < 0.75 * med:
            verdict = (f"That's LOWER than usual for the distance: {cur:.2f} {iunit} "
                       f"vs a usual ~{med:.2f} — efficient this trip.")
        else:
            verdict = (f"That's about NORMAL for the distance: {cur:.2f} {iunit} vs a "
                       f"usual ~{med:.2f} across {len(hist)} prior voyage(s).")
        return f"Per-distance check (confidence: {conf}): {verdict}"

    def detect_anomaly(self, source, message):
        """Check whether `source` has DRIFTED from its known-good baseline.

        A baseline is a fixed KNOWN-GOOD window the operator designates (only they
        know which period was healthy), so we don't auto-create it — if none is
        set, explain how. Otherwise compare the most recent window to it and
        report the two layers (structure drift = possible fault; regime events =
        usage change), never asserting cause.
        """
        from galene.anomaly import has_baseline, load_baseline, detect_drift
        from galene.anomaly.report import render_drift

        dates = config.available_dates(source, self.vessel)
        if not dates:
            return f"No data for '{source}'."

        # BEHAVIORAL flags first — these need no baseline and are the plain-language
        # "is it operating normally" signal (e.g. stayed somewhere longer than usual).
        behavioral = self.behavioral_flags(source)
        beh_txt = ("BEHAVIOR (vs the asset's own history):\n"
                   + "\n".join("  - " + f for f in behavioral) + "\n\n") if behavioral else ""

        if not has_baseline(source):
            first, last = dates[0], dates[-1]
            note = (f"I don't have a known-good baseline for '{source}' yet, so I "
                    "can't check its sensor-relationship structure for faults — set "
                    "one to enable that:\n"
                    f"  galene baseline {source} --from {first} --to <a-healthy-end-date>\n"
                    f"Available dates: {first} .. {last}.")
            # if behavior itself flagged something, lead with that (it needs no baseline)
            return (beh_txt + note) if beh_txt else note
        # compare the most recent window (default: last 2 days, or all after the
        # baseline) to the baseline.
        baseline = load_baseline(source)
        base_dates = set(baseline.get("dates", []))
        recent = [d for d in dates if d not in base_dates] or dates[-2:]
        result = detect_drift(source, recent, self.vessel, baseline, with_mi=False)
        # stash the structured findings so a follow-up "why?" can EXPLAIN them
        self._last_drift = {"source": source, "result": result,
                            "behavioral": behavioral}
        # Label the raw signals the joint/AE detectors point at with their
        # field-semantics meaning, so the rendered report reads in human terms.
        self._label_drift_signals(source, result)
        return beh_txt + render_drift(result)

    def _label_drift_signals(self, source, result):
        """Add a 'meaning' to each flagged signal in the joint/AE findings using
        the field-semantics JSON (in place). No-op if fields aren't described."""
        sem = self._semantics_map(source)
        if not sem:
            return
        for key in ("joint_anomalies", "ae_anomalies"):
            block = result.get(key)
            if not block:
                continue
            for f in block.get("findings", []):
                for s in f.get("top_signals", []):
                    if isinstance(s, dict) and s.get("signal"):
                        s["meaning"] = self._signal_label(source, s["signal"], sem)

    def _drift_for_source(self, source):
        """The per-source baseline-drift portion of an anomaly check (NO behavioral
        block — that's vessel-level and reported once by the caller). Returns a
        rendered string: the drift report if a baseline exists, else a short
        'no baseline for <source>' note. Stashes _last_drift when it ran a detect."""
        from galene.anomaly import has_baseline, load_baseline, detect_drift
        from galene.anomaly.report import render_drift
        dates = config.available_dates(source, self.vessel)
        if not dates:
            return f"'{source}': no data."
        if not has_baseline(source):
            first, last = dates[0], dates[-1]
            return (f"'{source}': no known-good baseline yet — set one to enable the "
                    f"fault/relationship check:\n"
                    f"  galene baseline {source} --from {first} --to <a-healthy-end-date> "
                    f"(available {first} .. {last}).")
        baseline = load_baseline(source)
        base_dates = set(baseline.get("dates", []))
        recent = [d for d in dates if d not in base_dates] or dates[-2:]
        result = detect_drift(source, recent, self.vessel, baseline, with_mi=False)
        self._last_drift = {"source": source, "result": result,
                            "behavioral": self.behavioral_flags(source)}
        self._label_drift_signals(source, result)
        return render_drift(result)

    @staticmethod
    def _drift_headline(source, result):
        """One-line, high-level verdict for a source's drift result — counts, not
        raw edge tables. e.g. 'engine: relationship drift in 4 of 5 modes, a usage
        shift, and a sequencing change'. Returns (has_finding, line)."""
        l1 = result.get("layer1_structure_drift", [])
        l2 = result.get("layer2_regime_events", [])
        lt = (result.get("layer2_transition_events") or {}).get("findings", [])
        lj = [f for f in (result.get("joint_anomalies") or {}).get("findings", [])
              if f.get("n_flagged", 0) > 0]
        bits = []
        if l1:
            n_modes = len(l1)
            n_edges = sum(len(f.get("changed_edges", [])) for f in l1)
            bits.append(f"relationship drift in {n_modes} mode(s) ({n_edges} coupling change(s))")
        if lj:
            bits.append(f"joint anomalies in {len(lj)} mode(s)")
        if lt:
            bits.append("a sequencing change")
        if l2:
            bits.append(f"{len(l2)} usage shift(s)")
        if not bits:
            return False, f"{source}: no structural change vs its baseline."
        return True, f"{source}: " + ", ".join(bits) + "."

    def detect_anomaly_all(self, message):
        """Anomaly check across EVERY source — a HIGH-LEVEL roll-up, not a dump.
        Leads with the vessel-level behavioral flag (once), then ONE line per
        source (drift headline / no-baseline note / clean). The full per-source
        reports are stashed so a follow-up ('show <source> detail' / 'details')
        can expand them."""
        from galene.anomaly import (has_baseline, load_baseline, detect_drift)
        sources = self.all_sources()
        if not sources:
            return "No data sources found for this vessel."
        behavioral = self.behavioral_flags(sources[0])   # vessel-level, once
        flagged, clean, no_base = [], [], []
        details = {}                                      # source -> full render
        for src in sources:
            dates = config.available_dates(src, self.vessel)
            if not dates:
                continue
            if not has_baseline(src):
                no_base.append(src)
                continue
            baseline = load_baseline(src)
            base_dates = set(baseline.get("dates", []))
            recent = [d for d in dates if d not in base_dates] or dates[-2:]
            result = detect_drift(src, recent, self.vessel, baseline, with_mi=False)
            self._label_drift_signals(src, result)
            has, line = self._drift_headline(src, result)
            (flagged if has else clean).append(line)
            from galene.anomaly.report import render_drift
            details[src] = render_drift(result)
            self._last_drift = {"source": src, "result": result,
                                "behavioral": behavioral}
        # stash the per-source detail for a follow-up "show <source> detail"
        self._anomaly_details = details

        L = [f"Checked {len(sources)} data source(s)."]
        if behavioral:
            L.append("\nBehavior (asset's own history):")
            L += ["  - " + f for f in behavioral]
        if flagged:
            L.append("\nWorth a look:")
            L += ["  - " + x for x in flagged]
        if clean:
            L.append("\nLooked normal vs baseline: "
                     + ", ".join(x.split(":")[0] for x in clean) + ".")
        if no_base:
            L.append("\nNo baseline set (can't fault-check these yet): "
                     + ", ".join(no_base) + ".")
            L.append("  Set one with: galene baseline <source> --from <date> --to <date>")
        if not behavioral and not flagged:
            L.append("\nNothing notable: no behavioral change and no relationship "
                     "drift where a baseline exists.")
        if details:
            L.append("\n(Say \"show <source> detail\" — e.g. \"show "
                     f"{next(iter(details))} detail\" — for the full breakdown.)")
        L.append("\nObserved changes only — a change can be a fault OR a legitimate "
                 "operational change; the data can't say which.")
        return "\n".join(L)

    def anomaly_detail(self, source):
        """Full drift report for one source from the last all-sources sweep (the
        'show <source> detail' follow-up). Falls back to a fresh single-source
        check if we don't have it cached."""
        cache = getattr(self, "_anomaly_details", {}) or {}
        if source in cache:
            return cache[source]
        return self.detect_anomaly(source, "")

    def explain_drift(self, message):
        """Explain the LAST detected drift/behavioral change (a chat 'why?' follow-
        up). Grounds the LLM in the STRUCTURED detector findings + expert-confirmed
        asset facts + manual excerpts; never asserts the cause. Returns None if no
        prior anomaly result is available to explain."""
        import json as _json
        last = getattr(self, "_last_drift", None)
        if not last:
            return None
        source = last["source"]
        result = last["result"]
        # compact, LLM-friendly findings: only the interpretable parts
        findings = {
            "structure_drift": [
                {"regime": f["regime"], "confidence": f.get("confidence"),
                 "changed": [{"signals": c["edge"], "change": c["change"],
                              "delta": c["delta"]} for c in f.get("changed_edges", [])[:6]]}
                for f in result.get("layer1_structure_drift", [])
            ],
            "regime_events": result.get("layer2_regime_events", [])[:4],
        }
        trans = result.get("layer2_transition_events")
        if trans and trans.get("findings"):
            findings["sequencing_changes"] = [
                {"type": f["type"], "from": f["from"], "to": f["to"]}
                for f in trans["findings"][:5]]
        # Joint / AE anomalies: the DETECTOR points at raw signal columns; the
        # field-semantics JSON translates each into a human meaning right on the
        # finding (so "the anomaly is in signal X" reads as "in coolant temp").
        sem = self._semantics_map(source)
        for key in ("joint_anomalies", "ae_anomalies"):
            block = result.get(key)
            if block and block.get("overall_flagged_fraction", 0) > 0:
                findings[key] = [
                    {"regime": f["regime"], "flagged_fraction": f["flagged_fraction"],
                     "signals": self._label_top_signals(source, f.get("top_signals", [])[:3], sem)}
                    for f in block.get("findings", []) if f.get("n_flagged", 0) > 0][:4]
        if last.get("behavioral"):
            findings["behavioral"] = last["behavioral"]
        facts = "\n".join("- " + f.get("fact", "") for f in self.kb.asset_facts()) or "(none)"
        docs_txt = "(none)"
        if config.USER_DOC_DIR.is_dir() and any(config.USER_DOC_DIR.glob("*.pdf")):
            try:
                idx = DocumentIndex.build(config.USER_DOC_DIR)
                terms = [s for f in result.get("layer1_structure_drift", [])
                         for c in f.get("changed_edges", []) for s in c["edge"]]
                hits = idx.search(" ".join(dict.fromkeys(terms)) or source, k=3)
                if hits:
                    docs_txt = "\n".join(
                        f"- {c.text[:300]} ({c.citation()})" for c, _score in hits)
            except Exception:
                pass
        return self.provider.complete(
            _EXPLAIN_SYSTEM,
            _EXPLAIN_USER.format(asset_type=self.asset_type,
                                 findings=_json.dumps(findings, indent=2),
                                 facts=facts, docs=docs_txt, question=message))

    def describe_regimes(self, source, message):
        """Describe the OPERATING MODES (regimes) discovery found: how much time
        in each, and what distinguishes it (the signals furthest from the overall
        average, high or low). Data-driven from the discovery centroids — no
        hardcoded mode names like 'idle'/'cruise' (those are the user's to apply).
        """
        try:
            disc = self._discovery(source)
        except Exception:
            return f"No discovery for '{source}' yet."
        reg = disc.get("regimes", {})
        modes = reg.get("regimes", [])
        if not modes:
            return (f"No distinct operating modes were found for '{source}' "
                    f"({reg.get('note', 'insufficient varying data')}).")
        # overall mean per signal (across regimes, weighted by size) to judge
        # what's HIGH/LOW in each regime relative to normal.
        import statistics
        sigs = list(modes[0].get("centroid", {}).keys())
        overall = {}
        for sig in sigs:
            vals, wts = [], []
            for m in modes:
                if sig in m.get("centroid", {}):
                    vals.append(m["centroid"][sig]); wts.append(m.get("size", 1))
            if vals:
                tot = sum(wts) or 1
                overall[sig] = sum(v * w for v, w in zip(vals, wts)) / tot
        # spread per signal (to rank how distinguishing a deviation is)
        spread = {}
        for sig in sigs:
            cs = [m["centroid"][sig] for m in modes if sig in m.get("centroid", {})]
            spread[sig] = (max(cs) - min(cs)) if len(cs) > 1 else 0.0

        lines = [f"'{source}' runs in {reg['k']} operating mode(s) "
                 f"(silhouette {reg.get('silhouette')}):"]
        for m in sorted(modes, key=lambda x: x.get("fraction", 0), reverse=True):
            cen = m.get("centroid", {})
            # distinguishing signals = largest |centroid - overall| scaled by spread
            scored = []
            for sig in sigs:
                if sig in cen and spread.get(sig):
                    z = (cen[sig] - overall.get(sig, 0)) / spread[sig]
                    scored.append((abs(z), z, sig))
            scored.sort(reverse=True)
            desc = []
            for _, z, sig in scored[:3]:
                desc.append(f"{'high' if z > 0 else 'low'} {sig} ({cen[sig]:.4g})")
            lines.append(f"  - Mode {m['label']}: {m.get('fraction', 0)*100:.0f}% of "
                         f"the time — " + (", ".join(desc) if desc else "mixed") + ".")
        lines.append("(Modes are data-derived usage patterns; the labels/meaning "
                     "(idle, cruise, ...) are yours to assign.)")
        return "\n".join(lines)

    def describe_regimes_all(self, message):
        """Describe operating modes for EVERY source on the vessel (for 'list usage
        patterns for all sources'). Auto-runs discovery per source as needed;
        concatenates each source's regime description."""
        sources = self.all_sources()
        if not sources:
            return "No data sources found for this vessel."
        blocks = []
        for src in sources:
            if not self.has_discovery(src):
                try:
                    self.run_discovery(src)
                except Exception:
                    blocks.append(f"'{src}': couldn't analyze (skipped).")
                    continue
            blocks.append(self.describe_regimes(src, message))
        return "\n\n".join(blocks)

    def behavioral_flags(self, source=None):
        """Behavioral-norm check: does the asset's recent BEHAVIOR depart from its
        own history? First behavior: dwell time at a location ('stayed longer than
        usual'). Runs on the position track (not `source` — behavior is a vessel
        property). Returns a list of plain-language flag strings (may be empty).
        Reports the observed behavior, never the cause."""
        from galene.anomaly import detect_stops, flag_current_dwell
        pos_src = self._own_position_source()
        plat, plon = self._latlon_cols(pos_src) if pos_src else (None, None)
        if not (pos_src and plat and plon):
            return []
        globs = present_globs([config.source_glob(pos_src, self.vessel)])
        if not globs:
            return []
        stops = detect_stops(globs, plat, plon)
        flags = []
        dwell = flag_current_dwell(stops)
        if dwell:
            where = f"({dwell['lat']}, {dwell['lon']})"
            now = "currently " if dwell["ongoing"] else ""
            flags.append(
                f"The {self.asset_type} has {now}stayed in one location {where} for "
                f"~{dwell['current_h']:.0f}h — about {dwell['ratio_vs_median']:g}x its "
                f"usual stay (typically ~{dwell['usual_median_h']:g}h, rarely beyond "
                f"~{dwell['usual_p90_h']:g}h over {dwell['n_history']} prior stops, "
                f"confidence {dwell['confidence']}). Worth a look — this shows the "
                f"behavior changed, not why (could be operational, waiting, or a fault)."
            )
        return flags

    def _notable_facts(self, source):
        """Gather the structured facts the system already computes, as a compact
        dict for summarization: regimes + distribution, strongest relationships,
        notable signal stats (constant / near-range-edge), and drift findings IF a
        known-good baseline exists. Auto-runs discovery if missing. Pure data —
        no LLM here."""
        from galene.anomaly import has_baseline, load_baseline, detect_drift
        facts = {"source": source}

        if not self.has_discovery(source):
            self.run_discovery(source)
        try:
            disc = self._discovery(source)
        except Exception:
            disc = {}
        reg = disc.get("regimes", {})
        facts["regimes"] = {
            "k": reg.get("k", 0),
            "distribution": [
                {"label": r["label"], "pct": round(r.get("fraction", 0) * 100)}
                for r in reg.get("regimes", [])],
        }
        edges = disc.get("graphs", {}).get("global", {}).get("edges", [])
        edges = sorted(edges, key=lambda e: e.get("dcor", 0), reverse=True)
        facts["strongest_relationships"] = [
            {"a": e["a"], "b": e["b"], "dcor": round(e["dcor"], 2), "kind": e["kind"]}
            for e in edges[:8]]

        # notable signal stats from capabilities (constant / excluded fields)
        caps = list_capabilities({source: config.source_glob(source, self.vessel)})
        info = caps.get(source, {})
        facts["signal_count"] = info.get("signal_count", 0)
        facts["not_queryable"] = [
            {"name": e["name"], "reason": e["reason"]}
            for e in info.get("excluded", [])][:6]

        # behavioral-norm flags (dwell-time, ...) — vessel-level, no baseline needed
        beh = self.behavioral_flags(source)
        if beh:
            facts["behavioral_flags"] = beh

        # drift, only if the operator has set a known-good baseline
        if has_baseline(source):
            dates = config.available_dates(source, self.vessel)
            baseline = load_baseline(source)
            base_dates = set(baseline.get("dates", []))
            recent = [d for d in dates if d not in base_dates] or dates[-2:]
            drift = detect_drift(source, recent, self.vessel, baseline, with_mi=False)
            l1 = drift.get("layer1_structure_drift", [])
            # flatten the top changed edges across regimes, strongest first
            changed = []
            for f in l1:
                for c in f.get("changed_edges", []):
                    changed.append({"regime": f["regime"], "edge": c["edge"],
                                    "change": c["change"], "delta": c["delta"]})
            changed.sort(key=lambda c: abs(c["delta"]), reverse=True)
            facts["drift"] = {
                "baseline_window": baseline.get("window"),
                "window": drift.get("window"),
                "top_changes": changed[:8],
                "regime_events": drift.get("layer2_regime_events", [])[:5],
            }
            # sequencing (regime-transition) change — the ORDER modes occur in
            trans = drift.get("layer2_transition_events")
            if trans and trans.get("findings"):
                facts["drift"]["sequencing_changes"] = [
                    {"type": f["type"], "from": f["from"], "to": f["to"],
                     "baseline_prob": f["baseline_prob"], "window_prob": f["window_prob"]}
                    for f in trans["findings"][:5]
                ]
                facts["drift"]["sequencing_confidence"] = trans.get("confidence")
            # joint (Mahalanobis) anomalies — points jointly unusual for their mode
            joint = drift.get("joint_anomalies")
            if joint and joint.get("overall_flagged_fraction", 0) > 0:
                facts["drift"]["joint_anomalies"] = {
                    "overall_flagged_fraction": joint["overall_flagged_fraction"],
                    "by_mode": [
                        {"regime": f["regime"], "flagged_fraction": f["flagged_fraction"],
                         "top_signals": f.get("top_signals", [])[:3]}
                        for f in joint.get("findings", []) if f.get("n_flagged", 0) > 0][:4],
                }
            # multi-timescale: is the change a SUDDEN break or SLOW drift?
            model = baseline.get("regime_model")
            if model is not None and len(dates) >= 2:
                from galene.anomaly import multiscale_drift
                ms = multiscale_drift(source, dates, self.vessel, model,
                                      granularity="day", with_mi=False)
                cls = ms.get("classification", {})
                if cls.get("pattern") and cls["pattern"] not in ("stable",
                                                                  "insufficient_history"):
                    facts["drift"]["temporal_pattern"] = {
                        "pattern": cls["pattern"],
                        "note": cls.get("note", ""),
                        "final_cumulative": cls.get("final_cumulative"),
                    }
        return facts

    def summarize(self, source, message):
        """'What's notable / summarize' — synthesize the structured facts into a
        SHORT prioritized overview. Grounded: the LLM only reorganizes computed
        facts, never invents; always keeps the observed-not-cause caveat. No
        baseline required (folds drift in only if one exists)."""
        facts = self._notable_facts(source)
        # deterministic fallback (stub provider or LLM failure)
        det = self._summary_fallback(facts)
        try:
            import json as _json
            out = self.provider.complete(_SUMMARY_SYSTEM,
                                         _SUMMARY_USER.format(
                                             source=source,
                                             facts=_json.dumps(facts, indent=2)),
                                         max_tokens=700).strip()
            return out or det
        except Exception:
            return det

    def summarize_all(self, message):
        """Whole-vessel overview: summarize EVERY source (for 'what's notable in my
        data' / 'give me an overview'). Leads with the vessel-level behavioral
        flag ONCE, then a short per-source summary. Auto-discovers each as needed."""
        sources = self.all_sources()
        if not sources:
            return "No data sources found for this vessel."
        parts = []
        beh = self.behavioral_flags(sources[0])       # vessel-level, once
        if beh:
            parts.append("BEHAVIOR (vs the asset's own history):\n"
                         + "\n".join("  - " + f for f in beh))
        for src in sources:
            parts.append(f"[{src}]\n" + self.summarize(src, message))
        return "\n\n".join(parts)

    def _summary_fallback(self, facts):
        """Deterministic template summary (no LLM) — also the offline path."""
        s = facts["source"]
        L = [f"Notable in '{s}':"]
        for f in facts.get("behavioral_flags", []):   # lead with behavior changes
            L.append(f"- {f}")
        reg = facts.get("regimes", {})
        if reg.get("k"):
            dist = ", ".join(f"regime {d['label']} {d['pct']}%"
                             for d in reg.get("distribution", [])[:4])
            L.append(f"- {reg['k']} operating modes ({dist}).")
        rel = facts.get("strongest_relationships", [])
        if rel:
            top = rel[0]
            L.append(f"- Strongest coupling: {top['a']} ~ {top['b']} "
                     f"(dcor {top['dcor']}); {len(rel)} strong relationships overall.")
        if facts.get("not_queryable"):
            nq = facts["not_queryable"][0]
            L.append(f"- {len(facts['not_queryable'])} field(s) not queryable "
                     f"(e.g. {nq['name']}: {nq['reason']}).")
        d = facts.get("drift")
        if d and d.get("top_changes"):
            c = d["top_changes"][0]
            L.append(f"- Vs known-good {d['baseline_window']}: {len(d['top_changes'])} "
                     f"relationship(s) drifted (biggest: {c['edge'][0]} ~ "
                     f"{c['edge'][1]} {c['change']}).")
        if d and d.get("regime_events"):
            L.append(f"- {len(d['regime_events'])} usage/regime change(s) vs baseline.")
        elif not d:
            L.append("- No known-good baseline set, so no drift check — set one "
                     "with `galene baseline` to flag changes vs normal.")
        L.append("(Observed structure/changes; the data shows WHAT, not the cause.)")
        return "\n".join(L)

    def _fused_path(self):
        return config.ARTIFACTS_DIR / "discovery_fused.json"

    def _run_fusion(self):
        """Build (or rebuild) fused cross-source discovery -> discovery_fused.json."""
        from galene.discovery.fusion import fuse_sources
        from galene.discovery.relationships import build_per_regime_graphs
        srcs = self.all_sources()
        if len(srcs) < 2:
            return None
        globs = {s: config.source_glob(s, self.vessel) for s in srcs}
        cols, data = fuse_sources(globs)
        result = build_per_regime_graphs(cols, data, with_mi=False)
        self._fused_path().write_text(json.dumps(result, indent=2, default=str))
        return result

    def _cross_source_relationship(self, message, current):
        """If the message references signals from TWO DIFFERENT sources, answer
        from FUSED discovery (per-source graphs can't hold cross-source edges).
        Returns a reply string, or None if it's not a cross-source question.

        Fused signals are namespaced 'source__signal'; we detect two distinct
        source prefixes among the terms the user named."""
        low = (message or "").lower()
        # which sources are referenced (by source name OR by owning a named signal)
        refs = set()
        for s in self.all_sources():
            variants = {s.lower(), s.lower().replace("-", " "), s.lower().replace("-", "")}
            if any(v and v in low for v in variants):
                refs.add(s)
        # also map named signal words to their source
        for w in re.findall(r"[a-z_]+", low):
            if len(w) <= 2:
                continue
            for s in self.all_sources():
                if self._source_has_signal(s, w):
                    refs.add(s)
        if len(refs) < 2:
            return None   # not a cross-source question -> normal per-source path

        if not self._fused_path().exists():
            self._run_fusion()
        if not self._fused_path().exists():
            return None
        fused = json.loads(self._fused_path().read_text())
        edges = fused.get("graphs", {}).get("global", {}).get("edges", [])
        # keep only edges whose endpoints come from DIFFERENT sources
        def _src_of(sig):
            return sig.split("__", 1)[0] if "__" in sig else None
        cross = [e for e in edges if _src_of(e["a"]) and _src_of(e["b"])
                 and _src_of(e["a"]) != _src_of(e["b"])]
        # narrow to the referenced sources
        rel = [e for e in cross
               if {_src_of(e["a"]), _src_of(e["b"])} & refs == {_src_of(e["a"]), _src_of(e["b"])}
               or ({_src_of(e["a"]), _src_of(e["b"])} <= refs)]
        rel = rel or cross
        rel.sort(key=lambda e: e["dcor"], reverse=True)
        if not rel:
            return (f"No significant cross-source relationships found between "
                    f"{', '.join(sorted(refs))} (fused on the common time grid).")
        lines = [f"Cross-source relationships between {', '.join(sorted(refs))} "
                 "(fused on the time grid):"]
        for e in rel[:10]:
            lines.append(f"  - {e['a']} ~ {e['b']}: {e['dcor']:.2f} ({e['kind']})")
        return "\n".join(lines)

    def relationship_lookup(self, source, message):
        # CROSS-SOURCE: if the question spans two different sources (e.g.
        # "vibration vs pitch" — vibration in one source, pitch in nmea), a single
        # source's graph can't answer it. Use FUSED discovery instead.
        cross = self._cross_source_relationship(message, source)
        if cross is not None:
            return cross
        disc = self._discovery(source)
        edges = disc.get("graphs", {}).get("global", {}).get("edges", [])
        avail = self.signals(source)
        # try to resolve a signal the user named; else list strongest
        term_word = message
        target = resolve_signal(term_word, source, avail, self.provider, self.kb)
        if target:
            rel = [e for e in edges if e["a"] == target or e["b"] == target]
            rel.sort(key=lambda e: e["dcor"], reverse=True)
            if not rel:
                return f"No significant relationships found for {target}."
            lines = [f"Relationships for {target}:"]
            for e in rel[:10]:
                other = e["b"] if e["a"] == target else e["a"]
                lines.append(f"  - {other}: {e['dcor']:.2f} ({e['kind']})")
            return "\n".join(lines)
        # fallback: strongest overall
        edges = sorted(edges, key=lambda e: e["dcor"], reverse=True)[:10]
        return "Strongest relationships:\n" + "\n".join(
            f"  - {e['a']} ~ {e['b']}: {e['dcor']:.2f} ({e['kind']})" for e in edges)

    def reason(self, source, message, history):
        report = describe_discovery(self._discovery(source), source)
        docs = None
        if config.USER_DOC_DIR.is_dir() and any(config.USER_DOC_DIR.glob("*.pdf")):
            docs = DocumentIndex.build(config.USER_DOC_DIR)
        report = interpret_report(report, self.asset_type, self.provider, self.kb,
                                  docs=docs, max_relationships=6, max_clusters=2)
        cross = report["interpretation"]["cross_relationship"]
        if cross:
            c = cross[0]
            return f"About {c['hub']} and its relationships:\n{c['analysis']}"
        rels = report["interpretation"]["relationships"]
        return rels[0]["explanation"] if rels else "Nothing to explain yet."

    def capture_correction(self, source, message):
        self.kb.add_asset_fact(message, author="user")
        return ("Noted and saved — I'll use this in future reasoning. "
                "[stored as expert-confirmed knowledge]")


def _percentile_local(vals, p):
    """Simple linear-interpolation percentile (no numpy dependency here)."""
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k); hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)
