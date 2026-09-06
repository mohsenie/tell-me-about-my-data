# TODO — talk-to-my-data (Relational Fingerprinting)

Outstanding work and known issues. See SPEC.md for the method + invariants.
Ordered roughly by priority within each section.

## TESTS
Functional pytest suite (deterministic, no live LLM needed): `.venv/bin/python -m
pytest tests/` — 61 pass, 1 skip (the LLM intent-classification case; run it with
`TTMD_LLM=bedrock python -m pytest tests/`).

### WORKING CONVENTION (don't run the whole suite every change)
Run only the test file for the behavior you touched — each is seconds:
  compute/capabilities/timeparse/spatial -> `pytest tests/test_query.py`  (~12s)
  plots -> `tests/test_plots.py` (~1s) · knowledge/field-parse -> `test_knowledge_fields.py` (<1s)
  discovery/fusion/reporting -> `test_discovery_reporting.py` · anomaly -> `test_anomaly.py` (slow)
  orchestrator/chat_deps dispatch/modes/routing -> `test_dispatch.py` (~18s)
  intent prompts/routing rules -> `test_intent_routing.py`
Or by keyword: `pytest tests/ -k voyage`. Skip the heavy ones: `pytest -m "not slow"`.
Run the FULL suite only as a final gate before considering a change done. RULE:
when you ADD a capability, ADD its functional test in the matching file (assert the
OUTCOME, not LLM prose) so coverage stays high without needing full runs. Coverage of the functional library
(excludes cli.py wiring + unused io reader, see .coveragerc): ~64%. Well-covered:
query (compute/capabilities/plots 91-94% / spatial 85% / timeparse 90%), discovery
(relationships 95% / regimes 94% / fusion 91%), anomaly (baseline/behavioral 91%),
knowledge 60%, orchestrator 63%, chat_deps 50% (deterministic paths). Thin by
design: LLM-synthesis prose (interpret/documents/fields framing) — output is
non-deterministic, low value to assert. Files: tests/test_query.py, test_plots.py,
test_anomaly.py, test_discovery_reporting.py, test_dispatch.py (FakeProvider),
test_knowledge_fields.py, test_intent_routing.py (+ conftest.py FakeProvider/fixtures).

## STATUS (2026-09-06)
Core is BUILT and green (46-case deterministic suite passes). Working today:
vessel-scoped chat with auto-routing + auto-run prerequisites; discovery
(regimes + relationships); describe-fields with semantic ROLES; values / plots /
trends; positional (position / nearby / distance / between); voyage fuel + general
"X per Y" (per distance / per operating-hour / per any rate-or-counter);
signal-by-location (geo); summarize (what's-notable) + regimes (usage patterns);
RELATIONAL drift detector (`ttmd baseline`/`detect` + `anomaly` intent, persisted
regime model); BEHAVIORAL-norm detector (dwell-time, "stayed longer than usual",
no baseline); PRESENTATION MODES (operator/technician/analyst, `--mode` or
mid-chat switch). Scale-tested to a month. Remaining = the hardening items below.

---

## P0 — Core capability (baseline + drift detector) — BUILT; extensions remain

### Baseline store + drift detector (THE anomaly-detection step)
The "flag when the fingerprint drifts" step EXISTS and is solid (src/ttmd/anomaly/):
baseline store with persisted regime model, two-layer detector with stable regime
identity, report, CLI (`ttmd baseline` / `ttmd detect`), and a chat `anomaly` intent.
Self-consistency verified (a window vs its own baseline shows no structure drift).
Three named extensions remain to make it fully mature (see REMAINING below).

DONE:
- **Baseline store** (anomaly/baseline.py): from a known-good window (fixed, not
  rolling — invariant #5), persists per-regime edge fingerprints + regime
  centroids/fractions + a normal-variation BAND estimated from per-day fingerprint
  wobble. Saved to artifacts/baseline_<source>.json. Built from the SAME
  build_per_regime_graphs machinery (no re-implemented dependence).
- **Drift detector** (anomaly/detector.py), two layers:
  - Layer 1 (behavioral/fault): one-to-one regime matching by centroid, per-edge
    dcor diff, flag beyond 3x the variation band, rank by magnitude (#6),
    confidence by sample volume (#7).
  - Layer 2 (regime events): new/unseen regime (neutral, #8) + distribution shift.
- **Presentation** (anomaly/report.py): two types labeled distinctly; reports the
  OBSERVED change, never asserts cause.
- **CLI**: `ttmd baseline <source> --from --to` and `ttmd detect <source> --days N`.
- **Chat**: `anomaly` intent ("has anything drifted?"); asks the operator to set a
  known-good baseline first (only they know which period was healthy — not auto-run).

DONE (hardening):
- [x] **Persist + reuse the regime model** (KMeans + scaler): RegimeResult exposes
      model_params() (columns + scaler mean/scale + cluster centers);
      build_per_regime_graphs surfaces `regime_model`; baseline persists it and
      graphs_for_fixed_regimes()/assign_labels() ASSIGN a new window to the SAME
      regimes (nearest scaled center) instead of re-clustering. detect_drift uses
      identity matches when a model is present (regime L vs baseline L); legacy
      re-cluster+centroid path kept for model-less baselines. Removes regime-label
      instability; Layer-1 diffs are now like-for-like. Self-check still 0 drift.

REMAINING (drift-detector hardening — the three to finish the detector):

- [ ] **1. Regime transitions (Layer-2 events)**: detect abnormal/unexpected
      transitions between operating modes, not just distribution shifts. Requires
      carrying per-row regime labels + timestamps through discovery (currently
      load_numeric drops timestamp and clean_frame reorders/filters rows, so
      temporal order is lost). Then persist the baseline's usual transition matrix
      (which mode follows which, how often) and flag a new window's transitions
      that are rare/absent in the baseline. Classify significance so routine
      transitions (idle<->cruise) are context-only, not alerts.

- [ ] **2. Multi-timescale fingerprints**: build baselines/fingerprints at day /
      month / year granularity and compare adjacent-period vs long-baseline to
      separate SUDDEN breaks from SLOW drift, plus year-over-year for seasonality.
      Currently a single window vs a single baseline. Needs windowing helpers
      beyond date-partition granularity and a comparison mode that reports "abrupt
      change since yesterday" distinctly from "gradual drift over months".

- [ ] **3. Feed drift into the interpretation layer**: let the LLM EXPLAIN a
      flagged drift (which edges changed, in which mode) grounded in the knowledge
      base + docs — reuse the existing interpret pipeline. Detector output is
      structured (per-edge deltas, regime, confidence); pass it to interpret so the
      user gets "oil-pressure/speed coupling weakened in cruise — consistent with
      X per the manual", never asserting cause. Wire a chat follow-up ("why?") from
      an anomaly result into reasoning.

---

## P1 — Voyage / spatial querying (operators think in trips + places, not epochs)

Enables requests like "how much fuel from Gdynia to Scotland?". The DATA supports
it; some pieces now exist, some remain.

DONE: current position, ships-nearby-at-time (per-vessel + distance), time-
expression parsing ("13:00 yesterday").
DONE: **Voyage window from coordinates** — voyage_window() derives the trip time
window from the position track (departure = closest approach to the START point,
arrival = closest approach to the END point after departure; also reports track
distance). Role-resolved lat/lon, no hardcoded columns.
DONE: **Cross-source querying** — aggregate()/_integral() take a t_range; a rate
signal on the source that OWNS it is integrated over a window DERIVED FROM the
position source. Verified: EngineFuelRate integrated on 'engine' over a window
from 'ais-own'.
DONE: **Voyage intent in the router** — "how much fuel from <lat,lon> to <lat,lon>"
routes to the rate-signal source, derives the window from the track, integrates.
LLM routes + extracts the two points; code does the window + integral (propose/
dispose). Integrated unit derived from the rate unit (L/h -> L).
REMAINING:

- [x] **Efficiency / consumption-per-distance**: "fuel consumption per 20km / per
      mile / per nautical mile" — integral of the rate over the window / track
      distance in that window, scaled to the unit. Cross-source, rate-aware
      resolution (no fuel-level/temperature ambiguity). New `efficiency` intent +
      track_distance_km(). Distinct from per-time plots. ALSO a per-distance CHART:
      "plot/draw fuel consumption every 20km" -> distance_segments() cuts the track
      every N km, integrates the rate per segment, distance_series() plots
      litres-per-segment vs distance (spatial analog of the time-bucketed trend).
- [x] **Meta questions route to a named source**: "what data from ais-own" /
      "describe the vibration fields" -> capabilities/describe_fields route to the
      source named in the message (was showing the home source).
- [x] **Spatially-windowed plots**: "plot velocity_z from <coord> to <coord> every
      30 min" -> a trend plot restricted to the voyage window (aggregated_series
      t_range), distinct from a voyage consumption total.
- [x] **Per-distance charts for ANY signal (not just rates)**: "plot velocity_z
      every 10km" AVERAGES the signal per distance segment; a rate (fuel) is
      INTEGRATED per segment. efficiency() + _efficiency_plot() take bin_agg;
      signal resolved rate-first only when the message points at a rate.
- [x] **Presentation modes (operator/technician/analyst)**: `ttmd chat --mode` +
      mid-chat 'switch to X mode'. Reframes INTERPRETIVE intents for the audience
      (operator=short/plain/no-jargon, regime->usage wording; technician=signals+
      check; analyst=full detail unchanged). Presentation only, numbers identical.
      CONVERSATIONAL framing: the framing step now gets recent history + is told to
      answer the SPECIFIC question and NOT repeat already-stated facts (refer back
      instead), so multi-turn sessions read as a dialogue, not standalone reports.
- [x] **Behavioral-norm detector (additive)**: anomaly/behavioral.py — detect_stops
      + dwell_norm + flag_current_dwell. Learns the asset's normal dwell-time from
      its own history and flags "stayed in one location ~24h, ~37x usual" in plain
      language; no baseline needed; reports change not cause. Surfaced first in
      anomaly + summarize (operator's "is it behaving normally"). Verified on the
      real track. NEXT behaviors (same shape): daily-distance, speed-profile,
      stop-frequency.
- [x] **Operating modes / usage patterns (regimes intent)**: "list usage patterns",
      "what operating modes/regimes are there" -> describe_regimes() reports the
      discovered regimes (time distribution + distinguishing signals high/low vs
      the overall mean). Data-derived; mode labels left to the user. Was
      misrouting to capabilities (the field list).
- [x] **Summarize / what's-notable**: "what is notable", "give me an overview",
      "what should I pay attention to" -> new `summarize` intent. _notable_facts()
      gathers computed facts (regimes+distribution, strongest relationships,
      notable/non-queryable fields, drift IF a baseline exists); the LLM
      synthesizes a SHORT prioritized summary (grounded, never invents, observed-
      not-cause caveat), with a deterministic template fallback (offline/stub). No
      baseline required. The one intent where the LLM does more than route.
- [x] **Signal by location (geo)**: "is vibration higher in some locations" / "how
      does X vary by location". New `geo` intent + signal_family() (concept ->
      set of columns via name+description, magnitude = coherent-unit subset) +
      signal_by_location() (ASOF cross-source time-join, ~11km geo cells, high/low
      regions + uniform/varies verdict). Reports WHERE, not cause. Works for any
      signal/asset with a position track (temperature/noise by location, etc.).
- [x] **General "X per Y" ratio**: distance generalized to any denominator —
      "fuel per operating hour", "fuel per MWh", "X per revolution". per_ratio() +
      _total_over() compute total-X/total-Y where each total is by kind (rate ->
      integral, cumulative counter -> delta via _is_cumulative, distance -> track
      length). X/Y from field-semantics; a new source gets it with no code change.
      Known limit: a monotonic SENSOR can be misread as a counter by values alone
      (would need a field-semantics 'cumulative' role to fully disambiguate).
- [x] **Chart-word guard**: an explicit scatter/plot/graph/chart/draw word forces
      plot, never a relationship/reasoning text answer (fixes "scatter of X vs Y"
      flipping to relationship on the LLM).
- [x] **Cross-source relationship (fused)**: "relation between vibration and pitch"
      -> detects two referenced sources, auto-runs fused discovery, returns
      cross-source edges (source__signal ~ source__signal), instead of one
      source's within-graph. (Ref-detection could be tightened to avoid pulling in
      extra sources.)
- [x] **Distance travelled**: "how many km did the ship travel" -> length of the
      position track (track_distance_km over the window), not an integral of a
      speed column. New `distance` intent.
- [x] **Distance between two named vessels**: "distance between JORO and HARRIS at
      15:00" -> resolve each name->id->position (name broadcast is sparse in AIS,
      so match name to id first, then position by id), haversine between them. New
      `between` intent + entity_position_at().
- [x] **Consumption is rate-aware in value/plot too**: "average/plot fuel
      CONSUMPTION" resolves to the fuel RATE (integral), not the ambiguous
      level/temperature set (same principle as voyage/efficiency, now in
      compute_value/make_plot via _is_consumption).
- [x] **Position at a named time**: "where was the ship yesterday at 15:00" /
      "on 2026-09-02 at 19:00" now returns the fix AT that instant (parse_instant
      handles relative phrases AND absolute ISO dates; position() uses it with a
      30-min tolerance), not just the latest fix.
- [ ] **Reverse geocoding**: turn a position into a place name ("57.48,-4.25" ->
      "near Inverness"). Position answers are currently raw coordinates.
- [ ] **Geocoding / port lookup**: resolve place names ("Gdynia", "Scotland") to
      coordinates so the user can say place names instead of lat/lon. voyage
      currently accepts COORDINATES and tells the user to provide them when a
      place name is given. Needs a geocoder (opt-in; outbound network) or a user-
      provided port list (data/config-driven, not hardcoded).
- [ ] **Voyage / leg auto-detection**: derive legs from the track automatically
      (regime 'underway' periods between port calls) so a user can ask about "the
      last voyage" without giving endpoints. Departure/arrival = regime transitions.
- [ ] **Voyage/trip abstraction**: a first-class "trip" object (from/to/when/
      distance) users can reference ("the leg to Rotterdam", "last voyage").

---

## P1 — Discovery quality (makes results trustworthy)

- [ ] **Partial / conditional dependence** to prune the common-driver haze. On the
      engine "everything relates to everything via load", so clusters are dense
      and low-information (nmea: 329 edges, silhouette 0.10). Compute dependence
      conditioned on load / other signals to find DIRECT edges. Without this the
      cross-relationship analysis is noisy.
- [ ] **Time-aware (block) permutation significance** to kill autocorrelation
      artifacts. Slow-drifting signals (fuel temp) show phantom dependence; the
      current row-shuffle permutation breaks time structure. Use block permutation
      or test on changes/innovations, not raw levels.
- [ ] **Fused clustering blur**: combining heterogeneous sources lowers regime
      separation (fused silhouette ~0.27 vs vibration 0.92). Consider per-source
      regimes + cross-source relationships hybrid, or smarter feature selection.

---

## P2 — LLM / interpretation

- [ ] **Cost/token logging**: print actual input/output tokens + estimated cost
      per command (measured ~890 tokens/call; interpret makes ~20 calls).
- [ ] **Reduce calls-per-interpret**: batch multiple relationships into one prompt
      (currently ~15 per-relationship + ~5 cluster calls). Cuts cost several-fold.
- [ ] **Confirm Haiku-4.5 pricing** on the Bedrock pricing page (pricing API only
      exposes Claude-3-Haiku; do NOT quote 4.5 rate as fact until verified).
- [ ] **Report formatting**: constrain LLM output to a tight format. Currently
      verbose markdown (headers, long prose) from the model's natural style.
- [ ] **Embedding-based document retrieval**: current keyword-overlap search pulls
      some table-of-contents / boilerplate noise. Vector retrieval would match
      semantically and ignore formatting. Interface (`DocumentIndex.search`) is
      ready for a drop-in vector backend.
- [ ] **Batch document extraction** further (already batched by ~3500 chars);
      full 256-chunk manual is still dozens of calls (one-time, but reducible).

---

## P3 — Product surface

- [ ] **Web UI**: the whole thing is CLI-only. A non-technical operator/expert
      needs a visual interface (chat, plots inline, review). Big but eventual.
- [ ] **Expert-review UX**: `review-docs` is id-based (fixed) but CLI-only; a
      lightweight UI would help non-technical domain experts curate.
- [ ] **GPU (deferred, likely unnecessary)**: fast O(n log n) dCor solved the
      speed problem. Only revisit (CuPy/cuML) if fusing many sources over a full
      year without subsampling; note 8 GB VRAM (RTX 3070) ceiling.

---

## Known limitations (accepted / documented, revisit if they bite)

- **Structure not meaning**: discovery knows signals relate, not what they are;
  meaning comes from docs + expert (by design).
- **Undirected edges**: dCor/MI/Pearson are symmetric — no causal direction.
  Directed edges (lag / transfer entropy) are a hypothesis-only future extension.
- **Regime coverage**: rare modes (vibration active regime ~0.22%, ~96s) give thin
  baselines; real use needs long windows (~months of data).
- **Doc extraction needs the human gate**: some extracted facts are marginal;
  the promote/reject review step is what keeps quality high. Do not auto-trust.

---

## Recently done (for context)

- POSITIONAL / SPATIAL queries: "where is the ship now?" (latest lat/lon),
  "what ships were nearby at 13:00 yesterday?" (per-vessel snapshot + haversine
  distance to own-ship, cross-source ais-other vs ais-own).
- TIME-EXPRESSION parsing ("13:00 yesterday", "2 hours ago", "9am today"),
  resolved against the DATA's latest timestamp (not wall clock). UTC-correct.
- Renamed query/engine.py -> query/compute.py (was confusingly named like the
  engine source). Removed the last domain hardcode: fuel_consumption now resolves
  its rate signal + unit from FIELD SEMANTICS, not "EngineFuelRate"/litres.
- Field-semantics prerequisite offer for value/plot (units); missing-protocol
  notice pointing the user to sources.yaml; ambiguity clarification (deterministic
  signal-level + LLM intent-level); trend intent ("has X gone up?").
- NL chat ORCHESTRATOR: single surface, prerequisite checks + AUTO-RUN on demand
  (discovery / field-descriptions run inline the first time a question needs them,
  no yes/no gate), multi-intent routing (value/plot/relationship/reasoning/
  correction/capabilities), data-driven signal resolution.
- ROLE-BASED SPATIAL RESOLUTION (removes spatial hardcoding): field-semantics now
  proposes a semantic ROLE per field (latitude/longitude/identifier/entity_name/
  timestamp/none) across ALL columns (incl. text like name/callsign). position/
  nearby resolve lat/lon/id/name by ROLE, not hardcoded column names; identifier
  must be non-constant (rejects constant ingestion tags like device_id) so own-
  position vs multi-entity feed is auto-distinguished. Name patterns kept only as
  a pre-description fallback. describe-fields output budget scales with field
  count (fixes truncation on wide AIS source); JSON parser salvages truncation.
  A new source with differently-named columns works with NO code change.
- VESSEL-SCOPED SOURCE ROUTING: chat is scoped to the vessel, not one source.
  CLI arg is the VESSEL (`ttmd chat [vessel]`, default vessel if omitted); home
  source auto-picked (most signal-rich), overridable via --source. Each intent
  auto-routes to the source that can answer it (position->GPS/AIS, nearby->multi-
  vessel AIS, value/plot->source having the signal, relationship/reasoning->source
  owning the named signal else home). Detected from columns, no hardcoded source
  map. Pending phases remember their source so a deferred query runs on the right
  one.
- Query (values) + plots (scatter/trend/timeseries) + capabilities + describe-fields.
- asset_type + protocol now from sources.yaml (user metadata, not hardcoded).
- Fast O(n log n) distance correlation (~100x, results identical) — was O(n^2).
- Fully automatic fusion (filesystem source discovery + auto column selection).
- Time-window support (--days / --from / --to).
- Domain-scope gate (refuses off-domain + injection).
- 3-tier knowledge + document pre-analysis + expert promote/reject review.
- Review index-shift bug -> stable fact ids.
- ingest-docs chunk selection: even sampling across doc (was first-N front-matter).
- Bedrock wired (Converse API; eu-west-1 regional inference-profile prefix).
