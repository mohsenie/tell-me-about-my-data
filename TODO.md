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

- [x] **1. Regime transitions (Layer-2 events)** — DONE. Timestamp plumbing
      (load_numeric_with_time + label_sequence) carries time-ordered per-row regime
      labels through discovery. anomaly/transitions.py builds a baseline transition
      matrix (runs collapsed to real mode changes; P(next|current)); the baseline
      persists it (`transitions`); detect_drift compares the window and reports
      `layer2_transition_events` with UNSEEN / RARE / ABSENT transitions, classified
      by significance, confidence by baseline transition count. Rendered as
      "SEQUENCING CHANGE"; folded into summarize/reasoning facts. Observed change,
      never cause. Data-agnostic (labels have no hardcoded meaning). Tested.

- [x] **2. Multi-timescale fingerprints** — DONE. anomaly/timescale.py groups the
      available dates into periods at a granularity (day/month/year, parsed from the
      partition key — timescale-agnostic), builds one fingerprint per period against
      the fixed regime model, reduces each pair to a scalar distance (fingerprint_
      distance), and reports per period the adjacent-step distance + cumulative-vs-
      first distance. _classify separates sudden_break (a large adjacent step) from
      slow_drift (large cumulative, small steps), plus sudden_and_slow / stable.
      Folded into summarize/reasoning facts as drift.temporal_pattern. Tested
      (helpers + classifier branches + e2e). Year-over-year is the same mechanism at
      granularity="year" once multi-year data exists.

- [x] **3. Feed drift into the interpretation layer** — DONE. ChatDeps.explain_drift
      grounds the LLM in the STRUCTURED findings (which couplings moved + in which
      mode, sequencing changes, behavioral flags) + expert asset facts + field-
      semantic signal meanings + manual excerpts (keyword doc search on the changed
      signals). detect_anomaly stashes _last_drift; Orchestrator._is_why_followup
      (deterministic guard) routes a short causal follow-up ("why?", "what could
      cause that?") after an anomaly/summary turn into explain_drift instead of a
      fresh reasoning query. Prompt forbids asserting cause. Tested (guard + None-
      path + grounded call + routing).

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
- [x] **Reverse geocoding** — DONE (offline). config.known_places(vessel) reads a
      user-provided `places:` list (name/lat/lon/radius_km) from sources.yaml — NO
      network. spatial.nearest_place(lat, lon, places) returns the closest place
      within its radius (else None); spatial.place_label formats "Name (lat, lon
      [, ~N km away])" or raw coords. position() now reports place names. Verified:
      the real stop -> "Inverness Marina"; open water -> raw coords. Tested.
- [ ] **Forward geocoding / port lookup** (place name -> coords for voyage
      endpoints): the reverse direction is done; forward name->coord lookup over the
      same places list is the small remaining piece (voyage still takes coordinates).
- [x] **Voyage / leg auto-detection + trip abstraction** — DONE. spatial.detect_legs
      segments the track into port-calls (behavioral.detect_stops) and the gaps
      between consecutive stops become legs [{from,to (place-labeled), t_start,
      t_end, duration_h, distance_km}] (kept only if distance>move_km). ChatDeps:
      list_voyages (list detected legs), _last_leg / _is_last_trip guard, and a
      "last voyage" branch in voyage() that integrates the rate over the most recent
      leg window (via _voyage_over_window) — so "how much fuel on the last voyage"
      needs NO coordinates. Orchestrator routes "list/what voyages" to list_voyages.
      Verified on ais-own: 3 legs, last ending at Inverness Marina, 96 km/9.6h. Tested.
- [x] **Voyage/trip abstraction** — DONE (see above): the auto-detected leg is the
      first-class "trip" (from/to/when/distance) users reference as "the last
      voyage". Remaining nicety: reference a leg by destination place name ("the leg
      to Rotterdam") — a small extension of the name matching.

---

## P1 — Joint / higher-order multivariate detection (the 3rd detector)

The relational drift detector is PAIRWISE (A<->B couplings). It misses anomalies
that live only in a 3-way+ interaction with no pairwise footprint, and it scores
a window's STRUCTURE, not individual points. This track adds a JOINT detector,
complementary to relational-drift + behavioral-norm. Same per-regime pattern:
learn from the known-good window per regime, persist, score a new window.

- [x] **Per-regime Mahalanobis / covariance envelope** — DONE. anomaly/joint.py:
      fit_envelope (mean + ridge-regularized covariance in scaled space; threshold =
      max(chi2(0.999,df), empirical p99.9) for self-consistency on non-Gaussian
      data), mahalanobis (row scores), per_signal_contribution (explainability),
      build_joint_envelopes (per regime; auto-drops monotonic counters/cumulatives
      via a data-driven monotonicity test), detect_joint (per-mode flagged fraction
      + top contributing signals). Persisted in baseline as joint_envelopes;
      detect_drift adds joint_anomalies; render_drift shows "JOINT ANOMALY"; folded
      into summarize/reasoning/explain facts. Self-consistency verified (~0.1%).
      Training-free, deterministic, explainable. Point-level scoring. Tested.
- [x] **Autoencoder backend (optional)** — DONE. anomaly/autoencoder.py: per-regime
      MLP autoencoder (sklearn MLPRegressor, no torch/new dep) trained on known-good
      rows; reconstruction error = score; threshold = empirical p99.9 (self-consistent
      ~0.1%); PER-FEATURE error preserves attribution. Behind use_ae (CLI ttmd detect
      --ae), OFF by default; only trains where a regime has >=200 rows; seeded for
      determinism; trained on-the-fly from the baseline window (MLP not JSON-persisted).
      ae_available() gates it so the system is unchanged if sklearn is missing.
      Surfaced as ae_anomalies, rendered alongside the Mahalanobis joint result.
      Tested (nonlinear manifold learning + off-manifold flag, too-few-rows None,
      off-by-default + self-consistency e2e).

---

## P1 — Seasonality (compare like-season to like-season)

PROBLEM: the baseline is ONE fixed known-good window. Detecting in a different
season makes a legitimate seasonal difference read as "drift" — the system reports
it honestly and refuses to assert cause, but it can't LABEL it seasonal. Regime
segmentation already absorbs operating-mode seasonality (cold-start vs warm), and
multi-timescale already separates slow drift from sudden breaks; this track closes
the calendar-vs-single-baseline gap. Data-agnostic, deterministic, honest — no
hardcoded calendar (match by operating CONTEXT, not by month).

- [ ] **1. Seasonal baseline library (recommended first).** Allow MULTIPLE named
      known-good baselines (operator still designates each as healthy — never
      auto-invented). `build_baseline --label <name>`; baseline store keyed by
      label. Nothing else about the detectors changes — they just receive the
      chosen baseline.
- [ ] **2. Auto-select the baseline by REGIME DISTRIBUTION similarity.** At detect
      time, compute the window's regime fractions and pick the stored baseline whose
      regime distribution is most similar (context match, not calendar). Report WHICH
      baseline was chosen and HOW CLOSE the match is (confidence). Reuses the regime
      fractions already in the fingerprint; no new heavy math, no deps.
- [ ] **3. Poor-match honesty path.** If no stored baseline matches above a floor,
      say so explicitly ("this period doesn't resemble any known-good baseline —
      possibly a new season OR a real change") instead of forcing a bad comparison.
- [ ] **4. (optional) Provisional auto-baseline for zero-setup.** Propose a baseline
      from the most context-similar PRIOR period, clearly labeled provisional /
      unconfirmed ("compared against a similar past period, not an operator-confirmed
      healthy window"); one-step promote-to-confirmed (reuses the expert-confirm
      pattern). Removes cold-start friction without breaking the "operator picks
      healthy" invariant.
- [ ] **5. (optional, later) Periodicity layer — only if regimes under-split.**
      Start cheap: cyclic calendar features (hour/day/month as sin/cos) fed to regime
      clustering so modes can split on time-of-cycle when the data supports it (still
      unsupervised). Full seasonal decomposition (STL / periodic mean -> detect on
      residuals) is deeper, adds a periodicity-estimation assumption, needs several
      cycles of data, and complicates "which raw signal moved" explainability (label
      residuals as "seasonally-adjusted"). RISK: decomposition can HIDE a fault that
      aligns with a seasonal dip — only add with a concrete case where regimes fail.
- [ ] **6. Surface multi-timescale year-over-year as a first-class report** once
      multi-year data exists (period_key already supports granularity="year").

NOTE: most of the value is in items 1-3 (baseline selection). Regimes already handle
a lot of what decomposition would, so item 5 is a last resort.

---

## P1 — Discovery quality (makes results trustworthy)

- [x] **Partial / conditional dependence** — DONE. dependence.partial_correlation_
      matrix computes |partial correlation| for every pair from the precision matrix
      (inverse of the ridge-regularized correlation matrix), controlling for ALL
      other signals. pairwise_dependence annotates each edge with partial + a direct
      flag: an edge with high marginal Pearson but partial<0.1 that shrank by >0.15
      is INDUCED (common-driver haze), direct=False. Verified: a synthetic C->A,C->B
      collapses A~B (0.92->0.02 direct=False) while A~C survives; on engine,
      boost~oil-pressure (0.81) is flagged indirect (both driven by load). Surfaced
      in _edge_facts (direct/partial) + phrasing (describe_edge says "INDIRECT ...
      mostly explained by other signals"). Tested. Thresholds PARTIAL_DIRECT=0.1,
      PARTIAL_SHRINK=0.15.
- [x] **Time-aware (block) permutation significance** — DONE. dependence.
      block_permutation_pvalue(x, y): shuffles y in CONTIGUOUS blocks (preserving
      each series' short-range autocorrelation, breaking only cross-series
      alignment) -> an honest null; returns a p-value (fraction of permuted dCor >=
      observed). pairwise_dependence(with_significance=True) annotates candidate
      edges (dcor>=0.15) with pvalue + significant (p<=0.05). build_graph
      (with_significance) stride-samples to KEEP time order. Verified: independent
      random walks (phantom, dcor 0.49) come out non-significant; engine physical
      couplings significant p=0.01, 10/45 edges flagged phantom. Advisory annotation
      (not a hard drop). Tested.
- [x] **Fused clustering blur** — DONE (hybrid). fusion.fused_relationship_graph:
      computes the fused GLOBAL graph (the value = CROSS-SOURCE edges) but treats
      fused REGIMES as a diagnostic only — reports regime_quality {silhouette, k,
      blurred} (blurred if silhouette < FUSED_SILHOUETTE_MIN=0.35) with a note
      steering users to per-source operating modes. cmd_fused prints the blur
      warning. Verified: 4-source fusion silhouette 0.34 -> blurred=True, 257
      cross-source edges surfaced. Per-source regimes stay authoritative. Tested.

---

## P1 — Chat robustness (fixes from real usage)

- [x] **Consumption resolution: "how much fuel" / "consumes" no longer asks
      level-vs-temp.** _is_consumption now covers consume/consumes/burn/... AND
      "how much <X>"; the generic resolver (_resolve_or_ask), for a consumption
      question, resolves an ambiguous rate-namesake ("fuel") to the RATE signal
      instead of asking (non-consumption "fuel reading" still asks). Fixed the
      "Which one did you mean — EngineFuelRate, FuelLevel, FuelTemperature?" bug.
- [x] **Compound "how much + is it normal?" answered in one turn.** The voyage
      answer (both the coordinate path and the "last voyage" branch) appends a
      per-distance normality verdict when the question asks (_wants_normal_check):
      _voyage_efficiency_norm compares the latest leg's fuel/km to the MEDIAN/p90 of
      prior legs (the asset's own history — NO baseline needed), reporting HIGHER /
      LOWER / NORMAL with confidence by voyage count. Observed comparison, never
      cause. Needs >=2 valid prior legs (thin sample data may return nothing).
      TODO(next): general compound-question decomposition in the router is still not
      done — this fix is scoped to the fuel-normality case, the common one.

---

## P2 — LLM / interpretation

- [x] **Cost/token logging** — DONE. provider.UsageMeter wraps any provider and
      tallies calls + input/output tokens + an estimated cost (real usage from
      Bedrock's Converse `usage`, else ~4 chars/token estimate). get_provider(meter
      =True) opts in; `ttmd chat --usage` shows it (type 'usage' mid-chat, summary
      on exit). Prices via env (TTMD_PRICE_IN/OUT_PER_1K) so no unverified rate is
      hardcoded as fact. Tested.
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
- [x] **Expert-review UX (CLI)** — DONE. `ttmd review-docs list` now shows the full
      knowledge base by tier (expert-confirmed vs document-extracted/unverified) with
      stable ids; `ttmd review-docs promote <id>` (extracted -> confirmed) and `ttmd
      review-docs reject <id>` (remove). Uses the existing id-based promote_fact/
      reject_fact. Tested. A WEB UI for non-technical curation is still eventual.
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
