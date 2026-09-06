# Relational Fingerprinting — System Specification

> **Product:** talk-to-my-data, powered by Relational Fingerprinting.
> **Method (technical):** regime-conditional structural-drift anomaly detection,
> learned from raw data with zero prior knowledge.
> Living spec — resume work in a fresh context from this file.
> (Prior catalog/agent-centric spec archived at
> `docs/archive/SPEC-catalog-agent.v1.md`.)
> Last updated: 2026-09-06.

---

## 1. The idea in one paragraph

Point the system at raw, unlabeled telemetry. With **zero knowledge** of what any
signal means, it learns (a) the distinct **operating regimes** the asset moves
through and (b) the **relationship structure** between signals *within each
regime* — the "relational fingerprint" of normal behavior. Thereafter, an
**anomaly is a drift in that structure within its own regime**: relationships
that should hold (given what the asset is currently doing) stop holding. No
labels, no manual thresholds, no per-signal rules. The user then talks to this —
asking what changed, when, and where — grounded in what was actually discovered.

**Why structure, not values:** individual sensors can each look "fine" while the
*relationships between them* break — which is how real faults show up first.
Value-threshold monitoring misses this; structural monitoring catches it.

**Why per-regime:** an asset doing something different (idle -> cruise) is USAGE,
not a fault. Comparing only within the same regime separates "doing something
different" from "behaving wrong" — the core problem in anomaly detection.

---

## 2. Zero-knowledge, proven on real data

Everything below was demonstrated on real vessel telemetry (engine, vibration,
nmea, ais_own, ais_other), treating all columns as unlabeled:

- **Regimes found unsupervised.** Engine: 3 modes (cruise RPM~1720 / idle ~827 /
  cold-start, coolant 56 vs 76), silhouette 0.61. Vibration: 2 modes (at-rest vs
  vibrating — frequency ~0 vs 45-292 Hz), silhouette 0.92.
- **Relationships found label-free.** RPM<->fuel emerged at dCor 0.99 with no
  labels.
- **Nonlinear relationships caught that correlation misses.**
  FuelLevel<->FuelTemperature: Pearson 0.07 (looks unrelated) but dCor 0.38.
- **Relationships change by regime** — edges linear globally but nonlinear within
  cruise. Proves per-regime baselines are necessary.
- **Cross-source fusion reveals shared physics.** Fusing 4 sources on the common
  10Hz grid: nmea.roll <-> vibration.accel_y at dCor 0.92 — two independent
  sensors measuring the same ship motion. Found with zero labels; invisible
  per-source. Cross-sensor corroboration -> a sensor-health anomaly class.
- **Fails safe.** Anchored ais_own (near-constant signals) -> 0 edges reported,
  not invented. Honest "nothing here" beats fabricated structure.

---

## 3. Core method (the pipeline)

```
raw fused telemetry (unlabeled, common 10Hz grid)
  1. REGIME SEGMENTATION   - cluster rows into operating modes (unsupervised)
  2. RELATIONAL FINGERPRINT - per regime, learn the relationship graph:
        pairwise dependence (Pearson + distance-corr + MI), classified
        linear / nonlinear / none. This graph = "normal structure" for the mode.
  3. BASELINE STORE        - from a KNOWN-GOOD window, persist: each regime's
        fingerprint + normal-variation band, AND the regime-level model (known
        regimes, their usual distribution, usual transitions).
  4. DRIFT DETECTION (TWO LAYERS, both vs the known-good baseline):
        Layer 1 - BEHAVIORAL anomaly (likely FAULT): identify the window's
          regime, compare its fingerprint to the SAME-regime baseline; deviation
          beyond the normal band = candidate anomaly; score + localize which
          edges changed; rank.
        Layer 2 - REGIME EVENTS (usage changed - report, classify significance):
          a regime change is itself reportable, not just noise. Classify:
          routine transition (context only) / NEW regime not in baseline / abnormal
          transition / regime-distribution shift. E.g. a route change surfaces as
          a new regime or a distribution shift.
        Present the two DISTINCTLY (different severity: fault vs usage change).
        Report the OBSERVED change, never assert cause (route change vs fault look
        alike from data; only the human knows). Novelty != fault.
  5. TALK TO IT            - user queries the findings in natural language,
        grounded ONLY in discovered regimes/relationships/drifts.
```

### The three dependence measures (why all three)
- **Pearson** — linear strength (blind to nonlinear).
- **Distance correlation (dCor)** — any dependence; 0 iff independent.
- **Mutual information (MI)** — shared information; nonlinear-capable.
Comparing Pearson vs dCor classifies each edge's KIND (linear / nonlinear / none).
The nonlinear-but-zero-linear case is the one a plain correlation heatmap misses.

**Partial (conditional) correlation** further separates DIRECT edges from
common-driver-INDUCED ones: computed from the precision matrix (inverse of the
correlation matrix), it measures each pair's association after controlling for all
other signals. A pair with high marginal correlation but a partial correlation that
collapses (< 0.1, having shrunk > 0.15) is flagged `direct=False` — its link is
mostly shared-driver haze, not a real dependency. Every edge carries `partial` +
`direct`; the report describes induced edges as "INDIRECT — mostly explained by
other signals".

### Multi-timescale
Compute fingerprints at day / month / year scales. Adjacent-period comparison
catches sudden breaks; long-baseline comparison catches slow drift that hides at
short scales. Year-over-year handles seasonality.

---

## 4. Hard invariants (must never be violated)

1. **Zero prior knowledge to start.** Discovery runs on raw unlabeled data. No
   catalog, schema meaning, or manual thresholds are required to detect anomalies.
2. **Two detection layers, kept distinct.** (a) BEHAVIORAL anomaly = structure
   drift WITHIN the same regime (compare like-for-like; likely a fault). (b)
   REGIME EVENTS = the operating mode/distribution itself changed (usage change,
   e.g. route change) — reportable, but classified by significance (routine vs
   new/abnormal) and labeled separately from faults. Regime segmentation is the
   usage-vs-fault separator for (a) AND the event source for (b). Report the
   observed change; never assert its cause.
3. **Structure over values.** The primary signal is change in the relationship
   graph, not individual value thresholds.
4. **Nonlinear-aware.** Use dCor/MI, never Pearson alone. A relationship with
   ~zero linear correlation can be real.
5. **Baseline = KNOWN-GOOD window, not a rolling window.** A rolling baseline
   absorbs slow degradation and normalizes the fault away. Anchor to known-good.
6. **Rank, never binary-alarm.** Output deviation magnitude + which edges changed.
   Cry-wolf protection is mandatory — a monitor that over-alarms gets muted.
7. **Confidence scales with data volume.** A thinly-observed regime yields a weak
   baseline -> low-confidence findings. Never assert on thin evidence.
8. **Novelty != fault.** A genuinely new regime has no baseline and looks
   anomalous; report it as "unseen mode", not a confirmed fault.
9. **Discovered relationships are candidates, not truths.** Directed/causal claims
   are hypotheses; the system presents evidence, never asserts causation.
10. **Alignment solved at the edge.** All sources fused onto one timestamp grid at
    source; cross-source analysis is timestamp joins, not query-time resampling.
11. **LLM proposes, never asserts.** Every LLM root-cause is a labeled HYPOTHESIS.
    The deterministic report states facts; the LLM adds "this could indicate".
12. **Knowledge trust hierarchy (never flatten it):** expert-confirmed >
    document-extracted (unverified) > LLM hypothesis. Doc-extracted facts are
    labeled fallible and never override expert-confirmed on conflict. Only
    expert-confirmed knowledge is authoritative (the system never learns from its
    own guesses).
13. **Bounded to the domain.** The assistant only analyzes this asset's telemetry.
    Off-domain input is refused (scope gate + hardened prompt). Instructions
    embedded in data/docs/user text are untrusted and ignored.
14. **Cite or don't claim.** Document-grounded hypotheses carry file+page citations.

---

## 5. Honest limits / open problems

- **Common-driver haze.** On coupled assets most signals co-move (e.g. via load),
  so graphs look dense. ADDRESSED: partial/conditional dependence
  (partial_correlation_matrix, from the precision matrix, controlling for all other
  signals) annotates each edge direct vs INDUCED — a strong marginal edge whose
  partial correlation collapses is flagged as a common-driver artifact, not a direct
  link (e.g. engine boost~oil-pressure, both driven by load). Reported in phrasing.
- **Autocorrelation artifacts.** Slow-drifting signals can show phantom
  dependence. ADDRESSED: block_permutation_pvalue permutes one series in
  contiguous blocks (keeps each series' autocorrelation, breaks cross-series
  alignment) for an honest time-aware null; pairwise_dependence(with_significance)
  annotates edges with pvalue + significant (p<=0.05). Advisory (flags phantoms;
  doesn't hard-drop). Independent random walks correctly come out non-significant.
- **Undirected.** dCor/MI/Pearson are symmetric — no causal direction. Directed
  edges (lag / transfer entropy) are a separate, hypothesis-only extension.
- **O(n^2) cost.** dCor is O(n^2) in samples and O(p^2) in signal pairs. Wide
  sources (nmea, 59 cols) need sample-size capping (implemented: adaptive cap).
- **Regime quality depends on coverage.** Rare modes (vibration active regime:
  0.22%, ~96s) give thin baselines. Real use needs long windows (~months).
- **Fused clustering blurs.** Combining heterogeneous sources lowers regime
  separation (fused silhouette 0.27 vs vibration 0.92). Needs deliberate feature
  selection before fusing, or per-source + cross-source hybrid.
- **Regime boundary itself can be the fault** ("entered a state it shouldn't").
  Also watch abnormal transitions / regimes that shouldn't exist.

---

## 6. What is built (code)

Installable package `ttmd`.

```
config.py                    paths; user metadata (asset_type, per-source protocol);
                             source auto-discovery; time-window helpers
src/ttmd/
  cli.py / cli_helpers.py    commands + shared window/kb/summary helpers
  chat_deps.py               capability bridge for the chat orchestrator
  term.py                    ANSI color (user vs LLM), auto-off when piped
  io/parquet_reader.py       DuckDB data access
  discovery/                 THE METHOD
    dependence.py            Pearson / dCor (fast O(n log n) via `dcor`) / MI + classify
    regimes.py               segment_regimes() - KMeans + silhouette (usage split)
    relationships.py         build_graph / build_per_regime_graphs; clean_frame
    fusion.py                AUTO cross-source fusion on the 10Hz grid (auto col select)
    loader.py                partitioned + time-windowed loader (list of globs)
  reporting/                 deterministic human-readable report (phrasing/describe)
  query/                     DETERMINISTIC values + charts + spatial (no LLM)
    capabilities.py          what signals exist (+ why some aren't queryable)
    compute.py               aggregate() / fuel_consumption() (time-integral) /
                             latest(). fuel rate+unit resolved from field semantics.
    plots.py                 scatter (correlation) / trend (bucketed) / timeseries
    spatial.py               current_position() / ships_nearby() (haversine) /
                             nearest_place()+place_label() (offline reverse-geocode)
    timeparse.py             "13:00 yesterday" etc -> epoch (UTC, vs data's latest)
  interpretation/            LLM layer, bounded to the domain
    provider.py              LLMProvider + StubProvider + BedrockProvider (Converse)
    knowledge.py             tiered KB: confirmed / doc-extracted / field-semantics
    documents.py             PDF extract+index; extract_facts() pre-analysis (cited)
    fields.py                describe_fields() - protocol-aware field meaning proposals
    resolve.py               data-driven signal + aggregation resolution (no hardcode)
    interpret.py             per-relationship + cross-relationship reasoning (parallel)
    scope.py                 domain-scope gate (context-aware; refuses off-domain)
    orchestrator.py          CHAT BRAIN: intent + prerequisite checks + dispatch
                             + vessel-scoped source routing (per-intent)
ship-data/<vessel>/<source>/logs/date=*/*.parquet   partitioned telemetry
ship-data/<vessel>/sources.yaml   USER-declared asset_type + per-source protocol
user-documentation/          customer manuals (DPX-600.pdf)
artifacts/                   discovery_*.json, report_*.md, knowledge_*.json, plot_*.png
```

### Commands
`chat [vessel]` (orchestrator — the main surface; vessel-scoped, auto-routes to
sources; optional `--source` home override, optional `--mode`) · `discover` ·
`fused` · `baseline` · `detect` · `report` · `interpret` · `describe-fields` ·
`capabilities` · `query` · `plot` · `ingest-docs` · `review-docs` · `correct` ·
`ask`.

### Tests
Functional pytest suite (deterministic, no live LLM): `python -m pytest tests/` —
61 pass / 1 skip (LLM intent case, needs `TTMD_LLM=bedrock`). ~64% coverage of the
functional library (.coveragerc omits cli.py wiring + unused io reader); compute
core (discovery/query/spatial/anomaly) 85-95%. Tests assert OUTCOMES not LLM prose;
a `FakeProvider` (tests/conftest.py) drives dispatch tests without Bedrock.

### Running the chat + persona modes
```
ttmd chat                      # default vessel, analyst mode
ttmd chat vessel-001 --mode operator      # business/ops: short, plain, no jargon
ttmd chat vessel-001 --mode technician    # engineer: signals/couplings + what to check
ttmd chat vessel-001 --mode analyst       # DEFAULT: full detail with numbers
ttmd chat vessel-001 --source engine --mode operator
```
Mode is set with `--mode` at launch OR changed mid-chat by saying "switch to
business/technician/analyst mode" (synonyms: business/ops/manager/captain->operator,
tech/engineer/maintenance->technician, data/expert/raw->analyst). Presentation
ONLY — identical numbers/analysis across modes; only INTERPRETIVE answers
(summarize / anomaly / regimes / reasoning / relationship) are reframed; plain
values/plots are the same in every mode. Implemented in orchestrator.py
(`_norm_mode`, `_detect_mode_switch`, `_frame`, `_MODE_PERSONA`).

### Verified behavior (on real data + live Bedrock)
- **Chat orchestrator**: single surface; checks prerequisites and AUTO-RUNS a
  missing phase inline (discovery for relationship/reasoning; field descriptions
  for value/plot) the first time it's needed — no yes/no gate, just a short
  notice — then answers in the same turn. Routes intents (capabilities /
  describe_fields / value / plot / trend / relationship / reasoning / correction /
  position / nearby / voyage / efficiency / distance / between / geo / anomaly /
  summarize / regimes). Verified end-to-end.
- **Vessel-scoped source routing**: chat is scoped to the vessel, not a single
  source. Each question is routed to the source that can answer it — position ->
  a GPS/AIS source (lat/lon), nearby -> a multi-vessel AIS source (lat/lon + id),
  value/plot/trend -> the source that actually has the referenced signal,
  relationship/reasoning -> the discovery-bound home source. Routing is
  data-driven (detected from columns), not a hardcoded source map. The CLI arg is
  the VESSEL (`ttmd chat vessel-001`, or just `ttmd chat`); the home source is
  auto-picked (most signal-rich) and only used for signal-less relationship/
  reasoning. relationship/reasoning that NAME a signal route to the source that
  owns it. Verified: `chat vessel-001` then "where is the ship / list ships
  nearby / what correlates with engine speed" auto-route to ais-own / ais-other /
  engine (incl. discovering engine on confirm) without ever naming a source.
- **Data-driven signal resolution**: "fuel"/"rpm" -> actual columns via field
  semantics + LLM, no hardcoded mapping.
- **Behavioral-norm detection (is it OPERATING normally?)**: complements the
  relational drift detector. Derives interpretable BEHAVIORS from the position
  track — first: DWELL TIME at a location (stop detection: near-stationary within
  move_km, duration >= min_stop_s) — learns the asset's OWN normal (median/p90 of
  prior stops) and flags a current departure in plain language: "stayed in one
  location ~24h — ~37x its usual stay (~0.65h)". Needs NO baseline (norm is the
  asset's history); confidence scales with #prior stops; reports the observed
  behavior, never the cause. Surfaced FIRST in the anomaly + summarize paths (the
  operator's "is it behaving normally"). Verified on the real track (8 stops, a
  24h ongoing stop flagged at 37x normal). Additive; daily-distance / speed-profile
  behaviors fit the same shape next.
- **Presentation modes (operator / technician / analyst)**: `ttmd chat --mode
  operator` (or "switch to business mode" mid-chat) reframes the INTERPRETIVE
  answers (summarize / anomaly / regimes / reasoning) for the audience — operator:
  short, plain, no jargon, regime->usage wording ("staying in one place longer
  than usual"); the framing is CONVERSATIONAL — it sees recent history, answers
  the SPECIFIC question asked, and refers back to (doesn't repeat) facts already
  stated, so a session reads as a dialogue, not standalone reports;
  than usual", "a route it hasn't taken"); technician: which signals/couplings +
  what to check; analyst (default): full detail, unchanged (no LLM reframing).
  Presentation ONLY — the numbers/analysis are identical across modes; a plain
  value/plot answer is the same in every mode. Verified same drift, three framings.
- **Operating modes / usage patterns (regimes)**: "list usage patterns", "what
  operating modes / regimes are there", "how does it operate" -> describes the
  DISCOVERED regimes: how much time in each + the distinguishing signals (furthest
  from the overall weighted mean, high/low). Data-derived; leaves the mode LABELS
  (idle/cruise/...) to the user. Distinct from capabilities (the field list).
- **Summarize / what's-notable (LLM synthesis, grounded)**: "what is notable",
  "give me an overview", "what should I pay attention to" gathers the COMPUTED
  facts (regimes + distribution, strongest relationships, notable/non-queryable
  fields, and drift IF a known-good baseline exists) and the LLM synthesizes a
  SHORT prioritized summary — drift first, then usage, then structure. Grounded
  (only reorganizes facts, never invents), always with the observed-not-cause
  caveat; a deterministic template is the offline/stub fallback. No baseline
  required (folds drift in only when present). This is the one intent where the
  LLM does more than route — appropriate for an open-ended overview. Verified with
  and without a baseline.
- **Signal by location (geo)**: "is vibration higher in some locations" / "how
  does X vary by location" aggregates a signal — or a CONCEPT family like
  "vibration" (= all its axes, resolved from field names + descriptions, combined
  as a Euclidean magnitude over the coherent-unit subset) — over ~11 km geographic
  cells of the position track, cross-source via a DuckDB ASOF time-join, and
  reports the high/low regions with a uniform-vs-varies verdict. Serves sea-state-
  by-area; reports WHERE, never asserts cause. Data-driven concept->family +
  spatial binning, so it works for "temperature/noise by location" on any asset
  with a position track. Verified: vibration magnitude across 38 cells.
- **Cross-source relationship (fused)**: a relationship question spanning two
  sources (e.g. "relation between vibration and pitch", pitch in nmea) detects the
  two referenced sources, auto-runs FUSED discovery (common time grid), and
  returns cross-source edges (source__signal ~ source__signal) — a per-source
  graph can't hold these. `distance` (track length) and `between` (distance
  between two named entities, name->id->position for AIS name sparsity) intents.
- `discover` / `fused` on all 5 sources; windows (`--days`, `--from/--to`).
  Fused finds real physics label-free (nmea.sog ~ ais.sog 0.98; EngineSpeed ~ sog).
- **Query (deterministic)**: `query` (avg/max/integral...), `capabilities` (lists
  queryable + non-queryable fields with reasons). fuel = time-integral, not sum.
- **Default time window**: when a query (value / trend / efficiency) states NO
  time range, the default is ALL AVAILABLE DATA (not the latest window) — chosen
  for statistical stability on "what is the average ..." questions. The window is
  always stated back in the answer (e.g. "window 2026-09-01..09-04"), so it's
  explicit. Narrow with "over the past N days" / a date range. NOTE for ratios
  like efficiency (fuel per km): "all data" includes any idle periods (fuel burnt
  while not moving), which can inflate per-distance figures — narrow the window
  for an underway-only figure.
- **Plots (deterministic)**: scatter (correlation, with fitted line + r), trend
  (bucketed aggregate, arbitrary bins like `4h`), timeseries.
- **describe-fields**: protocol-aware (J1939/NMEA/AIS from sources.yaml) field
  meaning proposals with unit + aggregation + confidence; reviewable tier. Also
  proposes a semantic ROLE per field (latitude / longitude / identifier /
  entity_name / timestamp / none) over ALL columns incl. text. Output budget
  scales with field count (no truncation on wide sources); parser salvages a
  truncated array.
- **Role-based spatial resolution (data-agnostic)**: position/nearby resolve their
  columns (lat/lon, entity identifier, entity name) by ROLE from field-semantics,
  NOT by hardcoded column names. A role-tagged identifier must also be non-constant
  (rejects an ingestion asset tag like device_id that is constant per source), so
  the own-position source (single-entity lat/lon, no identifier) is distinguished
  from a multi-entity feed automatically. Name-pattern matching remains only as a
  pre-description fallback hint. New/renamed columns work with no code change.
- **Baseline + drift detector (the anomaly step)**: `ttmd baseline
  <source> --from --to` saves a known-good fingerprint (per-regime edge maps +
  centroids/fractions + a normal-variation band from per-day wobble). `ttmd detect
  <source> --days N` / chat "has anything drifted?" compares a window to it in TWO
  layers: (1) within-regime relationship-structure drift (one-to-one regime match,
  per-edge dcor diff beyond 3x the band, ranked, confidence by volume) = possible
  fault; (2) regime events = new/unseen mode (neutral, invariant #8) + distribution
  shift. Reports the OBSERVED change, never the cause. Self-consistency verified: a
  window vs its own baseline shows zero structure drift. The baseline PERSISTS its
  regime model (column order + scaler + cluster centers); on detect, the new window
  is ASSIGNED to those exact regimes (no re-clustering), so regime identity is
  stable across windows (regime 0 vs baseline 0) and Layer-1 diffs are like-for-like.
  Old model-less baselines fall back to re-cluster + centroid matching.
- **Regime transitions (Layer-2 temporal)** — the SEQUENCING of operating modes.
  Using the timestamp plumbing (time-ordered per-row regime labels), the baseline
  persists a TRANSITION MATRIX: consecutive runs collapsed to real mode changes
  (i->j, i!=j), counted and normalized to P(next=j | current=i). On detect, the
  window's transitions are compared: UNSEEN (i->j never in baseline = new
  sequencing), RARE (baseline P below a floor), ABSENT (common in baseline, missing
  now = a usual step dropped). Classified by significance, confidence by baseline
  transition count, reported as `layer2_transition_events` in the drift result and
  rendered as "SEQUENCING CHANGE" (plain "NEW/MISSING/UNUSUAL step"). Observed
  change, never cause. Data-agnostic — labels carry no hardcoded meaning.
- **Multi-timescale fingerprints** — separates a SUDDEN break from SLOW drift.
  Groups the available dates into periods at a granularity (day/month/year, parsed
  from the partition key — timescale-agnostic, nothing hardcoded), builds one
  fingerprint per period against the SAME persisted regime model, reduces each pair
  to a scalar distance (mean abs per-edge dcor change), then reports per period the
  distance to the PREVIOUS period (adjacent step) and to the FIRST period
  (cumulative). A large adjacent step -> sudden_break; large cumulative with only
  small steps -> slow_drift (both -> sudden_and_slow; neither -> stable). Folded
  into summarize/reasoning facts as `temporal_pattern`. Observed pattern, not cause.
- **Drift explanation ("why?" follow-up)** — the interpretation layer can EXPLAIN
  a detected change. `ChatDeps.detect_anomaly` stashes the structured findings
  (`_last_drift`); when the previous turn was an anomaly/summary answer and the
  user asks a short causal follow-up ("why?", "what could cause that?"), a
  deterministic guard (`Orchestrator._is_why_followup`) routes to
  `ChatDeps.explain_drift` instead of a fresh discovery-reasoning query.
  `explain_drift` grounds the LLM in the compact structured findings (which
  couplings moved + in which mode, sequencing changes, behavioral flags) plus
  expert-confirmed asset facts, field-semantic signal meanings, and manual
  excerpts (keyword doc search on the changed signals). The prompt forbids
  asserting a cause — it offers directions to CHECK and cites manual facts.
- **Joint multivariate detector (per-regime Mahalanobis / covariance envelope)** —
  the THIRD detector, catching points whose COMBINATION of readings is unusual for
  their mode even when every single signal is in range and no pairwise correlation
  moved (the gap a pairwise method can't see). Per regime, from the known-good
  window, fit mean + regularized covariance in the model's scaled space (anomaly/
  joint.py). Score a new row by squared Mahalanobis distance; flag beyond a
  threshold = max(chi-square(0.999, df), empirical p99.9 of the training distances)
  — the empirical calibration makes a window vs its own baseline self-consistent
  (~0.1%) even on non-Gaussian data. EXPLAINABLE: the distance decomposes into
  per-signal contributions, so findings name WHICH signals drove the score.
  Monotonic counters / cumulatives are auto-excluded (their mean drifts with time
  by construction — data-driven test, no hardcoded names). Training-free,
  deterministic, light numpy. Persisted in the baseline (`joint_envelopes`),
  surfaced as `joint_anomalies` in the drift result, rendered as "JOINT ANOMALY",
  folded into summarize/reasoning/explain. Observed deviation, never cause.
- **Optional autoencoder backend (nonlinear joint manifold)** — for regimes whose
  normal region is a curved manifold an ellipsoid can't fit. anomaly/autoencoder.py
  trains a small per-regime MLP autoencoder (sklearn MLPRegressor — no new heavy
  dep, no torch) on the known-good rows; reconstruction error is the score,
  threshold = empirical p99.9 of training errors (self-consistent ~0.1%), and the
  PER-FEATURE error preserves attribution (which signals reconstructed worst).
  DELIBERATELY optional and secondary: behind `use_ae` (CLI `ttmd detect --ae`),
  OFF by default — it adds a seeded-but-stochastic training step and only runs
  where a regime has >=200 rows, so it regresses the training-free character and is
  opt-in. Trained on-the-fly from the baseline window (the MLP isn't JSON-persisted).
  If sklearn were unavailable, `ae_available()` is False and everything else is
  unchanged. Surfaced as `ae_anomalies`, rendered alongside the Mahalanobis result.
- **Efficiency / consumption-per-distance (cross-source)**: "average fuel
  consumption per 20km / per mile / per nautical mile" integrates the rate over
  the window (litres) and divides by the track distance travelled in that window
  (track_distance_km on the position source), scaled to the requested distance
  unit. Rate-aware signal resolution (a consumption question means the RATE
  signal, so "fuel" -> the fuel rate, no ambiguity with level/temperature).
  Distinct from a per-TIME plot ("per hour"). Verified: 11.94 L per 20 km.
  A CHART is also supported and works for ANY signal per distance, not just a
  rate: "plot fuel consumption every 20km" integrates the rate per segment;
  "plot velocity_z every 10km" AVERAGES that signal per segment. Segments the
  track by distance (distance_segments), computes a per-segment value (integral
  for a rate, avg otherwise), plots value vs distance (distance_series) — spatial
  analog of the time-bucketed trend. A "scatter of X vs Y" is always a plot (a
  deterministic guard rescues it from the relationship classifier).
- **General "X per Y" ratio (data-driven)**: distance is just one denominator.
  "fuel per operating hour", "fuel per MWh", "X per revolution" compute total-X /
  total-Y over a window, where each "total" is derived by kind: a RATE ->
  time-integral; a CUMULATIVE COUNTER (monotonic, detected from values) -> delta
  (last-first); distance -> track length. X and Y resolve from field-semantics —
  no hardcoded signal names, so a new source/asset gets "X per Y" with no code
  change. Verified: EngineFuelRate per EngineTotalHoursOfOperation = 7.81 L/h.
  Known limit: a monotonic SENSOR can be mistaken for a counter by values alone
  (a semantic fact values can't fully disambiguate); harmless for real queries.
- **Voyage fuel (cross-source, coordinate-based)**: "how much fuel from
  <lat,lon> to <lat,lon>" derives the trip time window from the position track
  (voyage_window: departure = closest track approach to the start point, arrival =
  closest approach to the end AFTER departure; reports track distance) and
  integrates the rate signal over that window ON the source that owns it — window
  from one source, integral on another. Rate signal + integrated unit resolved
  from field-semantics (L/h -> L). LLM extracts the two points; code does the
  window + integral. Verified: EngineFuelRate over an ais-own-derived window,
  integrated on engine. Place names return a helpful "give coordinates" message
  (geocoder is a deferred, opt-in follow-up — no hardcoded place data).
- Interpretation: per-relationship + cross-relationship hypotheses, cited to the
  DPX-600 manual; LLM calls run in PARALLEL (~20 calls -> ~15s not ~90s).
- Learning loop: correction overrides guess; `--general-fact` generalizes.
- Doc pre-analysis + expert review (promote/reject by STABLE id).
- Domain-scope gate: refuses off-domain/injection; context-aware for follow-ups.
- dCor fast O(n log n): ~100x faster, identical results.

### Hardcoding status (honest)
- REMOVED: asset type (now from sources.yaml `asset_type`, fallback "asset");
  protocol context (user-declared in sources.yaml); signal-name mapping (resolved
  via field semantics). Discovery core is fully data-driven.
- REMAINS (acceptable engineering tuning, NOT domain assumptions): grid 10Hz,
  MAX_COLS_PER_SOURCE, sample caps, gap>1h threshold, batch sizes.
- REMAINS (real, to fix): standalone `query fuel_consumption` still special-cases
  EngineFuelRate/litres; the chat `value` path already resolves via field
  semantics instead. Unify by having query resolve from field semantics too.

### Bedrock config (learned live)
- eu-west-1 requires the regional inference-profile prefix, e.g.
  `eu.anthropic.claude-haiku-4-5-20251001-v1:0` (bare IDs fail on Converse).
- Env: `TTMD_LLM=bedrock`, `TTMD_BEDROCK_REGION`, `TTMD_BEDROCK_MODEL`.

---

## 7. Planned work (build order)

DONE since earlier drafts: NL chat orchestrator (prerequisite auto-run, multi-
intent) + vessel-scoped routing; query + plots; describe-fields with roles;
positional / nearby / distance / between; coordinate-based voyage fuel
(cross-source); efficiency (fuel-per-distance value + chart) + general "X per Y";
signal-by-location (geo); summarize (what's-notable) + regimes (usage patterns);
baseline + drift detector (two layers, persisted regime model, chat `anomaly`);
BEHAVIORAL-norm detector (dwell-time; "stayed longer than usual", no baseline);
PRESENTATION MODES (operator/technician/analyst); asset-type/protocol from
sources.yaml. Full deterministic test suite (46 intent cases + routing/voyage/
anomaly/timeparse/resolution/modes checks) green. Scale-tested to a month (~14M
rows/source): query ops sub-second to ~1s, discovery flat (row-capped), only
baseline-build grows with #days. See TODO.md for the full, current list.
Remaining highlights:

1. **Drift detector — 3 remaining extensions** (core is BUILT): (a) regime
   TRANSITIONS as Layer-2 events (needs per-row labels + timestamps through
   discovery); (b) MULTI-TIMESCALE (day/month/year; sudden break vs slow drift);
   (c) FEED DRIFT into the LLM interpretation layer so it explains flagged edges,
   grounded in KB + docs. (See TODO.md P0 REMAINING.)
2. **Voyage by place name** — coordinate-based voyage fuel is DONE (cross-source
   window+integral). REVERSE geocoding is DONE (offline): a user-provided `places:`
   list in sources.yaml (name/lat/lon/radius_km, no network) drives
   nearest_place()/place_label(), so position answers name the port ("Inverness
   Marina") when in range, raw coords otherwise. Auto LEG-DETECTION is DONE:
   spatial.detect_legs segments the track into port-calls (stops) and treats the
   gaps as legs (place-labeled, with distance + duration); "the last voyage" /
   "list voyages" work with NO endpoints (ChatDeps.list_voyages + a last-trip branch
   in voyage() that integrates the rate over the most recent leg window). Remaining:
   FORWARD name->coord for explicit voyage endpoints.
3. **Discovery refinements** — DONE: partial/conditional dependence (prunes
   common-driver haze), time-aware block-permutation significance (flags
   autocorrelation phantoms), fused-clustering blur diagnostic (per-source regimes
   authoritative + fused cross-source relationships + a blurred-silhouette warning).
4. **Retrieval upgrade** — embeddings for document search (current keyword overlap
   pulls ToC noise). **Report formatting** — constrain verbose LLM output.
5. **Unify query fuel_consumption** to resolve signal/unit/aggregation from field
   semantics (remove last domain hardcode; chat path already does this).

---

## 8. Product framing

**"talk-to-my-data" = ask your data what changed, without ever telling it what the
data is.** The differentiator is zero-knowledge onboarding: no manual tagging, no
rule-writing, no schema mapping to start finding anomalies. Point it at the lake,
it learns the asset's normal relational structure per operating mode, and it tells
you when that structure drifts. Deep dependence (nonlinear + cross-source) catches
faults value-thresholds and single-source tools miss.

(Business sizing, pricing, edge-OS/full-chain moat, and data-volume economics are
in MARKET.md — still valid and independent of this method choice.)

---

## 9. How to run

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .
ttmd discover engine       # regimes + per-regime relationship graphs
ttmd discover vibration    # -> artifacts/discovery_<source>.json
```
