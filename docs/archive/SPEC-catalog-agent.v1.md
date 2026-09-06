# Make My Data Talk — System Specification

> Living spec. Captures the product vision, architecture, decisions, and open
> questions so work can resume in a fresh context without re-deriving anything.
> Last updated: 2026-09-05.

---

## 1. Product Vision

**Make My Data Talk** is a **vertically integrated stack that owns the whole
chain** from sensor to spoken answer:

```
data collection (edge) -> edge processing/decode -> compression
  -> storage / data lake (S3 Parquet) -> AI analysis (discovery)
    -> human validation -> live querying (CHAT; voice DEFERRED)
      + on-the-fly charts / graphs / maps
      + agentic actions via a validated ACTOR catalog
```

A customer queries their asset telemetry in natural language and gets answers,
visuals, and (bounded) actions from an AI agent grounded in a validated catalog.

**v1 = text/chat only.** Voice and the "talks like a data scientist" persona are
DEFERRED (see 1c). The value is trustworthy answers + diagnosis; the interaction
modality is the wrapper, not the product. Build/prove the useful core first.

**Key owned assets / moats:**
- A **dedicated edge OS** that runs the software on any edge device, already
  cloud-integrated. (This is the hard, capital-intensive link — already built,
  which materially de-risks the "four companies at once" problem.)
- The **discovery -> validation -> catalog** pipeline (the reusable IP that
  turns messy sensor data into a trusted, queryable semantic layer).
- The **actor catalog** (bounded, validated real-world actions).

**Scope discipline (IMPORTANT):** the pitchable scope is **"any connected-asset
telemetry"**, NOT "any data". The core pipeline + edge OS are domain-agnostic;
domain semantics (AIS/NMEA, engine specs) are **pluggable domain packs**. Going
to market is **one deep vertical at a time**; the engine is horizontal, the
execution is vertical. Avoid the "horizontal-and-shallow" platform trap.

First vertical: **maritime / vessels** (real data; validated as a POC). The ship
data proved the pipeline handles genuinely messy, multi-schema sensor data and
fails safe on the unknown.

### Customer lifecycle

```
Step 0  Install edge device on an asset.
        Edge samples + preprocesses/decodes + writes partitioned Parquet -> S3.
Step 1  Collect ~3 months of data.
Phase 1 DISCOVERY (automated): our system scans + statistically analyzes data,
        optionally enriched by customer documentation, to draft a catalog.
Phase 2 VALIDATION (customer): a domain expert confirms/corrects the findings
        in plain language.
Phase 3 CONFIRMED CATALOG: approved catalog is produced; NL querying goes live.
```

### Target users (critical for UX)

- **Phase 2 validator = domain expert, NOT a data scientist.** A chief engineer
  / fleet manager. Knows the asset; does not know statistics. All validation
  questions must be in domain language. Hide MAD, envelopes, "origin",
  "confidence internals". Show numbers and plain-English claims; ask about
  meaning, units, and plausible ranges.
- **Account manager** curates document uploads and asset association (reduces
  input noise; does not guarantee extraction correctness).
- **Field user (deferred)** voice-only app on/near the asset. HIGH action-risk on
  LOW-fidelity channel (ASR in noise). Deferred out of v1.
- **Office user (v1 interaction target)** logged-in, known identity, chat +
  voice. Does analysis; further from the asset; high input fidelity.

---

## 1b. Actors — the second catalog (action safety foundation)

Alongside the DATA catalog (what can be queried) there is an **ACTOR catalog**
(who/what can be acted upon, and which actions are permitted). Same lifecycle:
discover -> validate -> approve. The agent can only invoke validated actors +
actions; it cannot invent a recipient or a capability. This is
"LLM proposes, code disposes" applied to ACTIONS, not just queries.

The actor catalog is **data-catalog trust apparatus PLUS an access-control &
safety layer** the data side never needed:
- **Authorization**: encodes WHO (by role) may invoke WHICH action under what
  conditions. Enforced in deterministic code below the LLM, never in the prompt.
- **Risk tiers per action**: low (notify/log -> auto + audit) / medium (create
  work order -> confirm) / high (physical/operational -> explicit human confirm,
  possibly forbidden to the agent).
- **One-shot actions vs standing rules**: "notify now" vs "notify IF X recurs".
  Standing rules = persistent trigger + bound action + cry-wolf protection
  (rate-limit/dedup) + expiry/review + audit trail. Standing rules are a
  separate, heavier safety class.
- **Freshness**: validated contacts must be kept current (a stale mechanic
  contact = missed alert).
- **Confirmation protocol**: read-back-before-execute for anything above
  notify/log (mandatory for voice).

**v1 actor boundary (STRONG recommendation): communication/notification actions
ONLY.** The agent may TELL a human; it must NEVER control the physical asset
(engine/steering/valves). This removes catastrophic-harm risk while keeping most
of the value. Physical actions are out of scope until the trust model is proven.

---

## 1c. Interaction model — one brain, thin channels

```
voice app  \
            >-- SHARED CORE (orchestrator, catalogs, actor authz, confirmation) --> query or action
chat app   /
```

- The **channel is I/O only**; reasoning, catalogs, bounding, and safety live in
  a shared core. Never fork logic per channel (divergent safety = incidents).
- **Identity is known in both channels.** Identity -> role + authorized
  asset-scope. This is ALSO the multi-tenancy / data-isolation boundary and MUST
  be enforced in deterministic code below the LLM (a prompt is not a security
  boundary).

### Office chat agent (v1 target): text NL querying + visuals

Command-with-context, in TEXT. A multi-turn, context-carrying chat:
- **Conversational/session state**: memory + reference resolution ("break THAT
  down", "why is IT high?") -> resolve to concrete catalog entities + prior
  results.
- **Clarifies instead of guessing** (the orchestrator's ambiguity path, surfaced
  as natural dialogue).
- **Explains reasoning + caveats** (transparency contract: "best around 720 RPM,
  but only 40 min of data there, so low confidence").
- **Text answer + on-screen chart/map/table.**
- **Latency still matters** (chat should feel responsive) -> favors the hot/cold
  data split, though less strictly than voice would.

### DEFERRED (post-v1): voice + "talks like a data scientist" persona

Removed from v1 scope deliberately. Rationale: for an office user at a screen,
voice adds ASR error, latency pressure, and cost without clearly beating typing;
it is demo-delight, not the value. The charming persona is a wrapper over
correctness, which is the real product. Add voice + persona AFTER trusted
text-based answers/diagnosis are proven with real customers. When revived it
needs: data-to-spoken-summary, spoken read-back confirmation for actions,
domain-vocabulary STT biasing.

### THE two-layer principle (fluid conversation / bounded execution)

The conversation is **fluid**; the actions are **bounded**. The agent chats,
remembers, and phrases naturally, but every time it TOUCHES data or invokes an
action it goes through the typed-tool + validation path. Freedom lives in
dialogue and explanation; rigor lives in execution. They do not conflict because
they operate on different things.

### Graceful bounds — bounded in capability, warm in delivery

Limits are real but must never FEEL like a wall. A good colleague has limits too
and says so gracefully, steering to what they CAN do.

```
out-of-bounds request
  -> deterministic layer returns a STRUCTURED refusal
     {status: unavailable, reason: not_catalogued|not_validated|
      not_authorized|no_data_in_range, alternatives: [...]}
  -> LLM narrates it warmly + offers the alternatives
```

- The LLM narrates the limit; it NEVER decides or removes it.
- **Honesty rule (hard):** phrase the limit nicely, but NEVER manufacture data or
  capability to smooth the conversation. Grace in phrasing; honesty in substance.
- **Explain WHY briefly** ("that sensor's not validated yet, so I won't give you a
  number I can't stand behind") — turns a refusal into trust-building. The
  validation discipline becomes a selling point in conversation.
- **Always offer a path forward** when one exists ("not that, but here's the
  nearest thing I can do"); when none exists, stay honest.
- **Tone**: competent, easygoing professional. Not robotic, not saccharine.
  Persona is a deliberate design task, not left to the base model.

---

## 2. Core Architecture Principles

### Two planes
- **Offline plane** (bootstrap once, then scheduled/triggered): turns raw
  Parquet (+ optional docs) into a reviewed, versioned semantic layer.
- **Online plane** (per user request, fast + bounded): LLM orchestrator + typed
  tools that read the APPROVED catalog and execute bounded queries.

### The one non-negotiable idea
**The LLM proposes; deterministic code disposes.** The model routes and phrases;
it never computes values freehand and never defines "normal". Correctness lives
in deterministic, catalog-driven code.

### Edge fusion onto a common time grid (KEYSTONE decision)
The edge does not just collect — it **fuses all heterogeneous sources onto one
common timestamp grid at a fixed sample rate**, regardless of each source's
native rate:

```
J1939 engine (mixed 1-50 Hz PGNs) \
NMEA nav (various)                 |  EDGE FUSION -> one unified grid @ fixed rate
AIS (event-driven)                 |  (resample/align/hold-last to e.g. 10 Hz)
vibration (kHz, edge-summarized)  /   every source, SAME timestamps
```

**Why this is the keystone:** alignment is solved ONCE, deterministically, at the
source. Downstream data is **pre-aligned by construction**, so:
- Cross-source queries ("fuel vs RPM", "vibration vs engine load") become simple
  JOINs on `timestamp` — NOT messy query-time as-of joins across mismatched rates.
- The hard part of multi-source analytics (time alignment, resampling, phase
  mismatch) never reaches the query builder or the agent. "LLM proposes, code
  disposes" pushed all the way down to the physics layer.
- This is also part of the MOAT: stream-to-cloud competitors ship raw,
  per-source, unaligned streams; we fuse at source.

**Data volume is tiny (MEASURED on real 5-day, 5-source capture):**
- 5 days x 24h x 5 sources (engine, vibration, nmea, ais_own, ais_other) ~= 100 MB
  total -> ~20 MB/vessel/day, ~600 MB/vessel/month, ~7.2 GB/vessel/year.
- 1 TB edge disk -> DECADES of local retention. The edge is a DURABLE LOCAL
  ARCHIVE, not a fragile collect-or-lose buffer: connectivity outages lose
  nothing (offload later), you hold a local ground-truth copy for audit, and you
  can re-process history at the edge if algorithms improve.
- Caveat: this sample had engine-off time (flat signals compress hugely). Active
  operation (rough seas, maneuvering, varying RPM) produces more entropy ->
  budget ~2-4x headroom (40-80 MB/day worst case). Still decades on 1 TB. Quote
  "years with huge margin", not "140 years".

**Sampling faster is nearly free (MEASURED on the real engine file):**
- Parquet (dictionary/RLE + delta + zstd) collapses repeated values. Slow signals
  forward-filled onto a fast grid compress to almost nothing.
- Upsampling the engine data 5x cost only **1.55x the bytes** (not 5x): 5x rows,
  ~1.5x size.
- Per-column: timestamp ~51% and EngineSpeed(RPM) ~38% of the file dominate;
  ALL slow signals combined were ~57 KB of a ~1 MB file (coolant temp = 0.9 KB).
- **Implication:** we can raise the common grid rate to preserve fast signals
  WITHOUT meaningful storage penalty for the slow ones — only genuinely-varying
  fast signals (RPM/torque) and the timestamp cost real bytes. So do NOT split
  into fast/slow/mid files to "save space" (the saving is negligible); keep a
  UNIFIED grid for join-free queries.

**Honest constraints of the common grid:**
1. **Grid rate must exceed the fastest phenomenon of interest**, else it aliases.
   10 Hz DOWNSAMPLES 50 Hz J1939 RPM -> acceptable only if sub-100ms RPM dynamics
   aren't needed. Grid rate is a PRODUCT decision set by the fastest thing you
   claim to detect. (For everything except RPM/torque, 10 Hz is already ample.)
2. **Filled vs real samples**: hold-last-value fills (slow signal on fast grid)
   are repeats, not fresh readings. The catalog SHOULD record per-signal native
   rate + fill method so the agent/query layer never mistakes filled rows for new
   measurements (breaks distinct-counts, rate-of-change, stuck-sensor detection).
3. **Single edge clock**: strength (no cross-device skew) but a single source of
   truth. Needs clock-integrity handling at the edge (monotonic time, drift/NTP-
   jump correction, gap marking on reboot) distinct from real "asset off" gaps.

### Knowledge tiers (drives trust + what can be automated)
- **Tier 1 — data facts** (ranges, sample rate, gaps, null/constant): fully
  automated, trustworthy.
- **Tier 2 — inferred meaning** (name-based guesses, units): PROPOSED,
  confidence-tagged, must be verified.
- **Tier 3 — domain truth** (units, safety limits, operating states, sensor
  mounting, conditional relationships): HUMAN or authoritative-document only.
  NEVER learn safety limits from telemetry.

### Knowledge-source precedence (highest authority first)
```
1. Customer documentation   (datasheets/manuals; when provided) - OPTIONAL
2. Published standards      (AIS ITU-R M.1371, NMEA 0183)
3. Data-derived facts       (Tier 1)
4. Name-based inference      (Tier 2 fallback guess)
```

### Hard invariants (must never be violated)
1. **Nothing is served to users until `review_status == APPROVED`.** The online
   plane loads only the approved catalog. Uncatalogued/unapproved fields are
   invisible, not improvised.
2. **Documents are OPTIONAL enrichment, never a dependency.** The system must
   produce a complete, valid draft catalog with zero documents (it does today).
   Docs only raise per-field quality and reduce Phase-2 expert effort.
3. **No document claim without a citation** (file + page + exact quoted text /
   region). Cite or don't claim.
4. **Extraction is still a proposal**, even from an authoritative doc; it routes
   through Phase-2 confirmation shown next to its citation.
5. **Data-derived "normal" is "normal as observed in window X"**, never ground
   truth. Every envelope states its basis; NOT a safety limit.
6. **Statistics can mislead; standards/domain knowledge override heuristics.**
   (e.g. constant own-ship MMSI is identity, not a dead sensor.)
7. **Document vs data conflicts are surfaced as high-priority review items**,
   never silently resolved.
8. **The agent proposes/computes relationships from data; it never asserts
   domain causation.** Present evidence + ask.
9. **The agent can only invoke VALIDATED actors + actions** (approved actor
   catalog). It cannot invent a recipient, channel, or capability.
10. **Action authorization + asset-scope (tenancy) are enforced in deterministic
    code below the LLM**, never in the prompt.
11. **v1 actions are communication-only.** The agent never controls the physical
    asset. Physical/operational actions are forbidden until the trust model is
    proven.
12. **Two-layer rule: conversation is fluid, execution is bounded.** Natural
    dialogue never becomes freehand querying.
13. **Graceful bounds, honest substance:** the LLM may phrase a limit warmly, but
    NEVER manufactures data or capability to smooth the conversation.
14. **Alignment is solved at the edge, not at query time.** All sources are fused
    onto one common timestamp grid at the source; downstream stays pre-aligned.
    Cross-source queries are timestamp JOINs, never agent-side resampling. Do not
    split into per-rate files (compression makes uniform sampling near-free).

---

## 3. Data Model (observed from real samples)

Sources arrive as partitioned Parquet on S3, named
`<source>-vessel-<id>-<ts>.parquet`. All sampled to a common **10 Hz** grid at
the edge (deliberate, enables clean cross-source joins — phase alignment still
to be verified on overlapping windows).

Source name = filename prefix before `-vessel-` (handles multi-token like
`ais-own`, `ais-other`).

### Sources seen so far
- **engine** (12 fields): EngineSpeed(rpm), EngineFuelRate(L/h, RATE),
  EngineCoolantTemperature, BoostPressure, EngineOilPressure(unit ambiguous
  kPa/psi), FuelLevel, FuelTemperature(-40..210 = sentinels?),
  EngineTotalHoursOfOperation(cumulative), etc. **Intermittent** (engine off =
  gaps; profiler auto-detected a ~5h gap). Operating-state logic mandatory.
- **nmea** (59 fields): GPS/nav/weather. Many `derived_*` / `dhxdr_*` fields are
  EDGE-COMPUTED, not standard — need customer definition. Data-quality issues
  seen: wind direction >360, water_temp_c 100% null.
- **vibration** (14 fields): edge-summarized (accel + FFT `frequency_*`), not raw
  waveform. `temprature` (misspelled) mounting unknown. Many constant channels.
- **ais-own** (11 fields): single MMSI (own ship). Standard AIS position report.
  Sample was "at anchor" (nav_status=1, sog=0).
- **ais-other** (20 fields): MULTI-vessel (25 MMSIs), mixed msg types (1/3/18/24/
  5), + static/voyage fields (name, callsign, imo, ship_type, destination,
  length_m, beam_m, draught_m, part). nav_status ~40% null (Class B/static don't
  carry it — expected).

### Domain traps encoded / to encode
- **Fuel consumption** = time-integral of EngineFuelRate. NOT average, NOT sum.
- **Fuel efficiency** = per-distance/work, NOT per-hour (idle minimizes per-hour).
  Conditional on engine_running AND underway. Definition still unresolved.
- **AIS rate_of_turn** is NONLINEARLY encoded (deg/min ~= (val/4.733)^2). NOT
  deg/min as-is. Flagged low-confidence, needs verification (raw vs decoded).
- **AIS "not available" sentinels**: heading 511, sog 1023, cog 3600, lat 91,
  lon 181, rot 128.
- **ship_type** and **nav_status** are code tables.

---

## 4. What Is Built (as of this spec)

Working Python package `ttmd` (installable, `pip install -e .`). Offline-plane
Phase-1 slice + catalog draft.

```
config.py                     project paths
src/ttmd/
  io/parquet_reader.py         ONLY module that knows how we read data
                               (DuckDB now; swap for DuckDB-on-Lambda later)
  profiling/                   Tier-1 data facts
    stats.py                   pure stats: TimeProfile, FieldStats, Envelope,
                               sample-rate + gap detection, robust median/MAD
                               envelopes. _f() coerces Decimal->float.
    source_profiler.py         orchestrate one source / a directory;
                               _source_name splits on '-vessel-'
  catalog/                     semantic layer
    models.py                  FieldEntry, Inference, Confidence, ReviewStatus.
                               Output key is `meaning` (origin: standard|
                               inference|none) + code_table, standard_valid_range,
                               standard_violation.
    inference.py               Tier-2 name-based GUESS map + seed operating_states
                               + metrics (single review surface for guesses)
    standards/
      ais.py                   AIS field defs, code tables (NAV_STATUS, MSG_TYPE),
                               decode notes, constant_is_ok override, valid_range
    resolvers.py               precedence merge per source (standard > data >
                               name-guess); AIS resolver wired for ais-own/ais-other
    builder.py                 profile -> catalog dict -> YAML; availability
                               override; standard-violation check; embeds code_tables
  cli.py                       `ttmd profile | catalog | all`
data/sample/                   5 sample parquet files
artifacts/                     machine_profile.json, catalog.draft.yaml (gitignored)
```

### Verified behavior
- `ttmd all` runs clean on 5 sources: 90 available / 21 unavailable fields.
- Confidence tally ~ {high:28, medium:16, low:7, none:60}.
- Origin tally ~ {standard:20, inference:31, none:60}.
- Handles structurally very different `ais-other` WITHOUT crashing and WITHOUT
  hallucinating: 10 shared AIS fields resolve from standard; 9 unknown static
  fields correctly flagged `TODO_REVIEW` (fails safe, does not guess).
- Auto-detected engine intermittency (5h gap) purely from data.
- AIS constant MMSI/msg_type/nav_status kept `available` via standards override.

### Known limitations (honest)
- Envelopes derived from a single short, single-condition sample (anchored /
  engine partly off) — structurally correct, NOT yet meaningful "normal". Needs
  the full ~3 months + multiple regimes.
- Constant-detection can't distinguish "broken" from "constant because anchored"
  (SOG/COG marked unavailable at anchor). Needs multi-condition data.
- `ais-other` multi-vessel cardinality not modeled (mmsi as grouping key vs
  identity). Consider a `cardinality` concept.
- Cross-source timestamp phase alignment assumed, not verified.
- AIS static-data fields (name/imo/ship_type/etc.) not yet in `standards/ais.py`.

### Sampling-rate finding (J1939) — informs edge design
Engine source is J1939 fused onto the common grid at ~10 Hz (measured ~5.9 Hz in
this sample). J1939 PGNs have DIFFERENT native rates: RPM/torque ~50 Hz,
coolant/oil ~1 Hz. Measured EFFECTIVE change-rate per signal:
- EngineSpeed ~5.1 Hz (genuinely fast; near grid rate).
- FuelRate/FuelTemp ~0.45 Hz; Boost ~0.16 Hz; OilPressure ~0.09 Hz;
  CoolantTemp ~0.01 Hz (431 changes in 12h).

**This is NOT waste — it's the alignment mechanism** (see "Edge fusion onto a
common time grid" in Section 2). The repeated slow-signal values are the
hold-last fills that put every source on one grid, and Parquet compresses them to
near-zero (all slow signals ~57 KB of ~1 MB). We trade free bytes for guaranteed
time-alignment and join-free cross-source queries. Do NOT "fix" it by splitting
into per-rate files.

Remaining real consideration:
- 10 Hz UNDER-samples 50 Hz native RPM -> aliasing of sub-100ms RPM dynamics.
  Raising the grid rate is cheap for slow signals (measured 5x rows = 1.5x bytes)
  but costs real bytes for RPM/timestamp. Set grid rate by the fastest phenomenon
  the product claims to detect.
- Market implication (see MARKET.md): the fast-transient argument is honest for
  RPM/torque; the durable edge is CONTINUITY + COMPLETENESS + ALIGNMENT-AT-SOURCE
  at low cost, not raw Hz.

---

## 5. Planned Work (not yet built)

### Restructure into three phases (small, clarifying)
```
src/ttmd/
  discovery/     Phase 1 (profiling/analyzers/documents move here)
  validation/    Phase 2
  catalog/       Phase 3 output + approved-only gate
```

### Phase 1 — deepen discovery (document-free, pure data)
Goal: produce a rich, evidence-backed DRAFT so Phase 2 is confirmation, not
investigation. Trade cheap compute for expensive expert time.
- **Correlation discovery**: propose data relationships (r-values) for expert to
  confirm as known physics. Never assert causation.
- **Operating-regime discovery**: cluster data into modes (off/idle/cruise/
  maneuver; anchor/underway). "I found N modes — what is each?"
- **Conditional envelopes**: normal-per-regime (the basis for correct anomaly
  detection later).
- **Data-quality report**: sentinels, stuck sensors, clock issues, cross-source
  alignment. Also feeds back to edge-config fixes.
- **Candidate metrics in domain language**: propose the business questions users
  will ask + how each is computed; expert confirms.
- Exploit the FULL ~3 months, not a slice.

### Phase 1 — document enrichment (OPTIONAL, top-precedence, LLM-dependent)
```
discovery/documents/
  ingest.py    load pdf/img/docx/xlsx/txt -> normalized pages (text + page images)
  extract.py   MULTIMODAL LLM -> claims, EACH with mandatory source citation
  link.py      propose document-claim -> data-field mappings (AM/expert confirm)
catalog/providers/  LLMProvider interface + adapter + stub
```
- File types: pdf, images/photos, word, excel, txt — "official" docs. Many are
  SCANNED / image PDFs and dense tables => requires a **vision-capable** model,
  not just OCR+text. Photos of nameplates feed device identity.
- Optional: no docs -> pipeline runs exactly as today (graceful degradation).
- Value is measurable: "docs pre-answered N of M fields."

### Phase 2 — validation engine (domain expert, plain language)
```
validation/
  queue.py     scan catalog -> risk-prioritized review items
  session.py   accept/correct/reject loop, records decisions + attribution
  decisions.py persist (who/when/what)
```
- Ask about MEANING / UNIT / PLAUSIBLE RANGE / INCLUDE-in-queries — never stats.
- Prioritize: must-review (low-confidence, ambiguous units, violations, ROT,
  unresolved metrics) vs quick-confirm (high-confidence standard) vs auto-skip
  (unavailable). Bulk-confirm-with-exceptions for standard fields.
- Show evidence in human terms (numbers, sample values, sparkline/chart).
- Confidence shown TRANSLATED ("I'm not sure — check this" vs "I'm confident").
- Corrections captured as attributed domain knowledge.
- Build LOGIC-first (CLI-testable) behind a WEB-UI-READY data model. Eventual
  target for non-technical users is a GUI.

### Phase 3 — approval gate + served catalog
```
catalog/
  approve.py   apply decisions -> catalog.approved.yaml (with approved_by/at)
  loader.py    load ONLY approved; refuse drafts (enforced gate)
```

### Online plane (post-catalog; the actual NL querying)
- Orchestrator/state machine: CLASSIFY_INTENT -> (ambiguous?) ASK_CLARIFY ->
  SELECT metric/method (enums from catalog) -> VALIDATE -> EXECUTE -> EXPLAIN.
- Bounding by STRUCTURE not prompts: enumerated tool schemas (constrained
  decoding) + server-side validation + explicit state machine. Prompt is guidance
  only, never the safety boundary.
- Typed tools (examples): get_metric, create_profile, detect_anomalies, plot,
  find_relationship. Enums (ship/source/metric) auto-generated from catalog.
- Anomaly/profiling engine: two-stage PROFILE (train window -> stored versioned
  profile) then DETECT (apply to target window). Fixed MENU of methods (v1:
  univariate envelope + conditional-on-load). Output contract ALWAYS states what
  "normal" was, what data defined it, and confidence.
- Multi-tenancy / per-customer data isolation ENFORCED BELOW the LLM. (Open.)

### Query execution (scale)
- Common queries hit recent, narrow slices (ship/source/date) -> cheap.
- Plan: DuckDB-on-Lambda for hot/selective queries; heavier scans may use Athena.
  Coordinator (list files, partition-prune, shard, merge) is the hard part if
  going custom. Hot/cold split likely. NOT yet built or decided.

---

## 6. Open Decisions (need product owner input)

1. **LLM provider** (blocks ONLY document extraction + online agent, not core).
   Recommendation: **AWS Bedrock** (data stays in AWS boundary; multimodal Claude
   for docs; tool-use for online agent; one vendor for both). Alternatives:
   OpenAI, Anthropic direct, local.
2. **Scanned/image PDFs in v1?** (Yes likely => vision model required.)
3. **Build sequence**: recommended = restructure -> Phase 2 validation (on
   current doc-free catalog) -> deepen Phase 1 analyzers -> documents last.
   Alt: deepen Phase 1 before validation.
4. **Phase 2 interface**: logic-first CLI now behind web-ready model (recommended)
   vs web UI up front.
5. **Efficiency definition** (per-hour vs per-distance/work) for fuel_efficiency.
6. **Operating-state thresholds** (engine_running EngineSpeed>50? underway
   sog>0.5?) — need expert confirmation.
7. **Multi-tenancy model** — how per-customer isolation is enforced below the LLM.
8. **`ais-other` cardinality** modeling (multi-entity source).

---

## 7. How to Run

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .            # duckdb, pyarrow, pandas, pyyaml
ttmd all                    # profile + catalog -> artifacts/
# review artifacts/catalog.draft.yaml
```

Repeatable: `ttmd all` is the offline-plane bootstrap; later triggered on a
schedule for drift/refresh.
```
