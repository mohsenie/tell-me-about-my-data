# Galene — Relational Fingerprinting

> **Galene: Ask the ocean. Find the shore.**

Zero-knowledge anomaly discovery, querying, and data intelligence over asset
telemetry — through a single chat.

Point it at raw, unlabeled sensor data (Parquet). With **no prior knowledge** of
what any signal means, it learns the asset's **operating regimes** and the
**relationship structure** between signals — the "relational fingerprint" of
normal behavior. You then just talk to it: ask what fields exist, compute values,
plot trends, explore correlations, request LLM reasoning, and correct it (it
learns). See `SPEC.md` for the method and invariants.

## The chat is the interface

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .
export GALENE_LLM=bedrock          # real reasoning (else a stub keeps it runnable)

galene chat               # vessel-scoped; defaults to the only/first vessel
galene chat vessel-001    # or name the vessel explicitly
```

### Choosing a persona mode when you open the chat

The chat tailors HOW it explains things to your audience (the numbers/analysis are
identical across modes — only the wording and level of detail change):

```bash
galene chat --mode general      # DEFAULT: business/operations — short, plain, no jargon
                              #   e.g. "the ship is staying in one place longer than usual"
galene chat --mode analyst      # which signals/couplings changed + what to check
galene chat --mode expert       # full raw detail: the numbers verbatim

galene chat vessel-001 --mode general           # combine with a vessel
galene chat vessel-001 --source engine --mode expert
```

You can also switch mid-conversation — just say **"switch to expert mode"**,
**"use analyst view"**, or **"explain like a general/business user"**. The three
modes (and their synonyms):
- **general** (DEFAULT) — plain, business/operations wording, no jargon
  (aliases: business, operations, ops, manager, captain, exec, operator, plain).
- **analyst** — concise + concrete: which signals/couplings changed, what to check
  (aliases: technician, tech, engineer, maintenance, mechanic).
- **expert** — full raw detail, verbatim, no reframing
  (aliases: data, detailed, raw, full).

Modes only reshape the INTERPRETIVE answers (what's-notable, anomaly, regimes,
reasoning); a plain value or plot is the same in every mode.

The chat is **vessel-scoped**: you just name the vessel (or nothing), and every
question is **routed automatically to whichever source can answer it** — you
never name or switch sources by hand. Ask where the ship is (routes to the
GPS/AIS source), what's nearby (routes to the multi-vessel AIS source), engine
fuel (routes to engine), or what correlates with a signal (routes to the source
that owns it), all in one conversation. A home source is auto-picked (the most
signal-rich one) for questions that don't name a specific signal; override with
`--source` if you want.

The chat also **orchestrates everything** — it checks prerequisites and, if a
phase (discovery, field descriptions) hasn't run, it **runs it automatically**
the first time a question needs it (with a short inline notice; no yes/no gate).
Then ask in plain language:

```
you> what correlates with rpm?
assistant> (first analyzed 'engine' to find its relationships)
           EngineFuelRate: 0.99, BoostPressure: 0.96, ...   (← discovery ran automatically)
you> average fuel rate over the past 2 days
assistant> (Described 10 fields for 'engine' (protocol: J1939).)
           avg EngineFuelRate: 7.68 L/h (window 2026-09-03..09-04, 499k rows).
you> plot average coolant temperature every 4 hours
you> why is fuel temperature related to engine speed?
you> no — it's because the fuel line runs near the turbocharger   (← it learns this)
you> where is the ship currently?              (← auto-routes to the AIS/GPS source)
assistant> Most recent position (2026-09-05 12:02 UTC): 57.48634, -4.25040 (lat, lon).
you> list the ships nearby yesterday at 13:00   (← auto-routes to the multi-vessel source)
assistant> 16 vessel(s) within 50 km of own-ship at 2026-09-04 13:00 UTC: ...
you> how much fuel from 58.48,-2.83 to 57.49,-4.25   (← voyage: window from the track, fuel on engine)
assistant> EngineFuelRate used from (58.4800, -2.8300) to (57.4900, -4.2500): 106.8 L.
           Window 2026-09-02 18:43 → 2026-09-04 12:04 UTC (41.4 h), track ~261 km
           [from 'ais-own' position, integrated on 'engine'].
```

Voyage queries are **cross-source**: the trip's time window is derived from the
position track (the closest approach to your start and end points), and the fuel
rate is integrated over that window on whichever source owns it. Give two
coordinates today; place names ("Gdynia to Scotland") need a geocoder, which is a
deferred, opt-in step (nothing about places is hardcoded).

**Time windows:** if you don't give a time range, queries default to **all
available data** (not just the latest) — for a stable "average". The answer always
states the window it used; narrow with "over the past N days" or a date range.
(For per-distance figures like fuel/km, "all data" includes idle periods, so
narrow the window if you want an underway-only figure.)

Intents the chat routes to (deterministic compute, LLM only routes/reasons):
capabilities · field descriptions · values · plots · correlations · reasoning ·
corrections · position · nearby · voyage · efficiency (fuel per km, value or a
per-distance chart) · distance (how far travelled) · between (distance between two
named vessels) · geo (signal by location) · anomaly · summarize (what's notable) ·
regimes (operating modes / usage patterns). Off-domain questions are refused.

**Behavioral-norm alerts:** beyond sensor-fault drift, it watches how the asset is
*operated* vs its own history and flags departures in plain terms — e.g. "the ship
has stayed in one location ~24h, about 37x its usual stay." Surfaced first when you
ask "is there anything to worry about". Needs no baseline (the norm is the asset's
own past); it reports the change, never the cause.

**Presentation modes:** `galene chat --mode general|analyst|expert` (or "switch to
expert mode" mid-chat) tailors HOW answers are explained — general (default) gives
short plain-language framing ("the ship is staying in one place longer than
usual"), analyst gives which-signals/what-to-check detail, expert gives the full
raw numbers verbatim. Same underlying analysis; only the wording/detail changes.

Ask "what is notable in my engine data" / "give me an overview" for a short
prioritized summary — it folds in drift if you've set a baseline, else summarizes
the discovered structure. It reports what stands out, never asserting the cause.

"fuel consumption"/"used"/"burnt" resolves to the fuel RATE (integrated) — no
"which fuel?" ambiguity. "How many km did we travel" traces the position track
(not an integral of a speed column). Meta questions naming a source ("what data
from ais-own", "describe the vibration fields") route to that source. A CHART can
be windowed to a leg ("plot velocity_z from <coord> to <coord> every 30 min").
A relationship question spanning TWO sources ("relation between vibration and
pitch") uses fused cross-source discovery. "Is vibration higher in some
locations?" maps a signal (or a concept family like all vibration axes) over
geographic cells of the track and reports where it's high/low (sea-state by area).
Per-distance charts work for ANY
signal, not just fuel: "plot velocity_z every 10km" averages it per distance
segment (a rate like fuel is integrated instead). And "X per Y" generalizes
beyond distance: "fuel per operating hour", "fuel per MWh", "X per revolution" —
Y can be distance, a rate, or a cumulative counter, all resolved from the data.

## The phases (also runnable as direct commands)

```
INGEST DATA  parquet in ship-data/  (edge-collected, time-fused)
  -> DISCOVER      regimes + relationship graph (Pearson+dCor+MI), zero-knowledge
  -> DESCRIBE-FIELDS  LLM proposes field meanings using user-declared protocol
  -> REPORT / INTERPRET  plain report + LLM root-cause hypotheses (cited, bounded)
  -> QUERY / PLOT  deterministic values + charts on demand
  -> BASELINE / DETECT  save a known-good fingerprint, then flag drift from it
  -> LEARN         expert corrections + promoted doc facts improve reasoning
```

```bash
# discovery (no LLM)
galene discover engine [--days 2 | --from .. --to ..]
galene fused                             # auto-fuse all sources -> cross-source
galene report engine

# anomaly detection (no LLM): set a known-good baseline, then flag drift from it
galene baseline engine --from 2026-09-01 --to 2026-09-02   # a period you consider healthy
galene detect  engine --days 2           # compare recent window to the baseline
#   -> layers: within-mode relationship drift (possible fault) + regime events
#      (usage change) + SEQUENCING change (the order modes occur in — new/missing
#      steps); reports the change, never asserts the cause

# querying / plotting (deterministic, no LLM)
galene capabilities [engine]             # what signals exist (+ why some aren't queryable)
galene query engine avg --signal EngineSpeed --days 2
galene plot engine trend --signal EngineCoolantTemperature --agg avg --bucket 4h --days 2
galene plot engine scatter --x EngineSpeed --y EngineFuelRate

# meaning + reasoning + learning (needs GALENE_LLM=bedrock)
galene describe-fields engine            # protocol-aware field meanings (reviewable)
galene interpret engine                  # report + cited LLM hypotheses
galene correct "EngineSpeed" "FuelTemperature" "reason" --general-fact "asset truth"
galene ingest-docs                       # pre-analyze user-documentation/*.pdf -> facts
galene review-docs list                  # review learned facts by tier (stable ids)
galene review-docs promote <id>          # extracted -> expert-confirmed
galene review-docs reject <id>           # remove a wrong extracted fact
galene chat --usage                      # track LLM tokens + estimated cost this session
```

## Data & metadata (user-provided, nothing hardcoded)

```
ship-data/<vessel>/<source>/logs/date=YYYY-MM-DD/*.parquet   telemetry
ship-data/<vessel>/sources.yaml                              user-declared context
```
`sources.yaml` declares `asset_type` and, per source, its `protocol` (J1939 /
NMEA 0183 / AIS / custom ...) and description. This is **user input**, not code —
the system reads it. Drop a new source folder in and it's auto-discovered with
zero code changes. Missing metadata -> graceful degradation (lower-confidence
name-only field guesses).

## Knowledge tiers (trust hierarchy)

Reasoning combines tiers ranked by authority; a wrong lower tier never overrides:
1. **Expert-confirmed** — corrections + promoted doc/field facts (authoritative).
2. **Document-extracted** — LLM-pulled from manuals, cited, **unverified**.
3. **Name/LLM proposal** — field-name guesses, hypotheses; speculative, labeled.

The LLM proposes/routes/reasons; deterministic code computes; the expert confirms.

## Layout

```
config.py                    paths, source + asset-type + protocol metadata, windows
src/galene/
  cli.py, cli_helpers.py, chat_deps.py, term.py
  io/                        DuckDB data access
  discovery/                 THE METHOD (dependence, regimes, relationships, fusion)
  anomaly/                   baseline + drift detector + behavioral-norm detector
    baseline.py / detector.py / report.py / behavioral.py
  reporting/                 deterministic human-readable report
  query/                     deterministic values + plots + spatial (compute, capabilities,
                             plots, spatial: position/nearby/voyage/distance/geo/segments)
  interpretation/            LLM layer (bounded to domain)
    provider / knowledge / documents / fields / interpret / scope / resolve
    orchestrator.py          the chat brain: intent + prerequisites + dispatch + modes
  chat_deps.py               capability bridge: every intent's implementation
ship-data/                   partitioned telemetry + sources.yaml
user-documentation/          customer manuals (e.g. DPX-600.pdf)
artifacts/                   discovery_*.json, report_*.md, knowledge_*.json, plot_*.png
```

## LLM (Amazon Bedrock)

```bash
export GALENE_LLM=bedrock
# defaults: eu-west-1, eu.anthropic.claude-haiku-4-5 (regional inference profile)
# override: GALENE_BEDROCK_REGION, GALENE_BEDROCK_MODEL
```
Without a provider, a StubProvider keeps the non-LLM pipeline runnable.

## Tests

```bash
pip install pytest pytest-cov
python -m pytest tests/                       # functional suite (no LLM needed)
python -m pytest tests/ --cov --cov-config=.coveragerc   # with coverage
GALENE_LLM=bedrock python -m pytest tests/      # also run the live intent-classification case
```
Deterministic functional tests assert OUTCOMES (computed values, files created,
structure) on the sample vessel data — no live LLM required (a `FakeProvider`
drives the dispatch tests). ~64% coverage of the functional library; the
correctness-critical compute (discovery, query, spatial, anomaly) is 85-95%.
LLM-synthesis prose paths are intentionally light (non-deterministic output).

## Notes

- Distance correlation uses the fast O(n log n) algorithm (`dcor`), ~100x faster
  than naive, identical results. See `src/galene/discovery/README.md`.
- **Direct vs indirect links:** partial correlation (controlling for all other
  signals) flags edges that are strong only because of a shared driver (e.g. two
  pressures that both track engine load), so the report separates a real
  relationship from common-driver haze.
- **Autocorrelation-aware significance:** a time-aware block-permutation test flags
  edges that look dependent only because both signals drift slowly (a shared trend),
  so phantom relationships are marked instead of trusted.
- **Honest fusion:** fusing dissimilar sources blurs joint clustering, so cross-source
  discovery reports the cross-source RELATIONSHIPS (the trustworthy output) and warns
  when the fused operating-modes are low-separation — per-source modes stay authoritative.
- **Place names, offline:** add a `places:` list (name/lat/lon) to `sources.yaml` and
  position answers name the nearest port ("Inverness Marina") instead of raw
  coordinates — no network, entirely from your list; raw coords when nothing is close.
- **Voyages, auto-detected:** ask "list my voyages" or "how much fuel on the last
  voyage" — it segments the track into port-calls, treats the gaps as legs
  (place-named, with distance and duration), and totals the rate over the most
  recent leg without you giving any coordinates.
- **Three anomaly detectors, all built:** (1) *relational drift* — `galene baseline`
  then `galene detect`, or ask "has anything drifted" — flags within-mode
  relationship-structure change (possible fault) + regime events (usage change) +
  regime-transition/sequencing change (the order operating modes occur in — a
  usual step missing or a new jump appearing), vs a known-good baseline;
  (2) *behavioral norm* — flags how the asset is
  OPERATED vs its own history (e.g. dwell-time: "stayed somewhere ~37x longer than
  usual"), no baseline needed; (3) *joint multivariate* — per-regime Mahalanobis
  covariance envelope that flags points whose COMBINATION of readings is unusual
  for their mode (even when each signal alone is in range), attributing the score
  to the signals that drove it. All report the observed change, never the cause.
- **Scales to a month+ with no changes** (measured): query ops stay sub-second to
  ~1s on ~14M rows/source; discovery is flat (row-capped sampling); the only cost
  that grows with calendar length is baseline-building (per-day loop). See
  `TODO.md` for remaining hardening (geocoding place names, discovery-quality
  refinements).
- **Optional autoencoder backend** (`galene detect --ae`, off by default): a per-regime
  MLP autoencoder for regimes whose normal region is a nonlinear manifold a
  covariance ellipsoid can't fit; reconstruction error scores each point, per-feature
  error keeps the attribution. Opt-in because it trades the training-free character
  for nonlinear reach.
- **Math finds it, your labels explain it:** the detectors point at raw signal
  columns; the field-semantics JSON then translates each into a human meaning on the
  finding ("coolant temperature (degC) [EngineCoolantTemperature] 60%"). The two are
  decoupled — describing your fields makes every anomaly readable, and an undescribed
  column just shows its raw name (never a guess).
- **Ask "why?"** after an anomaly answer and it EXPLAINS the detected change —
  grounded in the structured findings (which couplings/steps moved), your
  expert-confirmed facts, and manual excerpts — offering what to check, never
  asserting a cause.
- **Multi-timescale** (built): groups periods (day/month/year) and separates a
  *sudden break* (an abrupt step between adjacent periods) from *slow drift* (a
  gradual accumulation vs the earliest period); folded into the summary/reasoning.
```
