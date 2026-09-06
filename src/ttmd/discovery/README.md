# Discovery — unsupervised structure learning from raw data

Learns the structure of an asset's telemetry **with zero labels**. It answers
"how do these signals relate, and what operating modes exist?" without knowing
what any signal *means*. Its output feeds interpretation (LLM reasoning),
querying (relationship lookups / scatter plots), and the regime-conditional
anomaly layer (per-regime baselines).

Philosophy: **structure first (automatic), meaning second (human-validated).**

---

## What it produces

1. **Signal dependence** — for every pair of signals, three measures:
   - **Pearson** — linear strength (blind to nonlinear relationships).
   - **Distance correlation (dCor)** — ANY dependence; 0 iff independent.
   - **Mutual information (MI)** — shared information; nonlinear-capable.
   Comparing them classifies each relationship's KIND: `linear` / `nonlinear` /
   `none`. The `nonlinear` case (low Pearson, high dCor) is the one a plain
   correlation heatmap MISSES — that's the whole reason all three are used.

2. **Operating regimes** — unsupervised segmentation of time into modes
   (e.g. cruise / idle / cold-start), via clustering. This SEPARATES usage
   patterns from anomalies: later, anomaly detection compares like-for-like
   regime, never across regimes.

3. **Per-regime relationship graphs** — the relationship graph computed WITHIN
   each regime, not blurred across all data. These are the trustworthy edges:
   relationships that hold in a given operating mode.

---

## Why it works (proven on real engine data)

- **Found RPM<->fuel and the full engine coupling from numbers alone** — no
  labels. Pearson ~0.97, dCor ~0.99.
- **Caught relationships Pearson missed** — e.g. FuelLevel<->FuelTemperature:
  Pearson 0.07 ("unrelated") but dCor 0.38, significant. The zero-linear-
  correlation blind spot, solved.
- **Segmented 3 real operating modes with no labels**: cruise (RPM~1720),
  idle (RPM~827), cold-start (coolant only 56 vs 76). Silhouette ~0.61.
- **Showed relationships CHANGE by regime** — several edges are linear globally
  but nonlinear within the cruise regime. This is exactly why per-regime
  baselines matter: a global model misjudges what is normal.

---

## Modules

| File | Role |
|---|---|
| `dependence.py` | Pearson / dCor / MI measures + `classify()` -> Kind |
| `regimes.py` | `segment_regimes()` — KMeans + silhouette; usage separator |
| `relationships.py` | `build_graph()` / `build_per_regime_graphs()` — the graph |
| `loader.py` | thin numeric-data loader for a source |

Run: `ttmd discover <source>` (e.g. `ttmd discover engine`) ->
`artifacts/discovery_<source>.json` with regimes + per-regime graphs.

---

## How it fits the product

Discovery is the foundation the other layers build on:

- **Feeds interpretation** (`ttmd interpret` / `chat`): the LLM reasons about the
  discovered relationships, grounded in the knowledge base + user docs.
- **Feeds anomaly detection** (planned): per-regime baseline graphs are what a
  live window is compared against — anomaly = deviation from the SAME-regime
  baseline.
- **Grounds querying**: relationship lookups ("correlation of X and Y") and
  scatter plots read the discovered graph.

---

## Honest limits

- **Structure, not meaning.** Knows "signal A relates to signal B", not that they
  are fuel and RPM. Meaning still needs human/document validation.
- **Discovered edges are candidates, not truths.** They pass through the same
  NEEDS_REVIEW -> APPROVED gate as catalog fields before the agent may use them.
- **Undirected.** dCor/MI/Pearson are symmetric — no causal direction. Directed
  edges (lag/transfer entropy) are separate, and remain HYPOTHESES.
- **Common-driver haze.** On an engine, most signals co-move with load, so graphs
  look densely connected. Partial/conditional dependence is needed to find
  DIRECT edges vs "related only via load". (Refinement, not yet implemented.)
- **Autocorrelation artifacts.** Slow-drifting signals (e.g. fuel temperature)
  can show phantom dependence. Time-aware (block) permutation significance is
  needed to confirm edges are real. (Refinement, not yet implemented.)
- **Regime quality depends on data coverage.** A rarely-seen mode has a weak
  baseline; a genuinely new mode looks anomalous (novelty != fault).
