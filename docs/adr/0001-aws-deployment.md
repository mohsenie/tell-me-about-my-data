# ADR 0001 — AWS deployment: serverless, near-zero idle cost

- Status: Proposed
- Date: 2026-09-12
- Context owner: (Galene)

## Context

Galene today is a single-tenant CLI: DuckDB reads local Parquet, artifacts and
knowledge are local JSON, and everything runs on demand. The target is a
multi-tenant AWS deployment with one hard constraint:

> **Cost as close to zero as possible when idle** — the ONLY cost we accept when
> no one is using the system is the scheduled background evaluations users
> explicitly set up (watches). No always-on servers, no provisioned databases,
> no idle compute.

Users do NOT trigger heavy analysis interactively. The demanding compute
(discovery, baseline building, drift / anomaly / regime detection) runs only in
the background on a schedule. Interactive use is light: chat to set up / manage
watches and actors, and quick data lookups.

The engine is **DuckDB throughout** (interactive and background), for one mental
model and local-dev/prod parity. See ADR-note at the end for why not Athena.

## Decision

A fully serverless, scale-to-zero design. Three planes; every component bills
per-use (or per-GB stored), so idle cost ≈ storage only.

```
                        ┌─────────────────────────────────────────┐
  user (chat/API)  ───▶ │ INTERACTIVE PLANE                        │
                        │  API Gateway (HTTP) -> Lambda (light)     │
                        │  - watch/actor CRUD, quick DuckDB queries │
                        │  - session state in DynamoDB              │
                        └───────────────┬──────────────────────────┘
                                        │ writes watch rows
                                        ▼
                        ┌─────────────────────────────────────────┐
   EventBridge          │ SCHEDULER PLANE                          │
   Scheduler (tick) ──▶ │  Dispatcher Lambda                       │
                        │  - query watches WHERE next_run <= now    │
                        │  - enqueue each due watch to SQS          │
                        └───────────────┬──────────────────────────┘
                                        │
                          ┌─────────────┴─────────────┐
                          ▼                           ▼
                  SQS light-queue             SQS heavy-queue
                          │                           │
                  Worker Lambda (DuckDB)      Fargate task (DuckDB + numpy/
                  - threshold / value checks    sklearn/dcor) run-and-exit
                          │                     - drift / anomaly / regime
                          └───────────┬───────────────┘
                                      ▼
                          resolve actor -> notify (SNS/SES/webhook)
                          update watch: next_run, last_state (de-dup)

  DATA:  S3 (Parquet telemetry, discovery artifacts, baselines)
  STATE: DynamoDB on-demand (watches, actors, knowledge, session state)
  LLM:   Amazon Bedrock (already the provider)
```

### Idle-cost accounting (the whole point)

| Component | Idle cost | Why |
|---|---|---|
| S3 | storage only | no charge when not read |
| DynamoDB (on-demand) | ~zero | pay-per-request, no provisioned capacity |
| Lambda (interactive + workers) | zero | pay per invocation only |
| Fargate (heavy workers) | zero | tasks run-and-exit; nothing runs idle |
| API Gateway | zero | pay per request |
| EventBridge Scheduler | negligible | the scheduled tick(s) |
| SQS | ~zero | pay per message |
| Bedrock | zero | pay per token, only on use |

**Accepted idle cost = the scheduled tick + the watch evaluations it fires.**
That is the user's explicit choice (they set up the watch). Everything else is
genuinely zero when no one is using it.

Deliberately AVOIDED (all carry idle cost): ECS *services* (always-on),
RDS/Aurora provisioned (hourly even when idle — DynamoDB on-demand instead),
provisioned-concurrency Lambda, NAT Gateways (hourly — keep Lambdas out of
private-subnet-with-NAT where possible, or use VPC endpoints).

## Compute routing: light vs heavy

The decision is made at **watch-creation time** and stored as a `runtime` field
on the watch (stable, validated up front), NOT recomputed at dispatch.

- **Light (Worker Lambda, DuckDB):** a **bounded-window aggregate / integral** —
  e.g. "fuel per journey > 150", "avg coolant > X over last day". Single-pass,
  small memory, seconds. Container-image Lambda (the dep tree is too big for a
  zip layer).
- **Heavy (Fargate task, DuckDB + numpy/sklearn/dcor):** anything that **fits or
  scores a model or scans unbounded data** — drift detection (per-regime dCor
  fingerprints), Mahalanobis / AE joint detection, regime segmentation,
  multi-timescale, baseline building (cost grows with calendar length). Multi-GB
  memory, minutes, no 15-min cap.

Static mapping by condition type: `threshold -> light`, `anomaly -> heavy`,
`regime -> heavy`. **Size guard:** even a light watch is promoted to heavy (or
its window capped) if an estimate of scan bytes (from S3 object sizes / partition
count, computed before launch) exceeds a threshold — protects the light Lambda
from OOM on a surprise "all data" window.

Both lanes import the SAME DuckDB query module, so a threshold check and a
discovery run read Parquet identically. Local dev == prod.

**Heavy = burst, not idle.** Each Fargate task is launched with a per-job CPU/mem
allocation (sized from the scan estimate — a big window gets a big task, a modest
one a small task), does its work for minutes, and EXITS. Nothing runs between
jobs, so heavy compute is "bump to whatever the job needs, then drop to zero."
AWS Batch is the upgrade if heavy volume becomes spiky (queue + retries + spot).

## Heavy INTERACTIVE queries (allowed, but warned / async)

Users don't trigger heavy BACKGROUND analysis, but they may ask a heavy question
in chat. Handle it without any always-on runtime:

1. **Estimate first.** The size-guard estimate (S3 object sizes / partition count)
   gives rough scan bytes -> rough time + cost BEFORE running anything.
2. **Warn-before-proceed.** If over a threshold, the chat says: "to answer this I
   need to scan ~N GB — about 30s to set up plus some compute, roughly $X. Happy
   to proceed?" On confirm, run on a Fargate task and return in chat.
3. **Async alternative — 'email me the result'.** The user can instead choose to
   be emailed. The system schedules a FIRE-ONCE job on the heavy-worker lane (a
   watch with no recurrence), the task runs, generates the result, emails it
   (SES), and EXITS. Chat returns immediately ("I'll email you when it's ready").
   No blocking, no idle cost, and no new infrastructure — it reuses the heavy
   worker path built for watches.

REJECTED for this path: an always-on reserved warm task (e.g. 0.5 vCPU) to make
heavy interactive queries snappy. It gives fast responses but costs ~$15-20/month
PER always-on task (per tenant if isolated), which breaks the near-zero-idle
constraint. If light-path latency ever matters, Lambda cold-start (~1-2s) or
provisioned concurrency is cheaper than a 24/7 task. Kept here only as a possible
future PREMIUM latency tier, not the base design.

## Cost transparency to the user

Extends the existing `UsageMeter` (already tallies Bedrock tokens + est. cost).
Prices come from configurable constants (env), never an unverified rate hardcoded
as fact — same discipline as the LLM-pricing note.

- **After an interactive query — actual metered cost** ("this query cost ≈
  $0.0001"): sum Bedrock tokens (metered) + Lambda GB-seconds (reported by the
  runtime) + S3 GETs / bytes scanned (DuckDB can report). Accurate-ish; labelled
  "≈" for region/rounding.
- **Before scheduling a watch — estimated monthly cost** ("this will cost ≈
  $X/month"): frequency (known from the interval) x per-run cost, where per-run is
  CALIBRATED by a dry run rather than guessed, at the current data volume. Clearly
  an ESTIMATE that rises as data accumulates. Lets the user pick a cheaper interval
  or a lighter condition before committing — a threshold-on-Lambda watch is
  pennies/month, an hourly-anomaly-on-Fargate watch is dollars/month, and showing
  that difference up front is a trust feature.

## Data model (state in DynamoDB)

Watches are ROWS with a schedule, not per-user files (per-user/per-watch files or
per-watch EventBridge rules do not scale — EventBridge has rule/schedule quotas;
files are a hand-rolled scheduler). Table keyed by tenant + watch id:

```
watch = {
  tenant_id, watch_id,
  source, condition_type: "threshold"|"anomaly"|"regime",
  runtime: "light"|"heavy",          # routing (set at create)
  params: {...},                     # e.g. {signal, op:">", value:150, unit:"L", per:"journey"}
  notify: <actor_id or label>,       # resolved via the actors layer
  interval,                          # how often to evaluate
  next_run,                          # dispatcher selects WHERE next_run <= now
  last_state,                        # for DE-DUP: notify only on CHANGE, not every hit
  last_fired, enabled, created_at
}
```

`next_run` = "the database is the schedule" (dispatcher does a query, not a cron
file). `last_state` = the anti-alarm-fatigue mechanism.

Actors, knowledge (field semantics / facts / corrections), and session state
(history, mode, last-intent, pending clarification) also move to DynamoDB,
tenant-namespaced. Telemetry + discovery artifacts + baselines -> S3, per tenant.

## Prerequisite refactor (first build step)

**A storage abstraction.** Today `config.py` builds local paths and the loaders
use `glob.glob()` to enumerate `date=` partitions; stores are local JSON. Cloud
needs: read Parquet from `s3://` (DuckDB httpfs — trivial), **list partitions via
S3 ListObjects** (not `glob` — this is the real work), and JSON stores backed by
DynamoDB/S3. Introduce a store interface with local-FS and AWS backends so I/O
isn't scattered. This is mostly mechanical (code already isolates I/O) and every
later phase depends on it.

## Open decisions (product calls, not plumbing) — MUST resolve before anomaly watches

1. **Baseline freshness.** Drift compares against an operator-designated
   known-good baseline. Background world: is a baseline set once and reused, or
   refreshed periodically? A stale baseline on a slowly-changing asset over/under-
   alarms over months. (Product decision.)
2. **Alert de-duplication.** A background anomaly watch running hourly will re-find
   the same ongoing drift every hour. `last_state` must gate so it notifies on
   CHANGE (new event / cleared), not on every positive evaluation. Invariant #6
   (rank, don't cry-wolf) applies double. Getting this right is what makes
   background alerting usable rather than muted.

## Phased build order

1. **Storage abstraction** (local + S3/DynamoDB backends); keep the CLI working.
2. **Data on S3** (DuckDB httpfs + S3 partition listing) — validates DuckDB↔S3.
3. **Watches: data model + CRUD** (DynamoDB rows) + interactive Lambda + API GW.
4. **Scheduler + dispatcher + SQS + light worker Lambda** (threshold watches end
   to end, with de-dup via `last_state`).
5. **Heavy worker (Fargate)** for anomaly / regime watches (resolve baseline
   freshness first).
6. **Notification delivery** (SNS/SES/webhook), resolved through the actors layer.
7. **Multi-tenancy / auth hardening** (tenant isolation, IAM scoping) — ties to
   the deferred auth iteration.

## Alternatives considered

- **ECS service / task-per-session:** rejected — always-on or per-session idle
  cost + slow startup; a session needs externalized STATE, not a container.
- **RDS/Aurora for state:** rejected — hourly idle cost; DynamoDB on-demand is
  zero-idle and the access patterns are key-value.
- **Athena for the light lane:** genuinely cheaper per-scan and serverless, BUT
  it can't run the heavy analytics (dCor / KMeans / Mahalanobis aren't SQL), so
  we'd run two engines. Chose DuckDB throughout for one mental model + local/prod
  parity; revisit only if light-lane scan volume × frequency makes per-TB cost
  material.
- **Per-user / per-watch schedule files or per-watch EventBridge rules:**
  rejected — hand-rolled scheduler / hits EventBridge quotas. Use DynamoDB
  `next_run` + one dispatcher.
- **Local / desktop execution as a CUSTOMER model** (ship a binary; use the
  customer's compute): rejected — pushes the proprietary method onto machines we
  don't control (code-secrecy problem) and doesn't solve the always-on background
  need. Local execution remains the DEV workflow only.
- **Always-on reserved warm task** (for snappy heavy-interactive): rejected —
  breaks near-zero-idle (~$15-20/mo/task). See the heavy-interactive section;
  kept only as a possible future premium latency tier.

## Consequences

- Idle cost ≈ S3 storage + the scheduled tick users opted into. No idle servers.
- Everything scales to zero and back up per request / per due-watch.
- Cost scales with USE (invocations, due watches, tokens, bytes stored), which is
  the desired property.
- Requires the storage abstraction up front and container-image Lambdas (heavy
  dep tree). Cold starts exist but only bite background jobs, where latency is a
  non-issue.
- Notification delivery and cross-tenant isolation are net-new surfaces; auth is
  still a deferred, separate iteration.

## NOT in scope here

Real-time streaming ingestion (the design is batch/scheduled); predicting future
values (needs conditions the data lacks); local/desktop execution as a customer
model (local DEV stays); and code-protection/IP concerns (moot once the method
runs only in our cloud, never shipped to a client).

Note: heavy INTERACTIVE queries ARE handled (warn-before-proceed or async email) —
see that section. What's out is heavy interactive analysis on an always-on runtime.
```
