"""Watch EVALUATION — given a watch definition, compute its condition against the
current data and return an outcome. Separated from the registry (store/scheduling)
and from delivery, so this core is unit-testable offline and runs identically
behind a cron line locally or a worker in the cloud.

Reuses the EXISTING deterministic machinery — nothing new analytically:
  - threshold : query.compute.aggregate over a window vs a bound (LIGHT).
  - anomaly   : anomaly.detect_drift vs a known-good baseline (HEAVY).
  - regime    : the regime-event layer of the same drift result (HEAVY).

An outcome is:
  {
    "state": <str>,     # de-dup key: what the runner compares to last_state.
                        # "ok" when the condition does NOT hold; a stable
                        # signature string when it DOES (so an unchanged ongoing
                        # condition doesn't re-alert).
    "fired": <bool>,    # convenience: state != "ok".
    "detail": <str>,    # human-readable, for the alert body. Observed, not cause.
    "value": <float|None>,
  }
Never asserts a cause (invariant): the detail reports the observed condition.
"""
from __future__ import annotations

import glob as _glob

import config
from galene.query.compute import aggregate


def _present(globs):
    return [g for g in globs if _glob.glob(g)]


def _window_globs(source, vessel, days=None):
    from galene.cli_helpers import resolve_window, present_globs
    globs, label = resolve_window(source, vessel, days=days)
    return present_globs(globs), label


def evaluate(watch: dict, vessel: str, kb=None) -> dict:
    """Evaluate one watch against current data. Dispatches by condition_type."""
    ct = watch.get("condition_type")
    if ct == "threshold":
        return _eval_threshold(watch, vessel)
    if ct in ("anomaly", "regime"):
        return _eval_drift(watch, vessel, kind=ct)
    return {"state": "error", "fired": False,
            "detail": f"unknown condition type '{ct}'", "value": None}


def _eval_threshold(watch: dict, vessel: str) -> dict:
    """A computed aggregate crossing a user-given bound. Data-driven: signal,
    aggregation, operator, value + optional window come from the watch params."""
    p = watch.get("params", {})
    source = watch["source"]
    signal = p.get("signal")
    op = p.get("op", ">")
    bound = p.get("value")
    agg = p.get("aggregation") or "avg"
    days = p.get("days")
    unit = p.get("unit", "")
    if signal is None or bound is None:
        return {"state": "error", "fired": False,
                "detail": "threshold watch needs a signal and a value", "value": None}

    present, label = _window_globs(source, vessel, days=days)
    if not present:
        return {"state": "ok", "fired": False,
                "detail": f"no data for '{source}' in the window", "value": None}
    res = aggregate(present, signal, agg, label, unit=unit)
    val = res.value
    if val is None:
        return {"state": "ok", "fired": False,
                "detail": f"no {signal} readings in the window", "value": None}

    breached = _compare(val, op, bound)
    if breached:
        u = f" {unit}" if unit else ""
        return {
            "state": f"breach:{agg}:{signal}:{op}:{bound}",   # stable while it holds
            "fired": True,
            "detail": (f"{agg} {signal} = {val:,.2f}{u} {op} {bound}{u} "
                       f"(window {label}). Observed threshold crossing — not a cause."),
            "value": float(val),
        }
    return {"state": "ok", "fired": False,
            "detail": f"{agg} {signal} = {val:,.2f} (within bound, window {label})",
            "value": float(val)}


def _compare(val, op, bound) -> bool:
    try:
        b = float(bound)
    except (TypeError, ValueError):
        return False
    return {
        ">": val > b, ">=": val >= b, "<": val < b, "<=": val <= b,
        "==": val == b, "!=": val != b,
    }.get(op, val > b)


def _eval_drift(watch: dict, vessel: str, kind: str) -> dict:
    """Relational drift (anomaly) or regime change vs the known-good baseline.
    Reuses detect_drift; the state signature summarizes WHAT changed so an
    unchanged ongoing drift doesn't re-alert."""
    from galene.anomaly import has_baseline, load_baseline, detect_drift
    source = watch["source"]
    dates = config.available_dates(source, vessel)
    if not dates:
        return {"state": "ok", "fired": False,
                "detail": f"no data for '{source}'", "value": None}
    if not has_baseline(source):
        return {"state": "no_baseline", "fired": False,
                "detail": (f"no known-good baseline for '{source}' — set one with "
                           f"`galene baseline {source} --from .. --to ..` to enable "
                           "this watch"), "value": None}
    baseline = load_baseline(source)
    base_dates = set(baseline.get("dates", []))
    recent = [d for d in dates if d not in base_dates] or dates[-2:]
    result = detect_drift(source, recent, vessel, baseline, with_mi=False)

    if kind == "anomaly":
        l1 = result.get("layer1_structure_drift", [])
        # signature: which regimes drifted + how many edges each (stable while the
        # same drift persists; changes when the drift changes -> re-notify)
        sig_parts = [f"r{f['regime']}:{len(f.get('changed_edges', []))}" for f in l1]
        if not sig_parts:
            return {"state": "ok", "fired": False,
                    "detail": f"no relationship-structure drift in '{source}'",
                    "value": 0.0}
        n_edges = sum(len(f.get("changed_edges", [])) for f in l1)
        return {
            "state": "drift:" + ",".join(sorted(sig_parts)),
            "fired": True,
            "detail": (f"relationship-structure drift in {len(l1)} mode(s) "
                       f"({n_edges} coupling change(s)) vs baseline "
                       f"{baseline.get('window')}. Observed change — not a cause."),
            "value": float(n_edges),
        }
    # regime: usage/regime events (distribution shift, new mode)
    l2 = result.get("layer2_regime_events", [])
    if not l2:
        return {"state": "ok", "fired": False,
                "detail": f"no regime/usage change in '{source}'", "value": 0.0}
    kinds = sorted({e.get("type", "event") for e in l2})
    return {
        "state": "regime:" + ",".join(kinds),
        "fired": True,
        "detail": (f"{len(l2)} regime/usage change(s) vs baseline "
                   f"({', '.join(kinds)}). Observed change — not a cause."),
        "value": float(len(l2)),
    }


# --- alert action (honest default: write a structured alert; delivery deferred) ---
def _emit_alert(watch: dict, outcome: dict, vessel: str) -> dict:
    """Produce a structured alert record for a fired watch, targeted at the
    resolved actor, and append it to artifacts/alerts.log (JSONL). Real delivery
    (email/webhook) is a separate opt-in step — this is the honest default action
    (a durable, scriptable record). Returns the alert dict."""
    import json as _json
    import datetime as _dt
    alert = {
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "vessel": vessel,
        "watch_id": watch["id"],
        "source": watch["source"],
        "condition_type": watch["condition_type"],
        "state": outcome["state"],
        "detail": outcome["detail"],
        "value": outcome.get("value"),
        "notify": watch.get("notify"),
        "notify_actor_id": watch.get("notify_actor_id"),
    }
    log = config.ARTIFACTS_DIR / "alerts.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as fh:
        fh.write(_json.dumps(alert) + "\n")
    return alert


def run_watches(vessel: str, registry, kb=None, now: float | None = None) -> dict:
    """Evaluate all DUE watches for the vessel, fire alerts ONLY on a state change
    (de-dup via the registry), and advance next_run. Stateless + idempotent: safe
    to call as often as a cron/tick likes — a watch only evaluates when its
    next_run has passed, and only notifies when its state changed.

    Returns a summary {evaluated, fired:[alerts], quiet:[watch_ids], errors:[...]}.
    This is the single entry point a `galene watch run`, a cron line, or a cloud
    worker all call."""
    due = registry.due(now=now)
    fired, quiet, errors = [], [], []
    for w in due:
        try:
            outcome = evaluate(w, vessel, kb=kb)
        except Exception as e:  # a bad watch must not sink the whole run
            errors.append({"watch_id": w["id"], "error": f"{type(e).__name__}: {e}"})
            continue
        changed = registry.record_evaluation(w["id"], outcome["state"], now=now)
        if changed and outcome.get("fired"):
            fired.append(_emit_alert(w, outcome, vessel))
        else:
            quiet.append(w["id"])
    return {"evaluated": len(due), "fired": fired, "quiet": quiet, "errors": errors}
