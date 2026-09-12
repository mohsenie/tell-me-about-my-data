"""Functional tests for the watch feature: registry (CRUD + scheduling + de-dup),
evaluation (threshold fire/no-fire), and the run loop (fire-then-quiet). Threshold
tests use the real vessel data with ground-truth bounds (avg EngineFuelRate ~7.61,
so >5 fires and >20 doesn't). No LLM. Temp stores — artifacts/ untouched.
"""
from __future__ import annotations

import time

import pytest

import config
from galene.anomaly.watches import WatchRegistry, CONDITION_TYPES
from galene.anomaly.watch_eval import evaluate, run_watches


def _reg(tmp_path):
    return WatchRegistry(tmp_path / "watches.json")


# ---------------- registry: CRUD + scheduling ----------------
def test_add_and_idempotent(tmp_path):
    r = _reg(tmp_path)
    p = {"signal": "EngineFuelRate", "aggregation": "avg", "op": ">", "value": 5}
    w = r.add("engine", "threshold", p, notify="Andrew", notify_actor_id="a1")
    assert w["condition_type"] == "threshold" and w["runtime"] == "light"
    assert w["next_run"] == 0 and w["last_state"] is None
    r.add("engine", "threshold", p)          # same definition -> update, no dupe
    assert len(r.all()) == 1


def test_runtime_routing_tag(tmp_path):
    r = _reg(tmp_path)
    assert r.add("engine", "threshold", {"signal": "x", "value": 1})["runtime"] == "light"
    assert r.add("engine", "anomaly", {})["runtime"] == "heavy"
    assert r.add("engine", "regime", {})["runtime"] == "heavy"


def test_due_and_delete(tmp_path):
    r = _reg(tmp_path)
    w = r.add("engine", "threshold", {"signal": "x", "value": 1})
    assert w in r.due()                       # next_run 0 -> due immediately
    assert r.delete(w["id"]) is True
    assert r.all() == []


def test_record_evaluation_dedup(tmp_path):
    """State change gates notification; unchanged state stays quiet; advancing
    next_run means it isn't due again until the interval passes."""
    r = _reg(tmp_path)
    w = r.add("engine", "threshold", {"signal": "x", "value": 1}, interval_s=3600)
    now = time.time()
    assert r.record_evaluation(w["id"], "breach:1", now=now) is True    # ok->breach
    assert r.record_evaluation(w["id"], "breach:1", now=now) is False   # unchanged
    assert r.record_evaluation(w["id"], "ok", now=now) is True          # breach->ok (resolved)
    # next_run advanced -> no longer due right now
    assert w["id"] not in [x["id"] for x in r.due(now=now)]


# ---------------- evaluation: threshold ----------------
def test_threshold_fires_and_holds(globs, has_data):
    w = {"id": "t1", "source": "engine", "condition_type": "threshold",
         "params": {"signal": "EngineFuelRate", "aggregation": "avg",
                    "op": ">", "value": 5, "unit": "L/h"}}
    out = evaluate(w, config.DEFAULT_VESSEL)
    assert out["fired"] is True
    assert out["state"].startswith("breach")
    assert out["value"] > 5


def test_threshold_does_not_fire(globs, has_data):
    w = {"id": "t2", "source": "engine", "condition_type": "threshold",
         "params": {"signal": "EngineFuelRate", "aggregation": "avg",
                    "op": ">", "value": 20}}
    out = evaluate(w, config.DEFAULT_VESSEL)
    assert out["fired"] is False and out["state"] == "ok"


def test_threshold_missing_params_is_error(has_data):
    w = {"id": "t3", "source": "engine", "condition_type": "threshold",
         "params": {"signal": "EngineFuelRate"}}     # no value
    out = evaluate(w, config.DEFAULT_VESSEL)
    assert out["fired"] is False and out["state"] == "error"


# ---------------- run loop: fire once, then de-dup quiet ----------------
def test_run_watches_fires_then_dedups(tmp_path, has_data):
    r = _reg(tmp_path)
    r.add("engine", "threshold",
          {"signal": "EngineFuelRate", "aggregation": "avg", "op": ">",
           "value": 5, "unit": "L/h"}, notify="Andrew", notify_actor_id="a1")
    r.add("engine", "threshold",
          {"signal": "EngineFuelRate", "aggregation": "avg", "op": ">", "value": 20})
    # run 1: both due; the >5 fires, the >20 stays ok
    s1 = run_watches(config.DEFAULT_VESSEL, r)
    assert s1["evaluated"] == 2 and len(s1["fired"]) == 1
    assert s1["fired"][0]["notify"] == "Andrew"
    # run 2, forced due far in the future: states unchanged -> no new alerts
    s2 = run_watches(config.DEFAULT_VESSEL, r, now=time.time() + 10 ** 6)
    assert s2["evaluated"] == 2 and len(s2["fired"]) == 0


def test_anomaly_watch_without_baseline_is_quiet(tmp_path, has_data):
    """An anomaly watch on a source with no baseline reports 'no_baseline' and
    does not fire a false alert."""
    r = _reg(tmp_path)
    # pick a source unlikely to have a baseline in the sample artifacts
    r.add("vibration", "anomaly", {})
    s = run_watches(config.DEFAULT_VESSEL, r)
    assert len(s["fired"]) == 0                # no baseline -> no alert
