"""Watch registry — named, persisted watches that fire an alert when a CONDITION
holds. This module owns the STORE + SCHEDULING bookkeeping only; the actual
condition EVALUATION lives in watch_eval.py and DELIVERY is separate. That split
(store / evaluate / deliver) is what keeps the core locally testable and lets the
same evaluation code run behind a cron line now or an EventBridge tick later.

A watch is a row, not a schedule file:
  - `interval_s`  : how often to evaluate (seconds).
  - `next_run`    : epoch; the runner picks watches WHERE next_run <= now.
  - `last_state`  : the last evaluation outcome — used to notify on CHANGE only
                    (de-dup / anti-cry-wolf), not on every positive evaluation.
Condition types (data-driven, reuse existing detectors):
  - "threshold" : a computed value/voyage total crossing a bound (LIGHT).
  - "anomaly"   : relational drift vs a known-good baseline (HEAVY).
  - "regime"    : an operating-mode / regime change vs baseline (HEAVY).

The notify target is resolved via the actors layer AT CREATION time (so an
ambiguous "Andrew" is clarified up front and the concrete actor id is stored) —
no ambiguity, and no LLM call, at fire time.

Stored as JSON at artifacts/watches.json.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path


CONDITION_TYPES = ("threshold", "anomaly", "regime")
# routing (matches the AWS ADR): threshold is a bounded aggregate (light),
# drift/regime fit or score a model (heavy). Set at creation, stored on the watch.
_RUNTIME = {"threshold": "light", "anomaly": "heavy", "regime": "heavy"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _watch_id(source: str, condition_type: str, params: dict) -> str:
    """Stable short id from the watch's defining fields, so a repeat definition
    updates rather than duplicates."""
    key = f"{source}:{condition_type}:{json.dumps(params, sort_keys=True)}"
    return hashlib.sha1(key.encode()).hexdigest()[:8]


class WatchRegistry:
    """CRUD + scheduling bookkeeping over the watches store. Pure data + JSON."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._watches: list[dict] = []
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self._watches = data.get("watches", [])

    # --- read ---
    def all(self) -> list[dict]:
        return list(self._watches)

    def get(self, watch_id: str) -> dict | None:
        return next((w for w in self._watches if w["id"] == watch_id), None)

    def for_source(self, source: str) -> list[dict]:
        return [w for w in self._watches if w.get("source") == source]

    def due(self, now: float | None = None) -> list[dict]:
        """Enabled watches whose next_run has passed — what the runner evaluates.
        A never-run watch (next_run None/0) is due immediately."""
        now = time.time() if now is None else now
        return [w for w in self._watches
                if w.get("enabled", True) and (w.get("next_run") or 0) <= now]

    # --- create / update / delete ---
    def add(self, source: str, condition_type: str, params: dict,
            notify: str | None = None, notify_actor_id: str | None = None,
            interval_s: int = 3600, description: str = "",
            author: str = "user") -> dict:
        """Add (or update, idempotent by source+type+params) a watch. The notify
        target is a resolved actor id (+ a human label) supplied by the caller,
        which resolves it via the actors layer at creation."""
        if condition_type not in CONDITION_TYPES:
            raise ValueError(f"unknown condition type '{condition_type}'")
        wid = _watch_id(source, condition_type, params)
        existing = self.get(wid)
        if existing:
            existing.update({
                "params": params, "notify": notify,
                "notify_actor_id": notify_actor_id,
                "interval_s": interval_s, "description": description,
                "enabled": True, "updated_at": _now(),
            })
            self.save()
            return existing
        watch = {
            "id": wid,
            "source": source,
            "condition_type": condition_type,
            "runtime": _RUNTIME[condition_type],   # light|heavy (routing seam)
            "params": params,                       # {signal, op, value, unit, per, ...}
            "notify": notify,                       # human label (e.g. "Andrew")
            "notify_actor_id": notify_actor_id,     # resolved at creation
            "interval_s": interval_s,
            "description": description,
            "enabled": True,
            "next_run": 0,                          # due immediately on first run
            "last_state": None,                     # for de-dup (notify on change)
            "last_fired": None,
            "author": author,
            "created_at": _now(),
            "updated_at": _now(),
        }
        self._watches.append(watch)
        self.save()
        return watch

    def delete(self, watch_id: str) -> bool:
        before = len(self._watches)
        self._watches = [w for w in self._watches if w["id"] != watch_id]
        if len(self._watches) != before:
            self.save()
            return True
        return False

    def set_enabled(self, watch_id: str, enabled: bool) -> dict | None:
        w = self.get(watch_id)
        if w:
            w["enabled"] = enabled
            w["updated_at"] = _now()
            self.save()
        return w

    # --- scheduling / de-dup bookkeeping (called by the runner) ---
    def record_evaluation(self, watch_id: str, state: str,
                          now: float | None = None) -> bool:
        """Record an evaluation outcome. Advances next_run by the interval and
        stores last_state. Returns True if this is a state CHANGE worth notifying
        (the de-dup gate): fire when the new state differs from the last one
        (e.g. ok->breached, or a new anomaly signature), stay quiet when
        unchanged. Never-notify for the 'ok'/'clear' state unless it is a change
        back from a fired state (so a resolved condition can notify once)."""
        now = time.time() if now is None else now
        w = self.get(watch_id)
        if not w:
            return False
        prev = w.get("last_state")
        w["last_state"] = state
        w["next_run"] = now + int(w.get("interval_s", 3600))
        changed = (state != prev)
        # a "fire" state that changed -> notify; a return to ok that changed ->
        # notify once (resolved); ok->ok or same-fire->same-fire -> quiet.
        notify = changed and (state not in (None, "") )
        if notify:
            w["last_fired"] = _now()
        self.save()
        return notify

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "watches": self._watches,
            "updated_at": _now(),
        }, indent=2))
