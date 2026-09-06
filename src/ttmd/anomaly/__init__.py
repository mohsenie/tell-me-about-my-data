"""Anomaly detection: baseline store + drift detector.

The product's core step. Discovery *characterizes* data (regimes + relationship
fingerprints); this module persists a KNOWN-GOOD baseline and flags when a new
window's fingerprint DRIFTS from it.

Two complementary layers, both compared against the known-good baseline:
  - Layer 1: within-regime relationship-structure drift (likely FAULT).
  - Layer 2: regime events (usage changed) — new regime, distribution shift.

Invariants honored:
  #5 baseline is a fixed known-good window, NOT rolling.
  #6 localize WHICH edges changed, ranked by magnitude.
  #7 confidence scales with data volume.
  #8 novelty != fault — a new regime is reported neutrally, not as a problem.
  Never assert the cause; report the observed change (a route change and a
  developing fault look alike from data — only the human knows which).
"""
from .baseline import build_baseline, save_baseline, load_baseline, has_baseline
from .detector import detect_drift
from .behavioral import detect_stops, dwell_norm, flag_current_dwell
from .transitions import (
    build_transition_matrix, detect_transition_anomalies, runs_from_labels)
from .timescale import (
    multiscale_drift, fingerprint_distance, group_dates, period_key)
from .joint import (
    build_joint_envelopes, detect_joint, fit_envelope, mahalanobis)

__all__ = [
    "build_baseline", "save_baseline", "load_baseline", "has_baseline",
    "detect_drift",
    "detect_stops", "dwell_norm", "flag_current_dwell",
    "build_transition_matrix", "detect_transition_anomalies", "runs_from_labels",
    "multiscale_drift", "fingerprint_distance", "group_dates", "period_key",
    "build_joint_envelopes", "detect_joint", "fit_envelope", "mahalanobis",
]
