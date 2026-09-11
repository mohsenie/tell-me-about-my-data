"""Optional autoencoder backend for the joint detector — nonlinear normal manifold.

The Mahalanobis envelope (anomaly/joint.py) models each regime's normal region as
an ELLIPSOID (a mean + covariance). That captures linear structure and the joint
distribution's second moments, which is enough for most faults. But if a regime's
normal operating region is a strongly CURVED / nonlinear manifold (a bent surface,
not an ellipsoid), an ellipsoid either over-covers it (misses anomalies inside the
bounding ellipsoid) or under-covers it (false-flags normal points on the far side).

This backend fits, per regime, a small AUTOENCODER (an MLP trained to reconstruct
its own input) on the known-good rows. A point that lies ON the learned normal
manifold reconstructs with low error; a point OFF it reconstructs poorly, so the
reconstruction error is the anomaly score. Because the error is PER-FEATURE, we
keep the same explainability as Mahalanobis: we can say which signals reconstructed
worst, i.e. which drove the score.

DELIBERATELY OPTIONAL and SECONDARY (behind a flag, off by default):
  - it adds a TRAINING step (stochastic; we fix random_state for determinism) and
    only runs where a regime has enough data — it regresses the system's
    training-free / deterministic character, so it's a backend you opt into for
    regimes the covariance model can't capture, not the default.
  - it uses sklearn's MLPRegressor (already a dependency) — no new heavy dep, no
    torch. If sklearn were unavailable, ae_available() is False and everything
    else keeps working unchanged.

Discipline unchanged: reports the OBSERVED reconstruction deviation, never the
cause; confidence scales with training rows; signals come from the regime model's
columns (data-agnostic, nothing hardcoded).
"""
from __future__ import annotations

import numpy as np


def ae_available() -> bool:
    """Whether the optional AE backend can run (sklearn present)."""
    try:
        from sklearn.neural_network import MLPRegressor  # noqa: F401
        return True
    except Exception:
        return False


# a regime needs at least this many known-good rows before we train an AE
_MIN_AE_ROWS = 200
# bottleneck as a fraction of input dims (compress -> forces a manifold, not identity)
_BOTTLENECK_FRAC = 0.5
_RANDOM_STATE = 0


def train_ae(scaled_rows: np.ndarray) -> dict | None:
    """Train a small autoencoder (MLP X->X) on known-good scaled rows. Returns a
    serializable-ish backend dict {model, threshold, n, df} or None if the backend
    is unavailable or there are too few rows. threshold = empirical p99.9 of the
    training reconstruction errors (honest, calibrated on the known-good window)."""
    if not ae_available() or scaled_rows is None or len(scaled_rows) < _MIN_AE_ROWS:
        return None
    from sklearn.neural_network import MLPRegressor
    d = scaled_rows.shape[1]
    bottleneck = max(1, int(round(d * _BOTTLENECK_FRAC)))
    # encoder->bottleneck->decoder, symmetric small MLP
    hidden = (max(bottleneck + 1, d), bottleneck, max(bottleneck + 1, d))
    ae = MLPRegressor(hidden_layer_sizes=hidden, activation="tanh",
                      solver="adam", max_iter=300, random_state=_RANDOM_STATE,
                      early_stopping=False)
    ae.fit(scaled_rows, scaled_rows)
    recon = ae.predict(scaled_rows)
    err = ((scaled_rows - recon) ** 2).mean(axis=1)
    threshold = float(np.quantile(err, 0.999))
    return {"model": ae, "threshold": threshold, "n": int(len(scaled_rows)),
            "df": int(d)}


def ae_errors(scaled_rows: np.ndarray, backend: dict) -> np.ndarray:
    """Per-row mean reconstruction error (the anomaly score)."""
    ae = backend["model"]
    recon = ae.predict(scaled_rows)
    return ((scaled_rows - recon) ** 2).mean(axis=1)


def ae_feature_errors(scaled_rows: np.ndarray, backend: dict) -> np.ndarray:
    """Per-row, per-feature squared reconstruction error (for attribution)."""
    ae = backend["model"]
    recon = ae.predict(scaled_rows)
    return (scaled_rows - recon) ** 2


def _confidence(n: int) -> str:
    if n >= 5000:
        return "high"
    if n >= 1000:
        return "medium"
    return "low"


def build_ae_backends(model: dict, columns: list[str], data: np.ndarray,
                      labels: np.ndarray, used_columns: list[str]) -> dict | None:
    """Per regime, train an AE on the known-good rows (over `used_columns` — the
    same non-monotonic columns the Mahalanobis envelope uses, so the two backends
    are comparable). Returns {mcolumns, regimes:{lab:backend}} or None if the
    backend is unavailable. Backends are kept IN MEMORY (the MLP isn't JSON-
    serializable), so this is used within a single detect run, not persisted."""
    if not ae_available():
        return None
    from .joint import _scaled
    full = _scaled(model, columns, data)
    model_cols = list(model["columns"])
    sel = [model_cols.index(c) for c in used_columns if c in model_cols]
    scaled = full[:, sel]
    regimes = {}
    for lab in sorted(set(int(x) for x in labels)):
        rows = scaled[labels == lab]
        b = train_ae(rows)
        if b is not None:
            regimes[str(lab)] = b
    if not regimes:
        return None
    return {"mcolumns": list(used_columns), "regimes": regimes}


def detect_ae(model: dict, columns: list[str], data: np.ndarray,
              labels: np.ndarray, ae_backends: dict) -> dict:
    """Score a new window against the per-regime AEs. Reports, per regime, the
    fraction of points whose reconstruction error exceeds the regime's threshold
    and the signals that reconstructed worst (attribution). Same shape/discipline
    as detect_joint. Observed deviation, never cause."""
    from .joint import _scaled
    full = _scaled(model, columns, data)
    mcols = ae_backends.get("mcolumns", list(model["columns"]))
    model_cols = list(model["columns"])
    sel = [model_cols.index(c) for c in mcols if c in model_cols]
    scaled = full[:, sel]
    regimes = ae_backends.get("regimes", {})
    findings = []
    total_pts = total_flagged = 0
    for lab in sorted(set(int(x) for x in labels)):
        b = regimes.get(str(lab))
        rows = scaled[labels == lab]
        if b is None or len(rows) == 0:
            continue
        err = ae_errors(rows, b)
        flagged = err > b["threshold"]
        n_pts = int(len(rows)); n_flag = int(flagged.sum())
        total_pts += n_pts; total_flagged += n_flag
        top = []
        if n_flag:
            fe = ae_feature_errors(rows[flagged], b).mean(axis=0)
            s = fe.sum()
            if s > 0:
                order = np.argsort(fe)[::-1][:4]
                top = [{"signal": mcols[i], "share": round(float(fe[i] / s), 3)}
                       for i in order if fe[i] > 0]
        findings.append({
            "regime": str(lab), "n_points": n_pts, "n_flagged": n_flag,
            "flagged_fraction": round(n_flag / n_pts, 4) if n_pts else 0.0,
            "confidence": _confidence(b.get("n", 0)),
            "top_signals": top,
        })
    findings.sort(key=lambda f: f["flagged_fraction"], reverse=True)
    return {
        "backend": "autoencoder",
        "findings": findings,
        "overall_flagged_fraction": (round(total_flagged / total_pts, 4)
                                     if total_pts else 0.0),
        "n_points": total_pts,
    }
