"""Joint multivariate detector — per-regime covariance envelope (Mahalanobis).

The THIRD detector, complementary to the other two:
  - relational drift watches PAIRWISE couplings (edge dcor within a regime);
  - behavioral norm watches interpretable BEHAVIORS (dwell time, ...);
  - this watches the JOINT distribution of all signals AT ONCE. It catches a row
    whose combination of values is unusual for its regime even when every single
    signal is individually in range and no pairwise correlation moved — the gap a
    pairwise method structurally can't see.

Method (training-free, deterministic, light numpy, explainable):
  Per regime, from the KNOWN-GOOD window, learn the mean and covariance of the
  signals (in the regime model's SCALED space, so all signals are comparable).
  Score a new row by its squared MAHALANOBIS distance to that regime's envelope:
      d2 = (x - mu)^T * Sigma^-1 * (x - mu)
  Under a roughly-Gaussian normal region, d2 follows a chi-square(df) law, giving
  a principled threshold (a high quantile). A point beyond it is "jointly unusual
  for its mode". Crucially d2 DECOMPOSES additively into per-signal contributions
  (via the whitened residual), so we can say WHICH signals drove the score —
  keeping the detector explainable (no black box), same as the other two.

Discipline: reports the OBSERVED joint deviation, never the cause. Confidence
scales with how many known-good rows defined the envelope. Data-agnostic — signals
come from the regime model's columns; nothing hardcoded. Covariance is regularized
(ridge) so it stays invertible on collinear / thin data.
"""
from __future__ import annotations

import numpy as np


# ridge added to the covariance diagonal (in scaled space, unit-variance-ish) so
# Sigma stays invertible under collinear signals / few samples.
_RIDGE = 1e-3
# chi-square upper-tail quantile used as the per-point flag threshold
_CHI2_Q = 0.999
# a regime needs at least this many known-good rows to fit a usable envelope
_MIN_ROWS = 50


def _scaled(model: dict, columns: list[str], data: np.ndarray) -> np.ndarray:
    """Project rows into the regime model's scaled space (model column order,
    mean/scale from the model). Missing columns -> the model mean (neutral).
    NaNs -> that column's model mean too, so a row is always scorable."""
    mcols = model["columns"]
    mean = np.asarray(model["scaler_mean"], float)
    scale = np.asarray(model["scaler_scale"], float)
    idx = {c: i for i, c in enumerate(columns)}
    n = len(data)
    X = np.empty((n, len(mcols)), float)
    for j, c in enumerate(mcols):
        if c in idx:
            col = data[:, idx[c]].astype(float)
            col = np.where(np.isnan(col), mean[j], col)
            X[:, j] = col
        else:
            X[:, j] = mean[j]
    return (X - mean) / scale


def fit_envelope(scaled_rows: np.ndarray) -> dict | None:
    """Fit a covariance envelope to known-good rows already in scaled space.
    Returns {mean, cov_inv, df, n, threshold} or None if too few rows.

    The threshold is the MAX of two cutoffs so it's both principled and honest:
      - chi-square(_CHI2_Q, df): the parametric tail under a Gaussian normal region;
      - the EMPIRICAL high quantile of the training distances (p99.9): calibrated
        on the known-good window itself, so a window compared to its own baseline
        flags ~0 (self-consistency) even when the data isn't perfectly Gaussian.
    Taking the max means we never flag more of the training data than the empirical
    tail allows, while still respecting the parametric floor for thin regimes."""
    if scaled_rows is None or len(scaled_rows) < _MIN_ROWS:
        return None
    mu = scaled_rows.mean(axis=0)
    d = scaled_rows.shape[1]
    cov = np.cov(scaled_rows, rowvar=False)
    cov = np.atleast_2d(cov) + _RIDGE * np.eye(d)
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)
    from scipy.stats import chi2
    chi_thr = float(chi2.ppf(_CHI2_Q, df=d))
    # empirical calibration on the training rows themselves
    diff = scaled_rows - mu
    train_d2 = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)
    emp_thr = float(np.quantile(train_d2, 0.999))
    return {
        "mean": mu.tolist(),
        "cov_inv": cov_inv.tolist(),
        "df": int(d),
        "n": int(len(scaled_rows)),
        "threshold": max(chi_thr, emp_thr),
        "chi2_threshold": chi_thr,
        "empirical_threshold": emp_thr,
    }


def mahalanobis(scaled_rows: np.ndarray, envelope: dict) -> np.ndarray:
    """Squared Mahalanobis distance of each scaled row to the envelope."""
    mu = np.asarray(envelope["mean"], float)
    cov_inv = np.asarray(envelope["cov_inv"], float)
    diff = scaled_rows - mu
    # row-wise quadratic form: sum((diff @ cov_inv) * diff, axis=1)
    return np.einsum("ij,jk,ik->i", diff, cov_inv, diff)


def per_signal_contribution(scaled_row: np.ndarray, envelope: dict,
                            columns: list[str]) -> list[tuple[str, float]]:
    """Approximate additive share of each signal in a row's Mahalanobis distance,
    using the diagonal-dominant term diff_i * (cov_inv @ diff)_i (exact when the
    off-diagonal cancels; a good ranking otherwise). Returns [(signal, share)]
    sorted desc, shares summing to ~1."""
    mu = np.asarray(envelope["mean"], float)
    cov_inv = np.asarray(envelope["cov_inv"], float)
    diff = scaled_row - mu
    contrib = diff * (cov_inv @ diff)          # per-coordinate contribution
    total = contrib.sum()
    if total <= 0:
        return []
    shares = [(columns[i], float(contrib[i] / total)) for i in range(len(columns))]
    shares.sort(key=lambda t: t[1], reverse=True)
    return shares


def _confidence(n: int) -> str:
    if n >= 5000:
        return "high"
    if n >= 1000:
        return "medium"
    return "low"


# a signal that moves in the SAME direction this fraction of consecutive steps is
# treated as a monotonic counter / cumulative (hours-run, odometer, draining tank)
# and EXCLUDED from the joint envelope: its mean drifts with time by construction,
# so it would flag every later window as "anomalous" for a non-anomalous reason.
_MONOTONIC_FRAC = 0.98


def _monotonic_mask(data: np.ndarray, columns: list[str]) -> list[bool]:
    """Per column, True if it's (near-)monotonic over the window — a counter or
    cumulative level rather than a fluctuating sensor. Data-driven: measures the
    fraction of consecutive steps that share the dominant sign; nothing hardcoded."""
    keep = []
    for i in range(data.shape[1]):
        col = data[:, i].astype(float)
        col = col[~np.isnan(col)]
        if len(col) < 3:
            keep.append(True)   # can't tell -> keep
            continue
        diffs = np.diff(col)
        nz = diffs[diffs != 0]
        if len(nz) == 0:
            keep.append(True)   # constant handled elsewhere
            continue
        dom = max((nz > 0).mean(), (nz < 0).mean())
        keep.append(dom < _MONOTONIC_FRAC)   # keep only NON-monotonic signals
    return keep


def build_joint_envelopes(model: dict, columns: list[str], data: np.ndarray,
                          labels: np.ndarray) -> dict:
    """Per regime, fit a covariance envelope from the known-good rows assigned to
    it. Returns {mcolumns, columns_used, regimes: {label: envelope}} keyed like
    the model. `labels` must align with `data` rows. Monotonic counters /
    cumulatives are excluded (their mean drifts with time by construction)."""
    scaled = _scaled(model, columns, data)
    mcols = list(model["columns"])
    # drop monotonic columns (measured in RAW model-space via _scaled's ordering)
    raw_ordered = _reorder_raw(model, columns, data)
    keep = _monotonic_mask(raw_ordered, mcols)
    used_idx = [j for j, k in enumerate(keep) if k]
    cols_used = [mcols[j] for j in used_idx]
    if not cols_used:
        used_idx = list(range(len(mcols)))
        cols_used = mcols
    envelopes = {}
    for lab in sorted(set(int(x) for x in labels)):
        rows = scaled[labels == lab][:, used_idx]
        env = fit_envelope(rows)
        if env is not None:
            envelopes[str(lab)] = env
    return {"mcolumns": cols_used, "regimes": envelopes}


def _reorder_raw(model: dict, columns: list[str], data: np.ndarray) -> np.ndarray:
    """Rows in the model's column order, RAW units (for monotonicity test)."""
    mcols = model["columns"]
    mean = np.asarray(model["scaler_mean"], float)
    idx = {c: i for i, c in enumerate(columns)}
    n = len(data)
    X = np.empty((n, len(mcols)), float)
    for j, c in enumerate(mcols):
        X[:, j] = data[:, idx[c]] if c in idx else mean[j]
    return X


def detect_joint(model: dict, columns: list[str], data: np.ndarray,
                 labels: np.ndarray, joint_baseline: dict) -> dict:
    """Score a new window against the per-regime covariance envelopes. For each
    regime present, report the fraction of points beyond the chi-square threshold
    and the signals contributing most to the flagged points.

    Returns {findings:[{regime, n_points, n_flagged, flagged_fraction, confidence,
    top_signals:[{signal, share}]}], overall_flagged_fraction}. Observed joint
    deviation, never cause."""
    full = _scaled(model, columns, data)
    mcols = joint_baseline.get("mcolumns", list(model["columns"]))
    # subselect the columns the envelope was fit on (monotonic ones were dropped)
    model_cols = list(model["columns"])
    sel = [model_cols.index(c) for c in mcols if c in model_cols]
    scaled = full[:, sel]
    envs = joint_baseline.get("regimes", {})
    findings = []
    total_pts = total_flagged = 0
    for lab in sorted(set(int(x) for x in labels)):
        env = envs.get(str(lab))
        rows = scaled[labels == lab]
        if env is None or len(rows) == 0:
            continue
        d2 = mahalanobis(rows, env)
        thr = env["threshold"]
        flagged = d2 > thr
        n_pts = int(len(rows)); n_flag = int(flagged.sum())
        total_pts += n_pts; total_flagged += n_flag
        # explain: average per-signal contribution over the FLAGGED points
        top = []
        if n_flag:
            contrib_acc = np.zeros(len(mcols))
            for r in rows[flagged]:
                mu = np.asarray(env["mean"], float)
                ci = np.asarray(env["cov_inv"], float)
                diff = r - mu
                c = diff * (ci @ diff)
                s = c.sum()
                if s > 0:
                    contrib_acc += c / s
            contrib_acc /= n_flag
            order = np.argsort(contrib_acc)[::-1][:4]
            top = [{"signal": mcols[i], "share": round(float(contrib_acc[i]), 3)}
                   for i in order if contrib_acc[i] > 0]
        findings.append({
            "regime": str(lab),
            "n_points": n_pts,
            "n_flagged": n_flag,
            "flagged_fraction": round(n_flag / n_pts, 4) if n_pts else 0.0,
            "confidence": _confidence(env.get("n", 0)),
            "top_signals": top,
        })
    findings.sort(key=lambda f: f["flagged_fraction"], reverse=True)
    return {
        "findings": findings,
        "overall_flagged_fraction": (round(total_flagged / total_pts, 4)
                                     if total_pts else 0.0),
        "n_points": total_pts,
    }
