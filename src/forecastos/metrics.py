from __future__ import annotations

import numpy as np


def _clean(y_true, y_pred):
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def mae(y_true, y_pred) -> float:
    a, b = _clean(y_true, y_pred)
    return float(np.mean(np.abs(a - b))) if len(a) else float("nan")


def median_ae(y_true, y_pred) -> float:
    a, b = _clean(y_true, y_pred)
    return float(np.median(np.abs(a - b))) if len(a) else float("nan")


def rmse(y_true, y_pred) -> float:
    a, b = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2))) if len(a) else float("nan")


def nrmse_std(y_true, y_pred, eps: float = 1e-8) -> float:
    a, b = _clean(y_true, y_pred)
    if not len(a):
        return float("nan")
    return float(rmse(a, b) / max(float(np.std(a)), eps))


def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    a, b = _clean(y_true, y_pred)
    if not len(a):
        return float("nan")
    mask = np.abs(a) > eps
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((a[mask] - b[mask]) / a[mask])) * 100.0)


def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    a, b = _clean(y_true, y_pred)
    if not len(a):
        return float("nan")
    denom = np.abs(a) + np.abs(b)
    return float(np.mean(200.0 * np.abs(a - b) / np.maximum(denom, eps)))


def wape(y_true, y_pred, eps: float = 1e-8) -> float:
    a, b = _clean(y_true, y_pred)
    if not len(a):
        return float("nan")
    return float(100.0 * np.sum(np.abs(a - b)) / max(float(np.sum(np.abs(a))), eps))


def mean_error(y_true, y_pred) -> float:
    """Signed bias: positive means forecasts are too low on average (actual - prediction)."""
    a, b = _clean(y_true, y_pred)
    return float(np.mean(a - b)) if len(a) else float("nan")


def r2(y_true, y_pred) -> float:
    a, b = _clean(y_true, y_pred)
    if len(a) < 2:
        return float("nan")
    denom = float(np.sum((a - np.mean(a)) ** 2))
    if denom <= 1e-12:
        return float("nan")
    return float(1.0 - np.sum((a - b) ** 2) / denom)


def directional_accuracy(y_true, y_pred, previous_actual: float | None = None) -> float:
    a, b = _clean(y_true, y_pred)
    if len(a) < 1:
        return float("nan")
    if previous_actual is None or not np.isfinite(previous_actual):
        if len(a) < 2:
            return float("nan")
        da = np.sign(np.diff(a))
        dp = np.sign(np.diff(b))
    else:
        da = np.sign(np.diff(np.r_[float(previous_actual), a]))
        dp = np.sign(np.diff(np.r_[float(previous_actual), b]))
    return float(100.0 * np.mean(da == dp)) if len(da) else float("nan")


def mase(y_true, y_pred, insample, seasonal_period: int = 1, eps: float = 1e-8) -> float:
    a, b = _clean(y_true, y_pred)
    hist = np.asarray(insample, dtype=float)
    hist = hist[np.isfinite(hist)]
    p = max(1, int(seasonal_period))
    if len(hist) <= p or not len(a):
        return float("nan")
    scale = np.mean(np.abs(hist[p:] - hist[:-p]))
    return float(np.mean(np.abs(a - b)) / max(scale, eps))


def skill_score(model_rmse: float, baseline_rmse: float) -> float:
    if not np.isfinite(model_rmse) or not np.isfinite(baseline_rmse) or baseline_rmse <= 1e-12:
        return float("nan")
    return float(100.0 * (1.0 - model_rmse / baseline_rmse))
