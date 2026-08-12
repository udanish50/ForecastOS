from __future__ import annotations

import numpy as np
import pandas as pd

from .types import DataProfile


def _frequency_label(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds <= 0:
        return "Unknown"
    minute = 60
    hour = 3600
    day = 86400
    if abs(seconds - minute) < 2:
        return "1 min"
    if seconds < hour:
        return f"{round(seconds / minute):g} min"
    if abs(seconds - hour) < 5:
        return "Hourly"
    if seconds < day:
        return f"{seconds / hour:g} hours"
    if abs(seconds - day) < 60:
        return "Daily"
    if abs(seconds - 7 * day) < 300:
        return "Weekly"
    return f"{seconds:g} sec"


def candidate_periods_from_step(seconds: float) -> list[int]:
    if not np.isfinite(seconds) or seconds <= 0:
        return [1]
    hour = 3600
    day = 86400
    # Daily + weekly candidates where computationally reasonable.
    vals = [1]
    daily = int(round(day / seconds))
    weekly = int(round(7 * day / seconds))
    for p in (daily, weekly):
        if 2 <= p <= 10080:
            vals.append(p)
    if seconds >= day * 0.8 and seconds <= day * 1.2:
        vals.extend([7, 30])
    return sorted(set(vals))


def _autocorr(series: pd.Series, lag: int) -> float:
    if lag <= 0 or len(series) <= lag + 3:
        return float("nan")
    return float(series.autocorr(lag=lag))


def profile_timeseries(df: pd.DataFrame, timestamp_col: str, target_col: str) -> DataProfile:
    ts = pd.to_datetime(df[timestamp_col], errors="coerce")
    y_raw = pd.to_numeric(df[target_col], errors="coerce")
    duplicate_timestamps = int(ts.duplicated().sum())
    missing_target_pct = float(100 * y_raw.isna().mean())
    valid = pd.DataFrame({"ts": ts, "y": y_raw}).dropna(subset=["ts"]).sort_values("ts")
    deltas = valid["ts"].diff().dt.total_seconds().dropna()
    median_step = float(deltas.median()) if len(deltas) else float("nan")
    if len(deltas) and median_step > 0:
        tolerance = max(1.0, median_step * 0.02)
        regularity = float(np.mean(np.abs(deltas - median_step) <= tolerance))
    else:
        regularity = 0.0
    candidates = candidate_periods_from_step(median_step)
    y = valid["y"].interpolate(limit_direction="both")
    seasonal_scores = [(abs(_autocorr(y, p)), p) for p in candidates if p > 1 and len(y) > 2 * p]
    seasonal_scores = [(s if np.isfinite(s) else -1.0, p) for s, p in seasonal_scores]
    selected = max(seasonal_scores)[1] if seasonal_scores else 1
    lag1 = _autocorr(y, 1)
    n = len(y)
    if n >= 10 and float(np.nanstd(y)) > 1e-12:
        x = np.arange(n, dtype=float)
        slope = float(np.polyfit(x, y.to_numpy(dtype=float), 1)[0])
        trend_strength = float(min(1.0, abs(slope) * n / (4 * np.nanstd(y) + 1e-12)))
        q = max(3, n // 5)
        early = float(np.nanmean(y.iloc[:q]))
        late = float(np.nanmean(y.iloc[-q:]))
        drift_score = float(min(1.0, abs(late - early) / (2 * np.nanstd(y) + 1e-12)))
    else:
        trend_strength = 0.0
        drift_score = 0.0
    warnings = []
    if regularity < 0.95:
        warnings.append("Timestamps are not perfectly regular; forecasts use the median observed interval.")
    if missing_target_pct > 5:
        warnings.append(f"Target contains {missing_target_pct:.1f}% missing values; interpolation is applied.")
    if duplicate_timestamps:
        warnings.append(f"Found {duplicate_timestamps} duplicate timestamps; duplicates are aggregated.")
    numeric_features = [c for c in df.select_dtypes(include=np.number).columns if c != target_col]
    categorical_features = [c for c in df.columns if c not in numeric_features + [timestamp_col, target_col]]
    return DataProfile(
        rows=len(df),
        start=valid["ts"].min(),
        end=valid["ts"].max(),
        inferred_frequency=_frequency_label(median_step),
        median_step_seconds=median_step,
        regularity_score=regularity,
        missing_target_pct=missing_target_pct,
        duplicate_timestamps=duplicate_timestamps,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        candidate_seasonal_periods=candidates,
        selected_seasonal_period=selected,
        trend_strength=trend_strength,
        autocorr_lag1=lag1 if np.isfinite(lag1) else 0.0,
        drift_score=drift_score,
        warnings=warnings,
    )
