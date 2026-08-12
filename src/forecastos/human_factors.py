from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Readiness:
    level: str
    label: str
    summary: str
    issues: list[str]


def readiness(profile, horizon: int, n_rows: int) -> Readiness:
    issues: list[str] = []
    severity = 0
    if profile.regularity_score < 0.95:
        issues.append("Timestamps are irregular; forecasts will use the median observed interval.")
        severity += 1
    if profile.missing_target_pct > 5:
        issues.append(f"Target has {profile.missing_target_pct:.1f}% missing values; interpolation can add uncertainty.")
        severity += 1
    if profile.drift_score >= 0.6:
        issues.append("Recent values differ substantially from earlier history; drift risk is high.")
        severity += 2
    elif profile.drift_score >= 0.25:
        issues.append("Some distribution drift is present; use backtest stability and intervals when deciding.")
        severity += 1
    if n_rows < max(80, horizon * 5):
        issues.append("History is short relative to the requested horizon; complex models may overfit.")
        severity += 2
    if severity >= 3:
        return Readiness("caution", "Proceed with caution", "The data can be forecast, but important reliability risks need attention.", issues)
    if severity >= 1:
        return Readiness("review", "Review recommended", "The data is usable, with a few conditions worth reviewing before acting on results.", issues)
    return Readiness("ready", "Ready to forecast", "The dataset passes the main readiness checks for this prototype.", issues)


def trust_label(score: float) -> tuple[str, str]:
    if score >= 85:
        return "High", "Backtests and data checks are comparatively strong. Still use the interval and domain judgment."
    if score >= 70:
        return "Moderate", "The forecast is useful for planning, but one or more reliability components deserve attention."
    if score >= 50:
        return "Limited", "Treat the forecast as supporting evidence rather than a primary decision signal."
    return "Low", "Reliability checks are weak. Investigate data quality, drift, horizon, or model stability before using the forecast."


def compute_estimate(rows: int, mode: str, deep_models: list[str] | None) -> tuple[str, str]:
    deep_models = deep_models or []
    load = rows * ({"Fast": 1.0, "Balanced": 1.8, "Maximum accuracy": 3.0}.get(mode, 1.8))
    if deep_models:
        load *= 1 + 1.1 * len(deep_models)
    if load < 120_000:
        return "Light", "Appropriate for an interactive CPU session."
    if load < 450_000:
        return "Moderate", "May take noticeably longer on shared CPU resources."
    return "Heavy", "Use selectively on Streamlit Cloud; a dedicated worker/GPU is better for repeated experiments."


def model_plain_name(name: str) -> str:
    mapping = {
        "Naive": "Last value",
        "Seasonal Naive": "Repeat last seasonal pattern",
        "Drift": "Trend continuation",
        "Ridge AR": "Regularized autoregression",
        "HistGradientBoosting AR": "Gradient-boosted autoregression",
        "Deep MLP AR": "Deep neural autoregression",
        "LSTM": "LSTM sequence model",
        "TCN": "Temporal convolutional network",
        "Transformer": "Transformer sequence encoder",
    }
    return mapping.get(name, name)


def safe_metric(value: float, decimals: int = 2) -> str:
    return "—" if not np.isfinite(value) else f"{value:.{decimals}f}"
