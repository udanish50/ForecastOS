from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class DataProfile:
    rows: int
    start: pd.Timestamp
    end: pd.Timestamp
    inferred_frequency: str
    median_step_seconds: float
    regularity_score: float
    missing_target_pct: float
    duplicate_timestamps: int
    numeric_features: list[str]
    categorical_features: list[str]
    candidate_seasonal_periods: list[int]
    selected_seasonal_period: int
    trend_strength: float
    autocorr_lag1: float
    drift_score: float
    warnings: list[str] = field(default_factory=list)


@dataclass
class BacktestResult:
    model_name: str
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    aggregate_metrics: dict[str, float]
    fit_seconds: float
    failure: str | None = None


@dataclass
class ForecastResult:
    model_name: str
    forecast: pd.DataFrame
    metrics: dict[str, float]
    trust_score: float
    trust_components: dict[str, float]
    leaderboard: pd.DataFrame
    feature_importance: pd.DataFrame
    diagnostics: dict[str, Any]
    fitted_model: Any | None = None
