from __future__ import annotations

import numpy as np
import pandas as pd


def choose_lags(n: int, seasonal_period: int) -> list[int]:
    base = [1, 2, 3, 6, 12, 24]
    if seasonal_period > 1:
        base += [seasonal_period, max(1, seasonal_period - 1), seasonal_period + 1]
        if 2 * seasonal_period < n // 2:
            base.append(2 * seasonal_period)
    return sorted({x for x in base if 1 <= x < max(2, n // 3)})


def time_features(ts: pd.Timestamp) -> dict[str, float]:
    hour = ts.hour + ts.minute / 60.0
    dow = ts.dayofweek
    doy = ts.dayofyear
    return {
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "is_weekend": float(dow >= 5),
    }


def make_row(history: list[float], ts: pd.Timestamp, lags: list[int], exog: dict[str, float] | None = None) -> dict[str, float]:
    row = {}
    for lag in lags:
        row[f"lag_{lag}"] = history[-lag] if len(history) >= lag else np.nan
    row.update(time_features(ts))
    if exog:
        for k, v in exog.items():
            try:
                row[f"exog__{k}"] = float(v)
            except (TypeError, ValueError):
                continue
    return row


def supervised_frame(
    df: pd.DataFrame,
    timestamp_col: str,
    target_col: str,
    lags: list[int],
    exog_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    exog_cols = exog_cols or []
    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)
    rows, targets = [], []
    start = max(lags) if lags else 1
    for i in range(start, len(df)):
        hist = y[:i].tolist()
        exog = {c: df.iloc[i][c] for c in exog_cols if c in df.columns}
        row = make_row(hist, pd.Timestamp(df.iloc[i][timestamp_col]), lags, exog)
        rows.append(row)
        targets.append(y[i])
    X = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    return X, pd.Series(targets, name=target_col)
