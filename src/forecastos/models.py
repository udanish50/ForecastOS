from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import choose_lags, make_row, supervised_frame
from .scaling import make_scaler


class BaseForecaster:
    name = "Base"
    supports_exog = False

    def fit(self, df: pd.DataFrame, timestamp_col: str, target_col: str, exog_cols: list[str], seasonal_period: int):
        raise NotImplementedError

    def predict(self, horizon: int, future_timestamps: list[pd.Timestamp], future_exog: pd.DataFrame | None = None) -> np.ndarray:
        raise NotImplementedError


class NaiveForecaster(BaseForecaster):
    name = "Naive"

    def fit(self, df, timestamp_col, target_col, exog_cols, seasonal_period):
        self.history = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)
        return self

    def predict(self, horizon, future_timestamps, future_exog=None):
        return np.repeat(self.history[-1], horizon).astype(float)


class DriftForecaster(BaseForecaster):
    name = "Drift"

    def fit(self, df, timestamp_col, target_col, exog_cols, seasonal_period):
        self.history = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)
        return self

    def predict(self, horizon, future_timestamps, future_exog=None):
        y = self.history
        slope = (y[-1] - y[0]) / max(1, len(y) - 1)
        return y[-1] + slope * np.arange(1, horizon + 1)


class SeasonalNaiveForecaster(BaseForecaster):
    name = "Seasonal Naive"

    def fit(self, df, timestamp_col, target_col, exog_cols, seasonal_period):
        self.history = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)
        self.period = max(1, int(seasonal_period))
        return self

    def predict(self, horizon, future_timestamps, future_exog=None):
        if self.period <= 1 or len(self.history) < self.period:
            return np.repeat(self.history[-1], horizon).astype(float)
        tail = self.history[-self.period :]
        return np.resize(tail, horizon).astype(float)


class DeepMLPForecaster(BaseForecaster):
    supports_exog = True
    name = "Deep MLP AR"

    def __init__(self, training_level: str = "balanced", scaler_kind: str = "standard", history_window: int | None = None):
        self.training_level = training_level.lower()
        self.scaler_kind = scaler_kind
        self.history_window = history_window

    def fit(self, df, timestamp_col, target_col, exog_cols, seasonal_period):
        self.timestamp_col = timestamp_col
        self.target_col = target_col
        self.exog_cols = list(exog_cols or [])
        self.history = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float).tolist()
        self.lags = choose_lags(len(df), seasonal_period, self.history_window)
        X, y = supervised_frame(df, timestamp_col, target_col, self.lags, self.exog_cols)
        if len(X) < 40:
            raise ValueError("Not enough rows after lag construction for the deep MLP model.")
        self.feature_names_ = list(X.columns)
        cap = {"fast": 8000, "balanced": 16000, "maximum accuracy": 30000}.get(self.training_level, 16000)
        if len(X) > cap:
            X, y = X.iloc[-cap:], y.iloc[-cap:]
        feature_model = Pipeline([
            ("scale", make_scaler(self.scaler_kind)),
            ("model", MLPRegressor(
                hidden_layer_sizes=(96, 48, 24),
                activation="relu",
                solver="adam",
                alpha=5e-4,
                learning_rate_init=8e-4,
                max_iter={"fast": 110, "balanced": 180, "maximum accuracy": 260}.get(self.training_level, 180),
                early_stopping=True,
                validation_fraction=0.12,
                n_iter_no_change=14,
                random_state=42,
            )),
        ])
        self.model = TransformedTargetRegressor(regressor=feature_model, transformer=StandardScaler())
        self.model.fit(X, y)
        y_arr = np.asarray(y, dtype=float)
        y_std = float(np.nanstd(y_arr) + 1e-8)
        self.prediction_bounds_ = (float(np.nanmin(y_arr) - 2.0 * y_std), float(np.nanmax(y_arr) + 2.0 * y_std))
        self.training_X_ = X
        self.training_y_ = y
        return self

    def predict(self, horizon, future_timestamps, future_exog=None):
        history = list(self.history)
        preds = []
        for step in range(horizon):
            exog = None
            if future_exog is not None and len(future_exog) > step:
                exog = {c: future_exog.iloc[step][c] for c in self.exog_cols if c in future_exog.columns}
            row = make_row(history, pd.Timestamp(future_timestamps[step]), self.lags, exog)
            X = pd.DataFrame([row]).reindex(columns=self.feature_names_, fill_value=0.0)
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            pred = float(self.model.predict(X)[0])
            if hasattr(self, "prediction_bounds_"):
                pred = float(np.clip(pred, *self.prediction_bounds_))
            history.append(pred)
            preds.append(pred)
        return np.asarray(preds, dtype=float)


class RegressionForecaster(BaseForecaster):
    supports_exog = True

    def __init__(self, kind: str = "ridge", scaler_kind: str = "standard", history_window: int | None = None):
        self.kind = kind
        self.scaler_kind = scaler_kind
        self.history_window = history_window
        self.name = "Ridge AR" if kind == "ridge" else "HistGradientBoosting AR"

    def fit(self, df, timestamp_col, target_col, exog_cols, seasonal_period):
        self.timestamp_col = timestamp_col
        self.target_col = target_col
        self.exog_cols = list(exog_cols or [])
        self.history = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float).tolist()
        self.lags = choose_lags(len(df), seasonal_period, self.history_window)
        X, y = supervised_frame(df, timestamp_col, target_col, self.lags, self.exog_cols)
        if len(X) < 12:
            raise ValueError("Not enough rows after lag construction for regression model.")
        self.feature_names_ = list(X.columns)
        if self.kind == "ridge":
            self.model = Pipeline([
                ("scale", make_scaler(self.scaler_kind)),
                ("model", Ridge(alpha=1.0)),
            ])
        else:
            self.model = HistGradientBoostingRegressor(
                max_iter=180,
                learning_rate=0.07,
                max_leaf_nodes=24,
                l2_regularization=0.5,
                random_state=42,
            )
        # Bound training memory for public demos while retaining recent dynamics.
        if len(X) > 30000:
            X, y = X.iloc[-30000:], y.iloc[-30000:]
        self.model.fit(X, y)
        self.training_X_ = X
        self.training_y_ = y
        return self

    def predict(self, horizon, future_timestamps, future_exog=None):
        history = list(self.history)
        preds = []
        for step in range(horizon):
            exog = None
            if future_exog is not None and len(future_exog) > step:
                exog = {c: future_exog.iloc[step][c] for c in self.exog_cols if c in future_exog.columns}
            row = make_row(history, pd.Timestamp(future_timestamps[step]), self.lags, exog)
            X = pd.DataFrame([row]).reindex(columns=self.feature_names_, fill_value=0.0)
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            pred = float(self.model.predict(X)[0])
            history.append(pred)
            preds.append(pred)
        return np.asarray(preds, dtype=float)


def build_model_zoo(mode: str = "Balanced", include_mlp: bool = False, deep_model_names: list[str] | None = None, scaler_kind: str = "standard", history_window: int | None = None) -> list[BaseForecaster]:
    mode_lower = mode.lower()
    models: list[BaseForecaster] = [
        NaiveForecaster(),
        SeasonalNaiveForecaster(),
        DriftForecaster(),
        RegressionForecaster("ridge", scaler_kind=scaler_kind, history_window=history_window),
    ]
    if mode_lower in {"balanced", "maximum accuracy"}:
        models.append(RegressionForecaster("hgb", scaler_kind=scaler_kind, history_window=history_window))
    if include_mlp:
        models.append(DeepMLPForecaster(mode_lower, scaler_kind=scaler_kind, history_window=history_window))
    if deep_model_names:
        from .deep_models import build_torch_models
        models.extend(build_torch_models(deep_model_names, mode_lower, scaler_kind=scaler_kind, history_window=history_window))
    return models
