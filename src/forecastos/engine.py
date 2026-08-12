from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .metrics import (
    directional_accuracy, mae, mape, mase, mean_error, median_ae, nrmse_std, r2, rmse, skill_score, smape, wape,
)
from .models import BaseForecaster, SeasonalNaiveForecaster, build_model_zoo
from .types import BacktestResult, ForecastResult


def future_timestamps(last_ts: pd.Timestamp, step_seconds: float, horizon: int) -> list[pd.Timestamp]:
    step = pd.to_timedelta(step_seconds if np.isfinite(step_seconds) and step_seconds > 0 else 86400, unit="s")
    return [last_ts + step * (i + 1) for i in range(horizon)]


def _make_future_exog(train: pd.DataFrame, exog_cols: list[str], horizon: int, override: pd.DataFrame | None = None) -> pd.DataFrame | None:
    if not exog_cols:
        return None
    rows = []
    for _ in range(horizon):
        row = {}
        for c in exog_cols:
            if override is not None and c in override.columns and len(override) > len(rows):
                row[c] = override.iloc[len(rows)][c]
            else:
                row[c] = train[c].iloc[-1]
        rows.append(row)
    return pd.DataFrame(rows)


def _fold_origins(n: int, horizon: int, n_folds: int) -> list[int]:
    min_train = max(20, horizon * 2)
    last_origin = n - horizon
    if last_origin <= min_train:
        return [max(10, last_origin)] if last_origin > 10 else []
    candidates = np.linspace(min_train, last_origin, num=n_folds + 1, dtype=int)[1:]
    return sorted(set(int(x) for x in candidates if x >= min_train and x + horizon <= n))


def backtest_model(
    model: BaseForecaster,
    df: pd.DataFrame,
    timestamp_col: str,
    target_col: str,
    exog_cols: list[str],
    seasonal_period: int,
    horizon: int,
    n_folds: int,
) -> BacktestResult:
    start = time.perf_counter()
    pred_rows, metric_rows = [], []
    try:
        origins = _fold_origins(len(df), horizon, n_folds)
        if not origins:
            raise ValueError("Not enough observations for the requested horizon and backtesting.")
        for fold, origin in enumerate(origins, start=1):
            train = df.iloc[:origin].copy()
            test = df.iloc[origin : origin + horizon].copy()
            fitted = copy.deepcopy(model).fit(train, timestamp_col, target_col, exog_cols, seasonal_period)
            fts = [pd.Timestamp(x) for x in test[timestamp_col]]
            fex = test[exog_cols].reset_index(drop=True) if exog_cols else None
            pred = fitted.predict(len(test), fts, fex)
            actual = test[target_col].to_numpy(dtype=float)
            fold_rmse = rmse(actual, pred)
            fold_mae = mae(actual, pred)
            previous_actual = float(train[target_col].iloc[-1]) if len(train) else None
            metric_rows.append({
                "fold": fold,
                "MAE": fold_mae,
                "MdAE": median_ae(actual, pred),
                "RMSE": fold_rmse,
                "NRMSE(std)": nrmse_std(actual, pred),
                "MAPE": mape(actual, pred),
                "sMAPE": smape(actual, pred),
                "WAPE": wape(actual, pred),
                "MASE": mase(actual, pred, train[target_col], seasonal_period),
                "R²": r2(actual, pred),
                "Bias": mean_error(actual, pred),
                "Directional accuracy (%)": directional_accuracy(actual, pred, previous_actual),
            })
            for j, (ts, a, p) in enumerate(zip(fts, actual, pred), start=1):
                pred_rows.append({"fold": fold, "horizon_step": j, "timestamp": ts, "actual": float(a), "prediction": float(p), "error": float(a - p)})
        pm = pd.DataFrame(pred_rows)
        fm = pd.DataFrame(metric_rows)
        metric_cols = ["MAE", "MdAE", "RMSE", "NRMSE(std)", "MAPE", "sMAPE", "WAPE", "MASE", "R²", "Bias", "Directional accuracy (%)"]
        aggregate = {k: float(fm[k].mean()) for k in metric_cols}
        return BacktestResult(model.name, pm, fm, aggregate, time.perf_counter() - start)
    except Exception as exc:
        return BacktestResult(model.name, pd.DataFrame(), pd.DataFrame(), {}, time.perf_counter() - start, failure=str(exc))


def _feature_importance(fitted: BaseForecaster) -> pd.DataFrame:
    if hasattr(fitted, "feature_importance_df_"):
        return fitted.feature_importance_df_.copy()
    if not hasattr(fitted, "model") or not hasattr(fitted, "training_X_"):
        return pd.DataFrame(columns=["feature", "importance"])
    X = fitted.training_X_
    y = fitted.training_y_
    if len(X) > 1500:
        X = X.iloc[-1500:]
        y = y.iloc[-1500:]
    try:
        pi = permutation_importance(fitted.model, X, y, scoring="neg_mean_absolute_error", n_repeats=3, random_state=42, max_samples=min(1.0, 1000 / max(1, len(X))))
        out = pd.DataFrame({"feature": X.columns, "importance": np.maximum(pi.importances_mean, 0.0)})
        total = out["importance"].sum()
        if total > 0:
            out["importance"] = 100 * out["importance"] / total
        return out.sort_values("importance", ascending=False).head(20).reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["feature", "importance"])


def _trust_components(best: BacktestResult, baseline_rmse: float, profile, horizon: int) -> dict[str, float]:
    model_rmse = best.aggregate_metrics.get("RMSE", float("nan"))
    skill = skill_score(model_rmse, baseline_rmse)
    baseline_component = float(np.clip(50 + (0 if not np.isfinite(skill) else skill * 2), 0, 100))
    rmses = best.fold_metrics["RMSE"].to_numpy(dtype=float) if not best.fold_metrics.empty else np.array([])
    cv = float(np.std(rmses) / (np.mean(rmses) + 1e-12)) if len(rmses) > 1 else 0.25
    stability = float(np.clip(100 * (1 - cv), 0, 100))
    data_quality = float(np.clip(100 * (0.55 * profile.regularity_score + 0.45 * (1 - min(profile.missing_target_pct / 25, 1))), 0, 100))
    history_ratio = profile.rows / max(horizon * 10, 30)
    history = float(np.clip(35 + 65 * min(history_ratio, 1), 0, 100))
    drift = float(np.clip(100 * (1 - profile.drift_score), 0, 100))
    return {
        "Baseline improvement": baseline_component,
        "Backtest stability": stability,
        "Data quality": data_quality,
        "History adequacy": history,
        "Drift resilience": drift,
    }


def train_forecast(
    df: pd.DataFrame,
    timestamp_col: str,
    target_col: str,
    exog_cols: list[str],
    profile,
    horizon: int,
    mode: str = "Balanced",
    deep_model_names: list[str] | None = None,
    progress_callback=None,
    scaler_kind: str = "standard",
) -> ForecastResult:
    n_folds = {"fast": 2, "balanced": 3, "maximum accuracy": 4}.get(mode.lower(), 3)
    deep_model_names = list(deep_model_names or [])
    include_mlp = "Deep MLP AR" in deep_model_names
    torch_names = [n for n in deep_model_names if n in {"LSTM", "TCN", "Transformer"}]
    models = build_model_zoo(mode, include_mlp=include_mlp, deep_model_names=torch_names, scaler_kind=scaler_kind)
    results = []
    for idx, model in enumerate(models, start=1):
        if progress_callback is not None:
            progress_callback(idx - 1, len(models), model.name, "running")
        result = backtest_model(model, df, timestamp_col, target_col, exog_cols, profile.selected_seasonal_period, horizon, n_folds)
        results.append(result)
        if progress_callback is not None:
            progress_callback(idx, len(models), model.name, "failed" if result.failure else "complete")
    valid = [r for r in results if not r.failure and np.isfinite(r.aggregate_metrics.get("RMSE", np.nan))]
    if not valid:
        failures = "; ".join(f"{r.model_name}: {r.failure}" for r in results)
        raise RuntimeError(f"All forecasting models failed. {failures}")
    baseline = next((r for r in valid if r.model_name == "Seasonal Naive"), next(r for r in valid if r.model_name == "Naive"))
    baseline_rmse = baseline.aggregate_metrics["RMSE"]
    for r in valid:
        r.aggregate_metrics["Skill vs baseline (%)"] = skill_score(r.aggregate_metrics["RMSE"], baseline_rmse)
    valid.sort(key=lambda x: (x.aggregate_metrics["RMSE"], x.fit_seconds))
    best = valid[0]
    model_template = next(m for m in models if m.name == best.model_name)
    fitted = copy.deepcopy(model_template).fit(df, timestamp_col, target_col, exog_cols, profile.selected_seasonal_period)
    fts = future_timestamps(pd.Timestamp(df[timestamp_col].iloc[-1]), profile.median_step_seconds, horizon)
    fex = _make_future_exog(df, exog_cols, horizon)
    pred = fitted.predict(horizon, fts, fex)
    errors = np.abs(best.predictions["error"].to_numpy(dtype=float))
    global_q = float(np.quantile(errors, 0.90)) if len(errors) else 0.0
    q_by_step = best.predictions.groupby("horizon_step")["error"].apply(lambda s: float(np.quantile(np.abs(s), 0.90))).to_dict() if len(errors) else {}
    intervals = np.array([q_by_step.get(i + 1, global_q) for i in range(horizon)], dtype=float)
    forecast = pd.DataFrame({"timestamp": fts, "forecast": pred, "lower_90": pred - intervals, "upper_90": pred + intervals})
    components = _trust_components(best, baseline_rmse, profile, horizon)
    trust = float(np.average(list(components.values()), weights=[0.30, 0.25, 0.20, 0.15, 0.10]))
    leaderboard = pd.DataFrame([
        {
            "Model": r.model_name,
            "MAE": r.aggregate_metrics["MAE"],
            "MdAE": r.aggregate_metrics["MdAE"],
            "RMSE": r.aggregate_metrics["RMSE"],
            "NRMSE(std)": r.aggregate_metrics["NRMSE(std)"],
            "MAPE": r.aggregate_metrics["MAPE"],
            "sMAPE": r.aggregate_metrics["sMAPE"],
            "WAPE": r.aggregate_metrics["WAPE"],
            "MASE": r.aggregate_metrics["MASE"],
            "R²": r.aggregate_metrics["R²"],
            "Bias": r.aggregate_metrics["Bias"],
            "Directional accuracy (%)": r.aggregate_metrics["Directional accuracy (%)"],
            "Skill vs baseline (%)": r.aggregate_metrics["Skill vs baseline (%)"],
            "Backtest seconds": r.fit_seconds,
        }
        for r in valid
    ]).sort_values("RMSE").reset_index(drop=True)
    diagnostics = {
        "backtest_predictions": best.predictions,
        "backtest_folds": best.fold_metrics,
        "baseline_rmse": baseline_rmse,
        "interval_abs_error_q90": global_q,
        "future_exog_baseline": fex,
        "model_failures": {r.model_name: r.failure for r in results if r.failure},
        "models_attempted": [r.model_name for r in results],
        "scaler_kind": scaler_kind,
    }
    return ForecastResult(best.model_name, forecast, best.aggregate_metrics, trust, components, leaderboard, _feature_importance(fitted), diagnostics, fitted)


def scenario_forecast(result: ForecastResult, df: pd.DataFrame, timestamp_col: str, exog_cols: list[str], column: str, pct_change: float) -> pd.DataFrame:
    if result.fitted_model is None or not getattr(result.fitted_model, "supports_exog", False) or column not in exog_cols:
        raise ValueError("The selected winning model does not support scenario covariates.")
    horizon = len(result.forecast)
    base = _make_future_exog(df, exog_cols, horizon)
    scenario = base.copy()
    scenario[column] = pd.to_numeric(scenario[column], errors="coerce") * (1 + pct_change / 100.0)
    fts = [pd.Timestamp(x) for x in result.forecast["timestamp"]]
    pred = result.fitted_model.predict(horizon, fts, scenario)
    out = result.forecast[["timestamp", "forecast"]].rename(columns={"forecast": "baseline"}).copy()
    out["scenario"] = pred
    out["delta"] = out["scenario"] - out["baseline"]
    out["delta_pct"] = 100 * out["delta"] / np.maximum(np.abs(out["baseline"]), 1e-8)
    return out


def stress_test(result: ForecastResult, df: pd.DataFrame, timestamp_col: str, target_col: str, exog_cols: list[str], profile) -> pd.DataFrame:
    if result.fitted_model is None:
        return pd.DataFrame()
    baseline = result.forecast["forecast"].to_numpy(dtype=float)
    horizon = len(baseline)
    fts = [pd.Timestamp(x) for x in result.forecast["timestamp"]]
    tests = []
    y_std = float(pd.to_numeric(df[target_col], errors="coerce").std())
    for label, modifier in [
        ("Recent target +0.1σ noise", 0.10),
        ("Recent target +0.5σ shift", 0.50),
        ("Recent target -0.5σ shift", -0.50),
    ]:
        alt_df = df.copy()
        k = min(max(3, horizon), max(3, len(alt_df) // 10))
        alt_df.loc[alt_df.index[-k:], target_col] = alt_df[target_col].iloc[-k:].to_numpy(dtype=float) + modifier * y_std
        model = copy.deepcopy(result.fitted_model)
        try:
            model.fit(alt_df, timestamp_col, target_col, exog_cols, profile.selected_seasonal_period)
            fex = _make_future_exog(alt_df, exog_cols, horizon)
            pred = model.predict(horizon, fts, fex)
            sensitivity = float(100 * np.mean(np.abs(pred - baseline)) / (np.mean(np.abs(baseline)) + 1e-8))
            stability = float(np.clip(100 - 5 * sensitivity, 0, 100))
            tests.append({"Stress test": label, "Forecast sensitivity (%)": sensitivity, "Stability score": stability})
        except Exception:
            tests.append({"Stress test": label, "Forecast sensitivity (%)": np.nan, "Stability score": np.nan})
    return pd.DataFrame(tests)
