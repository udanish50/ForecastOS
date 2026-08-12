import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forecastos.data import infer_target_column, infer_timestamp_column, prepare_frame
from forecastos.engine import train_forecast
from forecastos.metrics import mae, rmse, smape
from forecastos.profile import profile_timeseries
from forecastos.sample import make_sample


def test_column_inference_and_profile():
    df = make_sample(24 * 30)
    ts = infer_timestamp_column(df)
    target = infer_target_column(df, ts)
    assert ts == "timestamp"
    assert target == "load"
    profile = profile_timeseries(df, ts, target)
    assert profile.rows == len(df)
    assert profile.median_step_seconds == 3600
    assert profile.regularity_score > 0.99
    assert 24 in profile.candidate_seasonal_periods


def test_metrics_are_sane():
    a = [1, 2, 3]
    b = [1, 2, 4]
    assert np.isclose(mae(a, b), 1 / 3)
    assert np.isclose(rmse(a, b), np.sqrt(1 / 3))
    assert smape(a, a) == 0


def test_end_to_end_fast_forecast():
    raw = make_sample(24 * 35)
    work, _ = prepare_frame(raw, "timestamp", "load", [])
    profile = profile_timeseries(work, "timestamp", "load")
    result = train_forecast(work, "timestamp", "load", [], profile, horizon=12, mode="Fast")
    assert len(result.forecast) == 12
    assert result.forecast["forecast"].notna().all()
    assert 0 <= result.trust_score <= 100
    assert not result.leaderboard.empty


def test_exogenous_scenario_capable_model_can_train():
    raw = make_sample(24 * 35)
    work, _ = prepare_frame(raw, "timestamp", "load", ["temperature"])
    profile = profile_timeseries(work, "timestamp", "load")
    result = train_forecast(work, "timestamp", "load", ["temperature"], profile, horizon=8, mode="Balanced")
    assert len(result.forecast) == 8


def test_deep_mlp_can_join_tournament():
    raw = make_sample(24 * 24)
    work, _ = prepare_frame(raw, "timestamp", "load", ["temperature"])
    profile = profile_timeseries(work, "timestamp", "load")
    result = train_forecast(
        work,
        "timestamp",
        "load",
        ["temperature"],
        profile,
        horizon=6,
        mode="Fast",
        deep_model_names=["Deep MLP AR"],
    )
    assert "Deep MLP AR" in result.diagnostics["models_attempted"]
    assert len(result.forecast) == 6


def test_optional_torch_sequence_models_smoke():
    import pytest
    from forecastos.deep_models import TorchSequenceForecaster, torch_available

    if not torch_available():
        pytest.skip("Optional PyTorch dependency is not installed.")

    raw = make_sample(24 * 10)
    work, _ = prepare_frame(raw, "timestamp", "load", ["temperature"])
    step = work["timestamp"].iloc[-1] - work["timestamp"].iloc[-2]
    future_ts = [work["timestamp"].iloc[-1] + (i + 1) * step for i in range(2)]
    import pandas as pd
    future_exog = pd.concat([work[["temperature"]].tail(1)] * 2, ignore_index=True)

    for kind in ["lstm", "tcn", "transformer"]:
        model = TorchSequenceForecaster(kind, "fast").fit(work, "timestamp", "load", ["temperature"], 24)
        pred = model.predict(2, future_ts, future_exog)
        assert len(pred) == 2
        assert np.isfinite(pred).all()
        assert not model.feature_importance_df_.empty


def test_messy_csv_header_detection_and_feature_audit(tmp_path):
    import pandas as pd
    from forecastos.data import analyze_features, read_table

    p = tmp_path / "messy.csv"
    p.write_text(
        "Forecast export for customer XYZ\n"
        "Generated automatically\n"
        "timestamp,load,temperature,site_id,promo_schedule\n"
        "2026-01-01 00:00,10,2.1,A,0\n"
        "2026-01-01 01:00,11,2.2,A,1\n"
        "2026-01-01 02:00,12,,A,0\n"
    )
    df, meta = read_table(p, return_metadata=True)
    assert meta.detected_header_row == 2
    assert list(df.columns) == ["timestamp", "load", "temperature", "site_id", "promo_schedule"]
    ft = analyze_features(df, "timestamp", "load")
    assert "temperature" in ft["feature"].tolist()
    assert bool(ft.loc[ft["feature"] == "promo_schedule", "suggested_known_future"].iloc[0])


def test_causal_missing_value_preparation_and_categorical_encoding():
    import pandas as pd

    raw = make_sample(24 * 8)
    raw.loc[5:7, "load"] = np.nan
    raw["weather_state"] = ["clear", "cloudy"] * (len(raw) // 2)
    raw.loc[3:5, "weather_state"] = None
    work, warnings, report = prepare_frame(
        raw,
        "timestamp",
        "load",
        ["temperature", "weather_state"],
        return_report=True,
    )
    assert work["load"].isna().sum() == 0
    assert work["temperature"].isna().sum() == 0
    assert work["weather_state"].isna().sum() == 0
    assert "weather_state" in report.categorical_encoding
    assert report.target_missing_before >= 3


def test_three_scalers_are_available_and_auto_selects_one():
    import pandas as pd
    from forecastos.scaling import make_scaler, recommend_scaler

    x = np.array([[1.0], [2.0], [3.0], [100.0]])
    for kind in ["standard", "robust", "minmax"]:
        z = make_scaler(kind).fit_transform(x)
        assert np.isfinite(z).all()
    df = pd.DataFrame({"x": x.ravel()})
    kind, diag = recommend_scaler(df, ["x"])
    assert kind in {"standard", "robust", "minmax"}
    assert "reason" in diag


def test_extended_metrics_present_in_forecast():
    raw = make_sample(24 * 20)
    work, _ = prepare_frame(raw, "timestamp", "load", [])
    profile = profile_timeseries(work, "timestamp", "load")
    result = train_forecast(work, "timestamp", "load", [], profile, horizon=6, mode="Fast", scaler_kind="robust")
    for key in ["MdAE", "NRMSE(std)", "MAPE", "WAPE", "R²", "Bias", "Directional accuracy (%)"]:
        assert key in result.metrics
        assert key in result.leaderboard.columns
