from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forecastos.analyst import ask_openai, deterministic_brief
from forecastos.data import (
    analyze_features,
    infer_target_column,
    infer_timestamp_column,
    prepare_frame,
    prepare_future_features,
    read_table,
)
from forecastos.deep_models import torch_available
from forecastos.engine import future_timestamps, scenario_forecast, stress_test, train_forecast
from forecastos.human_factors import compute_estimate, model_plain_name, readiness, trust_label
from forecastos.profile import profile_timeseries
from forecastos.scaling import SCALER_LABELS, recommend_scaler
from forecastos.sample import make_sample

st.set_page_config(
    page_title="ForecastOS · Forecast Studio",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
:root {--ink:#111827;--muted:#5B6472;--line:#DDE3EA;--soft:#F7F9FC;--accent:#1D4ED8;--ok:#166534;--warn:#92400E;--bad:#991B1B;}
.block-container{max-width:1320px;padding-top:1.1rem;padding-bottom:4rem;}
h1,h2,h3{letter-spacing:-.02em;color:var(--ink)}
p,li{line-height:1.55}
.fos-top{border-bottom:1px solid var(--line);padding:.2rem 0 1rem;margin-bottom:1.2rem}
.fos-top h1{font-size:2.05rem;margin:0}.fos-top p{color:var(--muted);margin:.25rem 0 0;max-width:900px}
.step{display:flex;align-items:center;gap:.65rem;margin:1.8rem 0 .7rem}.step-num{width:30px;height:30px;border-radius:9px;background:#EEF3FF;color:#24499B;font-weight:800;display:flex;align-items:center;justify-content:center}.step h2{font-size:1.25rem;margin:0}
.panel{border:1px solid var(--line);border-radius:16px;padding:1rem 1.1rem;background:white;margin-bottom:.7rem}.panel p{color:var(--muted);margin:.25rem 0}
.window{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;border:1px solid var(--line);background:var(--soft);padding:.8rem 1rem;border-radius:12px;overflow:auto;white-space:nowrap}
.note{border-left:4px solid #64748B;background:#F8FAFC;padding:.72rem .9rem;border-radius:8px;margin:.55rem 0}.note.ok{border-color:#15803D;background:#F0FDF4}.note.warn{border-color:#B45309;background:#FFFBEB}.note.bad{border-color:#B91C1C;background:#FEF2F2}
.small{font-size:.88rem;color:var(--muted)}
div[data-testid="stMetric"]{border:1px solid var(--line);padding:.78rem .88rem;border-radius:14px;background:white;min-height:102px}
button,[role="button"],input,select,textarea{min-height:42px}*:focus-visible{outline:3px solid #2563EB!important;outline-offset:2px!important}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{animation-duration:.001ms!important;transition-duration:.001ms!important}}
</style>
""",
    unsafe_allow_html=True,
)


def step_header(number: int, title: str) -> None:
    st.markdown(f'<div class="step"><div class="step-num">{number}</div><h2>{title}</h2></div>', unsafe_allow_html=True)


def note(text: str, level: str = "neutral") -> None:
    cls = {"ok": "ok", "warn": "warn", "bad": "bad"}.get(level, "")
    st.markdown(f'<div class="note {cls}">{text}</div>', unsafe_allow_html=True)


def span_text(steps: int, step_seconds: float) -> str:
    if not np.isfinite(step_seconds) or step_seconds <= 0:
        return f"{steps} steps"
    sec = steps * step_seconds
    if sec < 3600:
        return f"{steps} steps ≈ {sec/60:.0f} minutes"
    if sec < 86400:
        return f"{steps} steps ≈ {sec/3600:.1f} hours"
    return f"{steps} steps ≈ {sec/86400:.1f} days"


def clear_results() -> None:
    for k in ["forecast_result", "forecast_profile", "forecast_work", "forecast_config", "stress", "scenario"]:
        st.session_state.pop(k, None)


st.markdown(
    """
<div class="fos-top">
  <h1>ForecastOS</h1>
  <p>Configure the forecasting problem explicitly. ForecastOS will not silently choose your target, history window, forecast horizon, or future weather/features.</p>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 1. DATA
# -----------------------------------------------------------------------------
step_header(1, "Choose the data")
source = st.radio("Data source", ["Upload my dataset", "Built-in energy example"], horizontal=True, label_visibility="collapsed")
read_meta = None
uploaded = None
if source == "Upload my dataset":
    uploaded = st.file_uploader("Historical dataset", type=["csv", "xlsx", "xls", "parquet"], help="Use one chronological table. You do not need to pre-build lag/sliding-window columns.")
    if uploaded is None:
        st.info("Upload a historical dataset to begin. The table should contain a timestamp column and the target you want to forecast.")
        st.stop()
    try:
        raw, read_meta = read_table(uploaded, return_metadata=True)
    except Exception as exc:
        st.error(f"Could not read the historical file: {exc}")
        st.stop()
else:
    raw = make_sample()

if len(raw) > 250_000:
    st.warning("The interactive Streamlit build uses the most recent 250,000 rows to protect memory. The uploaded source file is not changed.")
    raw = raw.tail(250_000).copy()

meta_cols = st.columns(4)
meta_cols[0].metric("Rows", f"{len(raw):,}")
meta_cols[1].metric("Columns", f"{len(raw.columns):,}")
meta_cols[2].metric("Missing cells", f"{int(raw.isna().sum().sum()):,}")
header_text = "Built-in schema"
if read_meta is not None:
    header_text = "Generic names" if read_meta.detected_header_row is None else f"Header row {read_meta.detected_header_row + 1}"
meta_cols[3].metric("Table header", header_text)
with st.expander("Preview raw data"):
    st.dataframe(raw.head(30), use_container_width=True, hide_index=True)
    if read_meta is not None:
        for msg in read_meta.notes:
            st.caption(msg)

# -----------------------------------------------------------------------------
# 2. EXPLICIT MAPPING
# -----------------------------------------------------------------------------
step_header(2, "Map the forecasting columns")
columns = list(raw.columns)
if source == "Built-in energy example":
    ts_default = columns.index("timestamp") if "timestamp" in columns else None
else:
    ts_default = None

ts_col = st.selectbox(
    "Timestamp column",
    columns,
    index=ts_default,
    placeholder="Choose the timestamp column",
    help="ForecastOS does not automatically commit to a timestamp column. You choose it here.",
)
if ts_col is None:
    with st.expander("Optional mapping helper"):
        st.write(f"Suggested timestamp candidate: **{infer_timestamp_column(raw) or 'none'}**. This is only a suggestion; nothing is selected automatically.")
    st.stop()

numeric_candidates = []
for c in columns:
    if c == ts_col:
        continue
    converted = pd.to_numeric(raw[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
    if converted.notna().mean() >= 0.70:
        numeric_candidates.append(c)
if not numeric_candidates:
    st.error("No column is numeric enough to use as a forecasting target after selecting the timestamp.")
    st.stop()

target_default = numeric_candidates.index("load") if source == "Built-in energy example" and "load" in numeric_candidates else None
target_col = st.selectbox(
    "Target column to forecast",
    numeric_candidates,
    index=target_default,
    placeholder="Choose the target",
    help="This is the variable ForecastOS will predict.",
)
if target_col is None:
    with st.expander("Optional mapping helper"):
        suggestion = infer_target_column(raw, ts_col)
        st.write(f"Suggested numeric target candidate: **{suggestion or 'none'}**. You remain in control of the selection.")
    st.stop()

feature_candidates = [c for c in columns if c not in {ts_col, target_col}]
future_feature_cols = st.multiselect(
    "Features whose future values will be available",
    feature_candidates,
    default=[],
    help="Examples: weather forecast, tariff schedule, holidays, promotions. Select a feature only if you can supply its value for every future forecast step. Leaving this empty produces a target-history + calendar forecast.",
)

feature_scan = analyze_features(raw, ts_col, target_col)
with st.expander("Feature audit — advisory only"):
    st.caption("ForecastOS can flag sparse, ID-like or leakage-looking columns, but it does not automatically add them to the model.")
    if feature_scan.empty:
        st.write("No additional columns to audit.")
    else:
        st.dataframe(feature_scan, use_container_width=True, hide_index=True)

# Initial time profile solely to convert step counts to real time.
try:
    raw_profile = profile_timeseries(raw, ts_col, target_col)
except Exception as exc:
    st.error(f"The selected timestamp/target mapping cannot form a time series: {exc}")
    st.stop()

# -----------------------------------------------------------------------------
# 3. WINDOWS
# -----------------------------------------------------------------------------
step_header(3, "Define the sliding window and forecast horizon")
max_context = max(2, min(720, max(2, len(raw) // 3)))
max_horizon = max(1, min(336, max(1, len(raw) // 5)))
w1, w2 = st.columns(2)
history_window = int(w1.number_input(
    "Previous steps used as history",
    min_value=2,
    max_value=max_context,
    value=min(24, max_context),
    step=1,
    help="If you choose 48, the model receives the previous 48 target observations. Sequence models use a 48-row context window; autoregressive tabular models receive lag_1 through lag_48.",
))
horizon = int(w2.number_input(
    "Future steps to forecast",
    min_value=1,
    max_value=max_horizon,
    value=min(12, max_horizon),
    step=1,
    help="The exact number of future time steps ForecastOS should predict.",
))

st.markdown(
    f'<div class="window">[ previous {history_window} steps ]  →  model  →  [ next {horizon} forecast steps ]</div>',
    unsafe_allow_html=True,
)
wc1, wc2 = st.columns(2)
wc1.caption(span_text(history_window, raw_profile.median_step_seconds))
wc2.caption(span_text(horizon, raw_profile.median_step_seconds))

note(
    "<strong>Weather format:</strong> keep historical weather in ordinary timestamped rows. Do not manually create <code>temp_t-1</code>, <code>temp_t-2</code>, etc. ForecastOS constructs windows internally. If weather will be used for future forecasting, supply a separate future-weather table for the horizon below.",
    "ok",
)

# -----------------------------------------------------------------------------
# 4. PREPROCESSING
# -----------------------------------------------------------------------------
step_header(4, "Choose preprocessing")
p1, p2 = st.columns(2)
regularize_grid = p1.checkbox(
    "Repair manageable timestamp gaps",
    value=True,
    help="Rebuilds a regular grid only when the inferred interval is sufficiently stable and the expansion is bounded. Target filling remains causal.",
)
scaler_choice = p2.selectbox(
    "Normalization / standardization",
    ["Auto", "Standard (z-score)", "Robust (median/IQR)", "Min-Max [0,1]"],
    index=0,
    help="Scalers are fitted inside each training fold. Auto chooses from the three supported scalers after inspecting the prepared training data.",
)

try:
    work, prep_warnings, prep_report = prepare_frame(
        raw,
        ts_col,
        target_col,
        future_feature_cols,
        regularize=regularize_grid,
        return_report=True,
    )
except Exception as exc:
    st.error(f"Preprocessing failed: {exc}")
    st.stop()

profile = profile_timeseries(work, ts_col, target_col)
auto_scaler, scaler_diag = recommend_scaler(work, [target_col] + prep_report.model_features)
scaler_kind = auto_scaler if scaler_choice == "Auto" else {
    "Standard (z-score)": "standard",
    "Robust (median/IQR)": "robust",
    "Min-Max [0,1]": "minmax",
}[scaler_choice]

pp = st.columns(5)
pp[0].metric("Rows ready", f"{len(work):,}")
pp[1].metric("Duplicates merged", prep_report.duplicate_timestamps_aggregated)
pp[2].metric("Gaps inserted", prep_report.inserted_time_rows)
pp[3].metric("Missing target after", prep_report.target_missing_after)
pp[4].metric("Scaler", SCALER_LABELS.get(scaler_kind, scaler_kind))
if scaler_choice == "Auto":
    st.caption(f"Auto-scaler reason: {scaler_diag.get('reason', '')}")
if prep_warnings:
    with st.expander(f"Preprocessing notes ({len(prep_warnings)})"):
        for msg in prep_warnings:
            st.warning(msg)

# -----------------------------------------------------------------------------
# 5. EXPLICIT FUTURE FEATURES
# -----------------------------------------------------------------------------
step_header(5, "Supply future weather / external features")
expected_ts = future_timestamps(pd.Timestamp(work[ts_col].iloc[-1]), profile.median_step_seconds, horizon)
future_exog = None
future_ready = not future_feature_cols
future_warnings: list[str] = []
future_alignment = None
feature_mapping: dict[str, str] = {}

if not future_feature_cols:
    note("No external future features selected. ForecastOS will use target history and calendar/time features only. It will not invent future weather values.", "ok")
else:
    st.write("You selected: **" + ", ".join(future_feature_cols) + "**. Supply one future value per selected feature for every forecast step.")
    template = pd.DataFrame({ts_col: expected_ts})
    for c in future_feature_cols:
        template[c] = np.nan
    st.download_button(
        "Download future-feature template",
        template.to_csv(index=False).encode("utf-8"),
        file_name="forecastos_future_features_template.csv",
        mime="text/csv",
        help="Fill these rows with your weather forecast or other known-future feature values, then upload the file below.",
    )
    future_upload = st.file_uploader("Future weather / feature file", type=["csv", "xlsx", "xls", "parquet"], key="future_upload")
    if future_upload is None:
        note("A future-feature file is required because external features are selected. You can remove those features to run a target-only forecast.", "warn")
    else:
        try:
            future_raw, future_meta = read_table(future_upload, return_metadata=True)
        except Exception as exc:
            st.error(f"Could not read future feature file: {exc}")
            future_raw = None
        if future_raw is not None:
            fc = list(future_raw.columns)
            st.dataframe(future_raw.head(min(20, horizon)), use_container_width=True, hide_index=True)
            align_options = ["Use row order (explicit)"] + fc
            future_alignment = st.selectbox(
                "How should future rows be aligned?",
                align_options,
                index=None,
                placeholder="Choose row order or a future timestamp column",
                help="Timestamp alignment is safer for weather data. Row-order alignment is available only when you intentionally prepared the rows in exact forecast order.",
            )
            if future_alignment is not None:
                future_time_col = None if future_alignment == "Use row order (explicit)" else future_alignment
                st.markdown("**Map each historical feature to its future-data column**")
                for feat in future_feature_cols:
                    exact_idx = fc.index(feat) if feat in fc else None
                    mapped = st.selectbox(
                        f"Future column for `{feat}`",
                        fc,
                        index=exact_idx,
                        placeholder=f"Choose future values for {feat}",
                        key=f"future_map__{feat}",
                    )
                    if mapped is not None:
                        feature_mapping[feat] = mapped
                if len(feature_mapping) == len(future_feature_cols):
                    try:
                        future_exog, future_warnings = prepare_future_features(
                            future_raw,
                            feature_mapping,
                            prep_report,
                            work,
                            horizon,
                            expected_timestamps=expected_ts,
                            future_timestamp_col=future_time_col,
                        )
                        future_ready = True
                        note(f"Future features validated for all {horizon} forecast steps.", "ok")
                        preview_future = pd.DataFrame({"forecast_timestamp": expected_ts})
                        for c in future_feature_cols:
                            preview_future[c] = future_exog[c].to_numpy()
                        with st.expander("Preview encoded/aligned future features"):
                            st.dataframe(preview_future, use_container_width=True, hide_index=True)
                    except Exception as exc:
                        st.error(f"Future feature validation failed: {exc}")
                if future_warnings:
                    for msg in future_warnings:
                        st.warning(msg)

# -----------------------------------------------------------------------------
# 6. MODELS
# -----------------------------------------------------------------------------
step_header(6, "Choose model effort")
m1, m2 = st.columns([1, 1.4])
mode = m1.radio("Backtest effort", ["Fast", "Balanced", "Maximum accuracy"], index=1, horizontal=False)

torch_ok = torch_available()
deep_options = ["Deep MLP AR"] + (["LSTM", "TCN", "Transformer"] if torch_ok else [])
deep_models = m2.multiselect(
    "Optional deep-learning models",
    deep_options,
    default=[],
    help="Classical baselines, Ridge AR and gradient boosting remain in the tournament. Deep models are opt-in and must beat simpler models on temporal backtests.",
)
if not torch_ok:
    m2.caption("Install the deep/PyTorch requirements to expose LSTM, TCN and Transformer.")
burden, burden_text = compute_estimate(len(work), mode, deep_models)
m2.caption(f"Estimated compute burden: **{burden}** · {burden_text}")

# -----------------------------------------------------------------------------
# 7. REVIEW + RUN
# -----------------------------------------------------------------------------
step_header(7, "Review the experiment and run")
read = readiness(profile, horizon, len(work))
review_cols = st.columns(5)
review_cols[0].metric("Target", target_col)
review_cols[1].metric("History window", history_window)
review_cols[2].metric("Forecast horizon", horizon)
review_cols[3].metric("Future features", len(future_feature_cols))
review_cols[4].metric("Frequency", profile.inferred_frequency)

st.markdown(
    f'<div class="window">Target: {target_col} &nbsp;|&nbsp; use t−1 … t−{history_window} &nbsp;|&nbsp; forecast t+1 … t+{horizon} &nbsp;|&nbsp; scaler: {SCALER_LABELS.get(scaler_kind, scaler_kind)}</div>',
    unsafe_allow_html=True,
)
if history_window + horizon * 2 + 12 >= len(work):
    note("The selected history window/horizon leaves relatively little data for temporal training folds. Reduce the history window or horizon, or provide more history.", "warn")
if read.issues:
    with st.expander("Data-readiness notes"):
        for x in read.issues:
            st.warning(x)

run_disabled = not future_ready
run = st.button("Run forecasting tournament", type="primary", disabled=run_disabled, use_container_width=True)
if run_disabled:
    st.caption("Run is disabled until all selected future features are explicitly supplied and aligned.")

if run:
    clear_results()
    status = st.status("Running temporal backtests…", expanded=True)
    progress = st.progress(0, text="Preparing models")

    def progress_callback(done: int, total: int, model_name: str, state: str) -> None:
        pct = 0 if total <= 0 else min(100, int(100 * done / total))
        human = model_plain_name(model_name)
        if state == "running":
            progress.progress(pct, text=f"Testing {human}")
            status.write(f"Testing **{human}** · `{model_name}`")
        elif state == "failed":
            status.write(f"⚠️ {human} could not complete and was excluded.")
        else:
            progress.progress(pct, text=f"Completed {done} of {total} models")

    try:
        result = train_forecast(
            work,
            ts_col,
            target_col,
            future_feature_cols,
            profile,
            horizon,
            mode,
            deep_model_names=deep_models,
            progress_callback=progress_callback,
            scaler_kind=scaler_kind,
            history_window=history_window,
            future_exog=future_exog,
        )
        progress.progress(100, text="Tournament complete")
        status.update(label=f"Complete · {model_plain_name(result.model_name)} selected", state="complete", expanded=False)
    except Exception as exc:
        status.update(label="Forecasting failed", state="error", expanded=True)
        st.error(str(exc))
        st.stop()

    st.session_state["forecast_result"] = result
    st.session_state["forecast_profile"] = profile
    st.session_state["forecast_work"] = work
    st.session_state["forecast_config"] = {
        "timestamp": ts_col,
        "target": target_col,
        "features": future_feature_cols,
        "history_window": history_window,
        "horizon": horizon,
        "mode": mode,
        "deep_models": deep_models,
        "scaler": scaler_kind,
        "regularize": regularize_grid,
        "future_alignment": future_alignment,
    }

result = st.session_state.get("forecast_result")
if result is None:
    st.stop()

# -----------------------------------------------------------------------------
# RESULTS
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("## Results")
trust_word, trust_help = trust_label(result.trust_score)
skill = result.metrics.get("Skill vs baseline (%)", float("nan"))
r = st.columns(5)
r[0].metric("Selected model", model_plain_name(result.model_name), help=f"Technical model: {result.model_name}")
r[1].metric("Forecast Trust", f"{result.trust_score:.0f}/100 · {trust_word}", help=trust_help)
r[2].metric("RMSE", f"{result.metrics['RMSE']:.3f}")
r[3].metric("MAE", f"{result.metrics['MAE']:.3f}")
r[4].metric("Skill vs seasonal", "—" if not np.isfinite(skill) else f"{skill:+.1f}%")

if skill < 0:
    note("The selected model did not beat the seasonal-naïve baseline. Treat the forecast cautiously and reconsider the history window, horizon, or data.", "bad")
elif result.trust_score < 70:
    note("Backtests show useful signal, but trust is not yet strong. Inspect fold stability and stress tests before operational use.", "warn")
else:
    note("Backtests support the selected model relative to the tested alternatives. Continue to use uncertainty intervals and domain judgment.", "ok")

forecast_tab, compare_tab, explain_tab, stress_tab, analyst_tab, export_tab = st.tabs([
    "Forecast", "Model comparison", "Explainability", "Stress & scenarios", "AI analyst", "Export & reproducibility"
])

with forecast_tab:
    hist = work[[ts_col, target_col]].tail(max(history_window * 2, horizon * 4, 100))
    f = result.forecast
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist[ts_col], y=hist[target_col], mode="lines", name="Observed history"))
    fig.add_trace(go.Scatter(x=f["timestamp"], y=f["upper_90"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=f["timestamp"], y=f["lower_90"], mode="lines", fill="tonexty", line=dict(width=0), name="90% empirical interval"))
    fig.add_trace(go.Scatter(x=f["timestamp"], y=f["forecast"], mode="lines+markers", name="Forecast", line=dict(width=2.5)))
    fig.update_layout(height=460, margin=dict(l=8, r=8, t=12, b=8), hovermode="x unified", xaxis_title="Time", yaxis_title=target_col, legend_title_text="")
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    st.caption(f"Model input uses the previous {history_window} steps and predicts the next {horizon} steps recursively. The interval is empirical from rolling-backtest absolute errors.")

    metric_order = ["MAE", "MdAE", "RMSE", "NRMSE(std)", "MAPE", "sMAPE", "WAPE", "MASE", "R²", "Bias", "Directional accuracy (%)", "Skill vs baseline (%)"]
    rows = []
    for k in metric_order:
        if k in result.metrics:
            rows.append({"Metric": k, "Value": result.metrics[k]})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with compare_tab:
    board = result.leaderboard.copy()
    board.insert(1, "Plain-language model", board["Model"].map(model_plain_name))
    st.dataframe(board, use_container_width=True, hide_index=True)
    st.caption("The winner is the model with the lowest rolling-backtest RMSE; compute time is only a tie-breaker. More complex models receive no preference.")
    trust_df = pd.DataFrame({"Component": list(result.trust_components.keys()), "Score": list(result.trust_components.values())}).sort_values("Score")
    st.bar_chart(trust_df.set_index("Component"), horizontal=True)
    with st.expander("Backtest fold details"):
        st.dataframe(result.diagnostics["backtest_folds"], use_container_width=True, hide_index=True)

with explain_tab:
    if result.feature_importance.empty:
        st.info("The winning model does not expose feature-based importance in this prototype.")
    else:
        imp = result.feature_importance.head(16).sort_values("importance")
        st.bar_chart(imp.set_index("feature"), horizontal=True)
        st.caption("Predictive sensitivity is not causation. Sequence-model explanations distinguish historical channels from `future__feature` values supplied for the forecast step.")
    st.markdown(f"**History window:** lag_1 through lag_{history_window} for supervised autoregressive models; an exact {history_window}-row sliding sequence for LSTM/TCN/Transformer.")

with stress_tab:
    if st.button("Run stress tests", key="stress_run"):
        with st.status("Perturbing recent target context…", expanded=False) as ss:
            st.session_state["stress"] = stress_test(result, work, ts_col, target_col, future_feature_cols, profile)
            ss.update(label="Stress tests complete", state="complete")
    if "stress" in st.session_state:
        st.dataframe(st.session_state["stress"], use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Scenario lab")
    scenario_candidates = [c for c in prep_report.numeric_features if c in future_feature_cols]
    if scenario_candidates:
        c1, c2 = st.columns(2)
        scenario_col = c1.selectbox("Future feature to perturb", scenario_candidates)
        pct = c2.slider("Change across future horizon (%)", -50, 50, 10, 1)
        if st.button("Run scenario", key="scenario_run"):
            try:
                st.session_state["scenario"] = scenario_forecast(result, work, ts_col, future_feature_cols, scenario_col, pct)
            except Exception as exc:
                st.warning(str(exc))
        if "scenario" in st.session_state:
            sc = st.session_state["scenario"]
            sf = go.Figure()
            sf.add_trace(go.Scatter(x=sc["timestamp"], y=sc["baseline"], name="Baseline", mode="lines"))
            sf.add_trace(go.Scatter(x=sc["timestamp"], y=sc["scenario"], name="Scenario", mode="lines"))
            sf.update_layout(height=340, margin=dict(l=8, r=8, t=12, b=8), hovermode="x unified", xaxis_title="Time", yaxis_title=target_col)
            st.plotly_chart(sf, use_container_width=True, config={"displaylogo": False})
            st.metric("Average model response", f"{sc['delta_pct'].mean():+.2f}%")
    else:
        st.info("Select and explicitly upload at least one numeric future feature to use the scenario lab.")

with analyst_tab:
    st.markdown(deterministic_brief(result, profile))
    st.caption("The optional AI analyst receives structured experiment evidence, not the raw uploaded dataset.")
    q = st.text_input("Ask about this experiment", placeholder="Why did this model win, and what is the biggest reliability risk?")
    if q and st.button("Ask AI analyst", key="ai_ask"):
        if os.getenv("OPENAI_API_KEY"):
            try:
                st.write(ask_openai(q, result, profile))
            except Exception as exc:
                st.warning(f"AI analyst unavailable: {exc}")
        else:
            st.info("Add OPENAI_API_KEY to Streamlit Secrets to enable the optional analyst. Forecasting itself does not need an API key.")

with export_tab:
    st.download_button(
        "Download forecast CSV",
        result.forecast.to_csv(index=False).encode("utf-8"),
        file_name="forecastos_forecast.csv",
        mime="text/csv",
    )
    st.json({
        "timestamp_column": ts_col,
        "target": target_col,
        "history_window_previous_steps": history_window,
        "forecast_horizon_steps": horizon,
        "frequency": profile.inferred_frequency,
        "future_features": future_feature_cols,
        "future_feature_mapping": feature_mapping,
        "future_alignment": future_alignment,
        "scaler": SCALER_LABELS.get(scaler_kind, scaler_kind),
        "regularize_timestamp_grid": regularize_grid,
        "selected_model": result.model_name,
        "deep_models_requested": deep_models,
        "metrics": result.metrics,
        "trust_score": result.trust_score,
    })
    with st.expander("Important limitations"):
        st.write("• ForecastOS does not fabricate future external-feature values. Selected weather/exogenous variables require an explicit future file.")
        st.write("• Backtests use the historically observed values of features selected as future-known; only select variables that would genuinely have been available at the forecast origin.")
        st.write("• Forecast intervals are empirical residual bands, not guaranteed probability coverage.")
        st.write("• Explainability and scenario responses describe model behavior, not causal effects.")
