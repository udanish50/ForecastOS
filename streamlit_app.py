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
from forecastos.data import analyze_features, infer_target_column, infer_timestamp_column, prepare_frame, read_table
from forecastos.deep_models import torch_available
from forecastos.engine import scenario_forecast, stress_test, train_forecast
from forecastos.human_factors import compute_estimate, model_plain_name, readiness, trust_label
from forecastos.profile import profile_timeseries
from forecastos.scaling import SCALER_LABELS, recommend_scaler
from forecastos.sample import make_sample

st.set_page_config(
    page_title="ForecastOS · Autonomous Forecast Scientist",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Human-factors goals: calm hierarchy, high contrast, keyboard-visible focus,
# generous target sizes, restrained motion, and readable line lengths.
st.markdown(
    """
<style>
:root { --fos-blue:#1D4ED8; --fos-ink:#111827; --fos-muted:#4B5563; --fos-border:#D7DEE8; --fos-soft:#F7F9FC; }
.block-container {max-width: 1380px; padding-top: 1.15rem; padding-bottom: 3rem;}
html, body, [class*="css"] {line-height: 1.55;}
.hero {padding: 1.55rem 1.7rem; border: 1px solid var(--fos-border); border-radius: 18px; background: linear-gradient(180deg,#fff,#fbfcff); margin-bottom: .9rem;}
.hero h1 {margin: 0 0 .2rem 0; font-size: clamp(2rem,4vw,2.65rem); letter-spacing:-.03em; color:var(--fos-ink);}
.hero p {max-width: 850px; margin:.35rem 0 0; color:var(--fos-muted); font-size:1.02rem;}
.eyebrow {font-size:.78rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:#3157A4;}
.fos-card {border:1px solid var(--fos-border); border-radius:16px; padding:1rem 1.1rem; background:#fff; height:100%;}
.fos-card h3 {font-size:1rem; margin:0 0 .25rem;}
.fos-card p {margin:.15rem 0; color:var(--fos-muted);}
.fos-note {border-left:4px solid #64748B; background:#F8FAFC; padding:.75rem .9rem; border-radius:8px; margin:.6rem 0;}
.fos-success {border-left-color:#15803D; background:#F0FDF4;}
.fos-review {border-left-color:#A16207; background:#FFFBEB;}
.fos-caution {border-left-color:#B91C1C; background:#FEF2F2;}
.small-muted {color:var(--fos-muted); font-size:.88rem;}
div[data-testid="stMetric"] {border:1px solid var(--fos-border); padding:.8rem .9rem; border-radius:14px; background:#fff; min-height:105px;}
button, [role="button"], input, select, textarea {min-height: 40px;}
*:focus-visible {outline:3px solid #2563EB !important; outline-offset:2px !important;}
[data-testid="stSidebar"] {border-right:1px solid #E5E7EB;}
[data-testid="stSidebar"] .block-container {padding-top:1.2rem;}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {animation-duration:.001ms !important; animation-iteration-count:1 !important; transition-duration:.001ms !important; scroll-behavior:auto !important;}
}
</style>
""",
    unsafe_allow_html=True,
)


def card(title: str, body: str, eyebrow: str | None = None) -> None:
    top = f'<div class="eyebrow">{eyebrow}</div>' if eyebrow else ""
    st.markdown(f'<div class="fos-card">{top}<h3>{title}</h3><p>{body}</p></div>', unsafe_allow_html=True)


def note(text: str, level: str = "neutral") -> None:
    klass = {"ready": "fos-success", "review": "fos-review", "caution": "fos-caution"}.get(level, "")
    st.markdown(f'<div class="fos-note {klass}">{text}</div>', unsafe_allow_html=True)


def clear_experiment() -> None:
    for key in ["forecast_result", "forecast_profile", "forecast_work", "forecast_config", "stress", "scenario"]:
        st.session_state.pop(key, None)


st.markdown(
    """
<div class="hero">
  <div class="eyebrow">ForecastOS</div>
  <h1>Forecast with evidence, not guesswork.</h1>
  <p>Upload a time series, benchmark statistical, machine-learning and optional deep-learning models, then inspect uncertainty, trust, explanations and stress tests before using the forecast.</p>
</div>
""",
    unsafe_allow_html=True,
)

intro_cols = st.columns(3)
with intro_cols[0]:
    card("1 · Understand the data", "ForecastOS checks frequency, seasonality, missingness, drift and whether the requested horizon is reasonable.")
with intro_cols[1]:
    card("2 · Compete models fairly", "Models are compared using rolling temporal backtests. A complex neural model wins only when the evidence supports it.")
with intro_cols[2]:
    card("3 · Decide with context", "Forecast intervals, trust components, explanations and stress tests are shown before export or AI interpretation.")

with st.sidebar:
    st.header("Set up forecast")
    st.caption("Required choices are shown first. Advanced controls are optional.")
    source = st.radio("Data", ["Built-in energy demo", "Upload my data"], index=0, help="Use the demo to learn the workflow before uploading sensitive or business data.")
    uploaded = None
    if source == "Upload my data":
        uploaded = st.file_uploader("Choose CSV, Excel, or Parquet", type=["csv", "xlsx", "xls", "parquet"])
    if st.button("Start a new experiment", use_container_width=True, help="Clears trained results from this browser session but does not delete your uploaded source file."):
        clear_experiment()
        st.rerun()

if source == "Built-in energy demo":
    raw = make_sample()
    read_meta = None
elif uploaded is not None:
    try:
        raw, read_meta = read_table(uploaded, return_metadata=True)
    except Exception as exc:
        st.error(f"I couldn't read this file. {exc}")
        st.stop()
else:
    st.info("Upload a dataset to continue. Minimum format: one timestamp column and one numeric target column.")
    st.stop()

if len(raw) > 250_000:
    st.warning("This public-demo configuration uses the most recent 250,000 rows to protect shared memory. The original file is not modified.")
    raw = raw.tail(250_000).copy()

default_ts = infer_timestamp_column(raw)
default_target = infer_target_column(raw, default_ts)
if default_ts is None or default_target is None:
    st.error("ForecastOS could not safely identify both a timestamp and numeric target. Check that the dataset contains both, then upload again.")
    st.stop()

with st.sidebar:
    st.divider()
    st.subheader("What should be forecast?")
    columns = list(raw.columns)
    ts_col = st.selectbox("Time column", columns, index=columns.index(default_ts), help="A timestamp identifies when each observation occurred.")
    numeric_candidates = []
    for c in columns:
        if c == ts_col:
            continue
        converted = pd.to_numeric(raw[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
        if converted.notna().mean() >= 0.8:
            numeric_candidates.append(c)
    if not numeric_candidates:
        st.error("No reliably numeric target column was found.")
        st.stop()
    target_default_idx = numeric_candidates.index(default_target) if default_target in numeric_candidates else 0
    target_col = st.selectbox("Target to predict", numeric_candidates, index=target_default_idx)

    feature_scan = analyze_features(raw, ts_col, target_col)
    usable_features = feature_scan.loc[feature_scan["recommended_usable"], "feature"].tolist() if not feature_scan.empty else []
    suggested_features = feature_scan.loc[feature_scan["suggested_known_future"], "feature"].tolist() if not feature_scan.empty else []
    exog_cols = st.multiselect(
        "Features known at forecast time",
        usable_features,
        default=[c for c in suggested_features if c in usable_features],
        help="ForecastOS detects usable features automatically, but only future-known variables are safe as covariates. Suggested defaults are conservative; confirm what will genuinely be available at prediction time.",
    )

    with st.expander("Preprocessing controls", expanded=False):
        regularize_grid = st.checkbox(
            "Repair manageable timestamp gaps",
            value=True,
            help="ForecastOS can rebuild a regular time grid when the inferred interval is reliable and the expansion is bounded. Missing target values are filled causally from prior observations.",
        )
        scaler_choice = st.selectbox(
            "Normalization / standardization",
            ["Auto", "Standard", "Robust", "Min-Max"],
            index=0,
            help="Auto selects among z-score StandardScaler, median/IQR RobustScaler, and [0,1] MinMaxScaler. Scalers are fitted inside each training fold to reduce leakage.",
        )

profile_raw = profile_timeseries(raw, ts_col, target_col)
work, prep_warnings, prep_report = prepare_frame(raw, ts_col, target_col, exog_cols, regularize=regularize_grid, return_report=True)
profile = profile_timeseries(work, ts_col, target_col)
auto_scaler, scaler_diag = recommend_scaler(work, [target_col] + prep_report.model_features)
scaler_kind = auto_scaler if scaler_choice == "Auto" else {"Standard": "standard", "Robust": "robust", "Min-Max": "minmax"}[scaler_choice]

max_h = max(1, min(336, len(work) // 5))
default_h = min(max_h, 24 if profile.median_step_seconds <= 3600 * 1.2 else 7)

with st.sidebar:
    horizon = st.number_input(
        "Forecast horizon (steps)",
        min_value=1,
        max_value=max_h,
        value=int(default_h),
        step=1,
        help=f"One step is approximately {profile.inferred_frequency}. ForecastOS limits the horizon to protect backtest quality.",
    )
    st.divider()
    st.subheader("Model effort")
    goal = st.radio(
        "Optimization goal",
        ["Balanced", "Fast exploration", "Maximum accuracy"],
        index=0,
        help="Balanced is recommended for most interactive use. Maximum accuracy uses more backtest folds and model training.",
    )
    mode = {"Balanced": "Balanced", "Fast exploration": "Fast", "Maximum accuracy": "Maximum accuracy"}[goal]

    with st.expander("Deep learning lab", expanded=False):
        st.caption("Deep learning is optional. ForecastOS still keeps naïve and classical baselines in the tournament.")
        deep_enabled = st.checkbox("Include deep-learning models", value=False)
        available_deep = ["Deep MLP AR"]
        torch_ok = torch_available()
        if torch_ok:
            available_deep += ["LSTM", "TCN", "Transformer"]
        else:
            st.info("LSTM, TCN and Transformer require the optional PyTorch dependencies in `requirements-deep.txt`.")
        default_deep = ["Deep MLP AR", "LSTM"] if torch_ok else ["Deep MLP AR"]
        deep_models = st.multiselect(
            "Models",
            available_deep,
            default=default_deep if deep_enabled else [],
            disabled=not deep_enabled,
            help="Deep MLP is CPU-native in the standard install. LSTM, TCN and Transformer are optional PyTorch sequence models.",
        )
        st.caption("Neural models are selected only if they beat simpler alternatives on rolling backtests.")
    if not deep_enabled:
        deep_models = []

    burden, burden_text = compute_estimate(len(work), mode, deep_models)
    st.caption(f"Compute burden: **{burden}** · {burden_text}")
    run = st.button("Run model tournament", type="primary", use_container_width=True)

# ---------- Data readiness ----------
st.markdown("## Data readiness")
read = readiness(profile, int(horizon), len(work))
rcols = st.columns([1.25, 1, 1, 1, 1])
with rcols[0]:
    note(f"<strong>{read.label}</strong><br><span class='small-muted'>{read.summary}</span>", read.level)
rcols[1].metric("Usable rows", f"{len(work):,}")
rcols[2].metric("Frequency", profile.inferred_frequency)
rcols[3].metric("Regular timestamps", f"{profile.regularity_score*100:.0f}%", help="Share of time gaps close to the median interval.")
rcols[4].metric("Drift indicator", f"{profile.drift_score*100:.0f}/100", help="A simple distribution-shift indicator comparing earlier and recent target levels.")

warnings = list(dict.fromkeys(profile_raw.warnings + prep_warnings + read.issues))
if warnings:
    with st.expander(f"Review {len(warnings)} data note{'s' if len(warnings) != 1 else ''}", expanded=read.level == "caution"):
        for w in warnings:
            st.warning(w)
else:
    st.success("No major data-quality warning was detected by the prototype checks.")

with st.expander("Automatic preprocessing & feature audit", expanded=False):
    if read_meta is not None:
        h = "No header detected" if read_meta.detected_header_row is None else f"Row {read_meta.detected_header_row + 1}"
        st.write(f"**Header detection:** {h}" + (f" · delimiter `{read_meta.delimiter}`" if read_meta.delimiter else ""))
        for msg in read_meta.notes:
            st.caption(msg)
    pc = st.columns(5)
    pc[0].metric("Rows in", f"{prep_report.original_rows:,}")
    pc[1].metric("Rows ready", f"{prep_report.final_rows:,}")
    pc[2].metric("Duplicate times", f"{prep_report.duplicate_timestamps_aggregated:,}")
    pc[3].metric("Inserted gaps", f"{prep_report.inserted_time_rows:,}")
    pc[4].metric("Missing after prep", f"{prep_report.target_missing_after + prep_report.feature_missing_after:,}")
    st.write(f"**Scaling:** {SCALER_LABELS[scaler_kind]}" + (" · automatically selected" if scaler_choice == "Auto" else " · user selected"))
    if scaler_choice == "Auto":
        st.caption(str(scaler_diag.get("reason", "")))
    if not prep_report.feature_table.empty:
        st.dataframe(
            prep_report.feature_table.style.format({"missing_%": "{:.1f}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Feature detection identifies candidates, constants, sparse columns, IDs and possible leakage. A detected feature is not automatically assumed to be known in the future.")

preview = work[[ts_col, target_col] + prep_report.model_features].tail(min(5000, len(work)))
fig = go.Figure()
fig.add_trace(go.Scatter(x=preview[ts_col], y=preview[target_col], name=target_col, mode="lines", line=dict(width=1.7)))
fig.update_layout(
    height=315,
    margin=dict(l=8, r=8, t=12, b=8),
    hovermode="x unified",
    xaxis_title="Time",
    yaxis_title=target_col,
    legend_title_text="",
)
st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
with st.expander("See detected time-series fingerprint"):
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Seasonal period", str(profile.selected_seasonal_period))
    f2.metric("Lag-1 correlation", f"{profile.autocorr_lag1:.2f}")
    f3.metric("Trend strength", f"{profile.trend_strength*100:.0f}/100")
    f4.metric("Missing target", f"{profile.missing_target_pct:.1f}%")
    st.caption("These diagnostics guide interpretation; the model winner is still determined by temporal backtesting.")

if run:
    status = st.status("Running fair temporal backtests…", expanded=True)
    progress = st.progress(0, text="Preparing model tournament")

    def progress_callback(done: int, total: int, model_name: str, state: str) -> None:
        pct = 0 if total <= 0 else min(100, int(100 * done / total))
        human = model_plain_name(model_name)
        if state == "running":
            progress.progress(pct, text=f"Testing {human} ({model_name})")
            status.write(f"Testing **{human}** · `{model_name}`")
        elif state == "failed":
            status.write(f"⚠️ {human} could not complete and will not be considered for selection.")
        else:
            progress.progress(pct, text=f"Completed {done} of {total} models")

    try:
        result = train_forecast(
            work,
            ts_col,
            target_col,
            exog_cols,
            profile,
            int(horizon),
            mode,
            deep_model_names=deep_models,
            progress_callback=progress_callback,
            scaler_kind=scaler_kind,
        )
        progress.progress(100, text="Model tournament complete")
        status.update(label=f"Complete · {model_plain_name(result.model_name)} selected by rolling backtests", state="complete", expanded=False)
    except Exception as exc:
        status.update(label="Forecasting could not complete", state="error", expanded=True)
        st.error(f"ForecastOS could not complete this experiment: {exc}")
        st.stop()
    st.session_state["forecast_result"] = result
    st.session_state["forecast_profile"] = profile
    st.session_state["forecast_work"] = work
    st.session_state["forecast_config"] = {
        "ts": ts_col,
        "target": target_col,
        "exog": exog_cols,
        "horizon": int(horizon),
        "mode": mode,
        "deep_models": deep_models,
        "scaler": scaler_kind,
        "regularize": regularize_grid,
    }
    st.session_state.pop("stress", None)
    st.session_state.pop("scenario", None)

result = st.session_state.get("forecast_result")
if result is None:
    st.info("When the setup looks right, choose **Run model tournament**. ForecastOS does not train anything until you explicitly start it.")
    st.stop()

cfg = st.session_state["forecast_config"]
current_cfg = {"ts": ts_col, "target": target_col, "exog": exog_cols, "horizon": int(horizon), "mode": mode, "deep_models": deep_models, "scaler": scaler_kind, "regularize": regularize_grid}
if cfg != current_cfg:
    st.warning("The setup has changed since these results were trained. The results below are preserved, but rerun the tournament before making a decision from the new setup.")

# ---------- Decision summary ----------
st.markdown("## Forecast decision summary")
trust_word, trust_help = trust_label(result.trust_score)
skill = result.metrics.get("Skill vs baseline (%)", float("nan"))
summary_cols = st.columns(4)
summary_cols[0].metric("Selected model", model_plain_name(result.model_name), help=f"Technical model: {result.model_name}")
summary_cols[1].metric("Forecast Trust", f"{result.trust_score:.0f}/100 · {trust_word}", help=trust_help)
summary_cols[2].metric("Skill vs baseline", "—" if not np.isfinite(skill) else f"{skill:+.1f}%", help="Positive means lower RMSE than the seasonal-naïve benchmark.")
summary_cols[3].metric("Backtest RMSE", f"{result.metrics['RMSE']:.3f}")

if skill < 0:
    note("<strong>Decision caution:</strong> the selected model did not improve on the seasonal baseline. Investigate the data, horizon or model setup before relying on this forecast.", "caution")
elif result.trust_score < 70:
    note(f"<strong>{trust_word} trust:</strong> use this forecast as supporting evidence and inspect the weak trust components before acting.", "review")
else:
    note(f"<strong>{trust_word} trust:</strong> backtests support using the forecast for planning, while the empirical interval and domain judgment remain important.", "ready")

failures = result.diagnostics.get("model_failures", {})
if failures:
    with st.expander(f"{len(failures)} model{'s' if len(failures) != 1 else ''} could not complete"):
        for name, why in failures.items():
            st.write(f"**{model_plain_name(name)} ({name})** — {why}")
        st.caption("A failed model is excluded from winner selection; other models remain comparable within the completed tournament.")

# Progressive disclosure: decision-first overview, technical detail in focused tabs.
overview_tab, compare_tab, explain_tab, stress_tab, analyst_tab, export_tab = st.tabs([
    "Forecast",
    "Model comparison",
    "Explainability",
    "Stress & scenarios",
    "AI analyst",
    "Export & details",
])

with overview_tab:
    st.subheader("Forecast and uncertainty")
    hist = work[[ts_col, target_col]].tail(max(100, 4 * int(horizon)))
    f = result.forecast
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=hist[ts_col], y=hist[target_col], mode="lines", name="Observed history"))
    fig2.add_trace(go.Scatter(x=f["timestamp"], y=f["upper_90"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig2.add_trace(go.Scatter(x=f["timestamp"], y=f["lower_90"], mode="lines", fill="tonexty", line=dict(width=0), name="90% empirical interval"))
    fig2.add_trace(go.Scatter(x=f["timestamp"], y=f["forecast"], mode="lines+markers", name="Forecast", line=dict(width=2.6)))
    fig2.update_layout(height=440, margin=dict(l=8, r=8, t=12, b=8), hovermode="x unified", xaxis_title="Time", yaxis_title=target_col, legend_title_text="")
    st.plotly_chart(fig2, use_container_width=True, config={"displaylogo": False})
    st.caption("The shaded band is built from 90th-percentile absolute errors observed in rolling backtests. It is an empirical uncertainty aid, not a guaranteed coverage probability.")

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("MAE", f"{result.metrics['MAE']:.3f}", help="Average absolute forecast error in target units.")
    b2.metric("sMAPE", f"{result.metrics['sMAPE']:.2f}%", help="Symmetric percentage error; safer than MAPE around small values.")
    b3.metric("MASE", f"{result.metrics['MASE']:.3f}", help="Error scaled by a seasonal-naïve benchmark. Below 1 is better than that benchmark.")
    b4.metric("Horizon", f"{len(result.forecast)} steps")
    with st.expander("See complete validation metrics"):
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Median AE", f"{result.metrics['MdAE']:.3f}")
        mc2.metric("WAPE", f"{result.metrics['WAPE']:.2f}%")
        mc3.metric("MAPE", "—" if not np.isfinite(result.metrics['MAPE']) else f"{result.metrics['MAPE']:.2f}%")
        mc4.metric("NRMSE / std", f"{result.metrics['NRMSE(std)']:.3f}")
        mc5, mc6, mc7 = st.columns(3)
        mc5.metric("R²", "—" if not np.isfinite(result.metrics['R²']) else f"{result.metrics['R²']:.3f}")
        mc6.metric("Forecast bias", f"{result.metrics['Bias']:+.3f}", help="Actual minus prediction. Positive means under-forecasting on average.")
        mc7.metric("Direction accuracy", "—" if not np.isfinite(result.metrics['Directional accuracy (%)']) else f"{result.metrics['Directional accuracy (%)']:.1f}%")
        st.caption("ForecastOS reports multiple complementary metrics because no single error measure is reliable for every scale, zero pattern or business objective.")

    st.markdown("**What to check before acting**")
    checks = []
    if profile.drift_score >= 0.6:
        checks.append("High drift: recent behavior differs from earlier training history.")
    if result.trust_score < 70:
        checks.append("Trust is below 70: inspect the trust decomposition and backtest folds.")
    if not np.isfinite(skill) or skill <= 0:
        checks.append("No positive skill over the seasonal baseline was demonstrated.")
    if not checks:
        checks.append("No major prototype warning is active. Still compare the interval with the cost of being wrong in your domain.")
    for x in checks:
        st.write(f"• {x}")

with compare_tab:
    st.subheader("Why this model won")
    board = result.leaderboard.copy()
    board.insert(1, "Plain-language model", board["Model"].map(model_plain_name))
    st.dataframe(
        board.style.format({"MAE": "{:.3f}", "MdAE": "{:.3f}", "RMSE": "{:.3f}", "NRMSE(std)": "{:.3f}", "MAPE": "{:.2f}", "sMAPE": "{:.2f}", "WAPE": "{:.2f}", "MASE": "{:.3f}", "R²": "{:.3f}", "Bias": "{:+.3f}", "Directional accuracy (%)": "{:.1f}", "Skill vs baseline (%)": "{:+.2f}", "Backtest seconds": "{:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Winner selection prioritizes the lowest rolling-backtest RMSE, using computation time only as a tie-breaker. Complexity itself receives no bonus.")

    st.markdown("### Trust decomposition")
    trust_df = pd.DataFrame({"Component": list(result.trust_components.keys()), "Score": list(result.trust_components.values())}).sort_values("Score")
    st.bar_chart(trust_df.set_index("Component"), horizontal=True)
    st.dataframe(trust_df.style.format({"Score": "{:.1f}"}), use_container_width=True, hide_index=True)
    st.caption("Forecast Trust is a heuristic decision aid built from baseline improvement, backtest stability, data quality, history adequacy and drift. It is not a probability that a forecast is correct.")

    with st.expander("Backtest fold details"):
        st.dataframe(result.diagnostics["backtest_folds"].style.format({"MAE": "{:.3f}", "MdAE": "{:.3f}", "RMSE": "{:.3f}", "NRMSE(std)": "{:.3f}", "MAPE": "{:.2f}", "sMAPE": "{:.2f}", "WAPE": "{:.2f}", "MASE": "{:.3f}", "R²": "{:.3f}", "Bias": "{:+.3f}", "Directional accuracy (%)": "{:.1f}"}), use_container_width=True, hide_index=True)
        st.caption("Large differences between folds can indicate regime sensitivity or unstable generalization.")

with explain_tab:
    st.subheader("What influenced the selected model?")
    if result.feature_importance.empty:
        st.info("This winner does not expose a feature-based explanation in the current prototype. Its selection is still supported by rolling backtests.")
    else:
        imp = result.feature_importance.head(12).sort_values("importance")
        st.bar_chart(imp.set_index("feature"), horizontal=True)
        top_names = result.feature_importance.head(3)["feature"].tolist()
        st.write("The strongest measured predictive sensitivities were **" + ", ".join(top_names) + "**.")
        if result.model_name in {"LSTM", "TCN", "Transformer"}:
            st.caption("For the selected sequence model, importance is channel-level permutation sensitivity on held-out training-tail examples.")
        else:
            st.caption("For supervised autoregressive models, importance is permutation sensitivity over fitted predictors.")
        note("<strong>Interpretation boundary:</strong> predictive importance is not causation. A high score means the model relied on that signal; it does not prove changing the variable will cause the target to change.")

    st.markdown("### Explanation vocabulary")
    with st.expander("Recognize common feature names"):
        st.write("• `lag_24` means the target value 24 steps earlier.")
        st.write("• `hour_sin/hour_cos` encode time-of-day cyclically.")
        st.write("• `dow_sin/dow_cos` encode day-of-week cyclically.")
        st.write("• `exog__temperature` means the selected known-at-forecast-time temperature feature.")

with stress_tab:
    st.subheader("Challenge the forecast before trusting it")
    st.write("Stress tests perturb recent context and measure how much the winning forecast changes. Lower sensitivity is generally more stable.")
    if st.button("Run stress tests", key="stress_button"):
        with st.status("Running controlled perturbations…", expanded=False) as stress_status:
            ss = stress_test(result, work, ts_col, target_col, exog_cols, profile)
            st.session_state["stress"] = ss
            stress_status.update(label="Stress tests complete", state="complete")
    if "stress" in st.session_state:
        ss = st.session_state["stress"]
        st.dataframe(ss.style.format({"Forecast sensitivity (%)": "{:.2f}", "Stability score": "{:.1f}"}), use_container_width=True, hide_index=True)
        st.caption("These are controlled sensitivity tests, not estimates of future accuracy under real-world shocks.")

    st.divider()
    st.subheader("Scenario lab")
    scenario_candidates = [c for c in prep_report.numeric_features if c in exog_cols]
    if scenario_candidates:
        sc1, sc2 = st.columns(2)
        scenario_col = sc1.selectbox("Change a known numeric feature", scenario_candidates)
        pct = sc2.slider("Scenario change (%)", -50, 50, 10, 1)
        st.caption("This asks how the fitted model responds when the selected future feature is shifted. It is not a causal intervention estimate.")
        if st.button("Compare scenario with baseline", key="scenario_button"):
            try:
                scenario = scenario_forecast(result, work, ts_col, exog_cols, scenario_col, pct)
                st.session_state["scenario"] = scenario
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
        st.info("To use percentage scenarios, select at least one numeric feature that is genuinely known at forecast time, then rerun the model tournament.")

with analyst_tab:
    st.subheader("AI forecast scientist")
    st.markdown(deterministic_brief(result, profile))
    st.caption("The AI analyst receives structured experiment evidence rather than the raw uploaded dataset.")

    starter = st.selectbox(
        "Start with a question",
        [
            "Choose a question…",
            "Why did ForecastOS select this model?",
            "What is the biggest reliability risk?",
            "Which signals mattered most?",
            "Is this forecast strong enough to support a decision?",
            "What experiment should I run next?",
        ],
        help="Example questions reduce the need to remember forecasting terminology. You can also type your own question below.",
    )
    custom = st.text_input("Or ask your own question", placeholder="Example: Why does confidence weaken at longer horizons?")
    question = custom.strip() or (starter if starter != "Choose a question…" else "")
    api_ready = bool(os.getenv("OPENAI_API_KEY"))
    if question and st.button("Answer from experiment evidence", type="primary", key="ask_button"):
        if api_ready:
            with st.status("Grounding the answer in experiment evidence…", expanded=False):
                try:
                    answer = ask_openai(question, result, profile)
                    st.write(answer)
                except Exception as exc:
                    st.warning(f"The AI analyst is unavailable: {exc}")
        else:
            st.info("AI analysis is optional and currently disabled. Add `OPENAI_API_KEY` to Streamlit Secrets. Numerical forecasting does not require an API key.")

with export_tab:
    st.subheader("Export forecast")
    export = result.forecast.copy()
    st.download_button(
        "Download forecast as CSV",
        export.to_csv(index=False).encode("utf-8"),
        file_name="forecastos_forecast.csv",
        mime="text/csv",
        use_container_width=False,
    )
    st.caption("Export includes timestamp, point forecast, and empirical 90% lower/upper bounds.")

    st.markdown("### Reproducibility details")
    st.json({
        "selected_model": result.model_name,
        "plain_language_model": model_plain_name(result.model_name),
        "target": target_col,
        "frequency": profile.inferred_frequency,
        "seasonal_period": profile.selected_seasonal_period,
        "known_at_forecast_time_features": exog_cols,
        "forecast_horizon_steps": int(horizon),
        "search_mode": mode,
        "deep_models_requested": deep_models,
        "scaler": SCALER_LABELS.get(scaler_kind, scaler_kind),
        "timestamp_regularization": regularize_grid,
        "preprocessing": {"duplicates_aggregated": prep_report.duplicate_timestamps_aggregated, "inserted_rows": prep_report.inserted_time_rows, "missing_target_before": prep_report.target_missing_before, "missing_target_after": prep_report.target_missing_after},
        "models_attempted": result.diagnostics.get("models_attempted", []),
        "metrics": result.metrics,
        "trust_score": result.trust_score,
    })
    with st.expander("Important limitations"):
        st.write("• One numeric target is forecast per experiment in this release.")
        st.write("• Final future covariates default to their latest observed value unless changed in the scenario lab.")
        st.write("• Empirical intervals use rolling-backtest residuals and are not formal coverage guarantees.")
        st.write("• Feature importance and scenarios describe model behavior, not causal effects.")
        st.write("• Deep models can overfit short series; they remain subject to the same baseline comparison as simpler models.")
