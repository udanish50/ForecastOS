# ForecastOS

**Autonomous Forecast Scientist** — a professional Streamlit application for trustworthy time-series forecasting.

> Upload time-series data → diagnose it → benchmark statistical, machine-learning and deep-learning models → quantify trust → explain the winner → stress-test it → explore scenarios → export the forecast.

ForecastOS is intentionally **decision-first**. It does not assume a neural model is best, and it does not use an LLM to generate the numerical forecast. Every candidate model must compete against simple temporal baselines using rolling backtests.

## v0.4 explicit windowing + future-feature control

ForecastOS now uses an explicit forecast-configuration workflow. The user selects the timestamp, target, previous-history window, future horizon and future-known features. Automatic timestamp/target inference is advisory only and never silently commits the experiment.

The app creates sliding windows internally, requires explicit future weather/exogenous values when such features are selected, aligns them by timestamp or deliberate row order, and supports Standard, Robust and Min-Max scaling. It will not silently carry the last observed weather value into the future. See [`WINDOWING.md`](WINDOWING.md) and [`PREPROCESSING.md`](PREPROCESSING.md).

## What is included

### Data intelligence

- Explicit timestamp and target mapping; automatic inference is advisory only
- Frequency and regularity detection
- Missing-target and duplicate-timestamp checks
- Seasonality candidates and selected seasonal period
- Lag-1 autocorrelation
- Trend-strength indicator
- Drift indicator
- Explicit previous-step history window and future forecast horizon
- Explicit future-known covariate selection with required future data
- Data-readiness state before training

### Forecast model tournament

Classical/statistical baselines:

- Last-value naïve
- Seasonal naïve
- Drift/trend continuation

Machine learning:

- Ridge autoregression
- Histogram Gradient Boosting autoregression

Deep learning:

- **Deep MLP autoregression**
- **LSTM** sequence model
- **TCN** (Temporal Convolutional Network)
- **Transformer** sequence encoder

Deep learning is opt-in from the UI. Complex models receive no selection advantage; the winner is chosen from temporal backtests.

### Evaluation and reliability

- Rolling temporal backtesting
- MAE
- RMSE
- sMAPE
- MASE
- Forecast skill versus seasonal naïve
- Empirical 90% uncertainty intervals
- Forecast Trust Score
- Trust decomposition
- Per-fold backtest diagnostics
- Explicit reporting of model failures

### Explainability and decision support

- Permutation sensitivity for supervised autoregressive models
- Channel-level permutation sensitivity for LSTM/TCN/Transformer models
- Plain-language explanation of common lag/time features
- Controlled forecast stress tests
- Known-future covariate scenario lab
- Evidence-grounded optional AI Forecast Scientist
- Forecast CSV export and reproducibility metadata

## Human-factors design

The interface is built around:

- decision-first information hierarchy
- progressive disclosure
- recognition rather than recall
- visible system status
- explicit error prevention
- automation-bias reduction
- calibrated trust language
- keyboard-visible focus
- larger interaction targets
- reduced-motion support
- plain-language labels beside technical model names

See [`HUMAN_FACTORS.md`](HUMAN_FACTORS.md) for the full design specification.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The standard requirements file includes PyTorch so the full Deep Learning Lab is available when deployed.

### Lightweight installation

If you want the smallest local environment and only need classical models plus the CPU-native Deep MLP:

```bash
pip install -r requirements-lite.txt
streamlit run streamlit_app.py
```

LSTM, TCN and Transformer will be hidden automatically when PyTorch is unavailable.

## Tests

```bash
pip install pytest
pytest -q
```

The test suite covers:

- column inference and data profiling
- metrics
- end-to-end classical forecasting
- exogenous-variable path
- Deep MLP tournament integration
- LSTM, TCN and Transformer smoke tests when PyTorch is installed

## Deploy to Streamlit Community Cloud

1. Push the repository to GitHub.
2. Create a new Streamlit Community Cloud app.
3. Choose your repository and `main` branch.
4. Set the entrypoint to `streamlit_app.py`.
5. Deploy with the root `requirements.txt`.

The deep sequence models are deliberately small, CPU-capable architectures and train only when a user explicitly enables them. For high traffic, large datasets, foundation models, or repeated maximum-accuracy runs, move heavy inference to a dedicated worker/GPU service and keep Streamlit as the interaction layer.

## Optional AI analyst

Add this to Streamlit Secrets:

```toml
OPENAI_API_KEY="..."
```

The LLM receives structured experiment evidence and the user's question. The raw uploaded dataframe is not sent by the current analyst implementation.

## Data format

Minimum:

| timestamp | target |
|---|---:|
| 2026-01-01 00:00 | 100.2 |
| 2026-01-01 01:00 | 104.7 |

Optional covariates may be selected only when their future values are genuinely available at forecast time. ForecastOS generates a future-feature template for those variables.

## Important interpretation boundaries

- Forecast Trust is a heuristic decision aid, not a probability of correctness.
- The 90% band is an empirical residual-based interval, not a formal coverage guarantee.
- Feature importance measures predictive sensitivity, not causality.
- Scenario analysis describes model response, not causal intervention effects.
- Selected future covariates must be explicitly supplied for every forecast step; ForecastOS does not invent them.
- Deep learning can overfit short series; it is never exempt from baseline comparison.

## Foundation-model roadmap

The repository keeps foundation-model deployment separate from the interactive CPU model tournament. Future adapters can support:

- Amazon Chronos
- Google TimesFM
- Salesforce Moirai / Uni2TS

For production, these should generally run behind a dedicated inference service instead of downloading multi-GB checkpoints during a Streamlit cold start.

## Research roadmap

1. Dataset fingerprint → learned model router across public forecasting benchmarks.
2. Dynamic ensembles conditioned on series characteristics and forecast horizon.
3. Feature × lag × horizon explanations.
4. Formal conformal interval calibration and coverage diagnostics.
5. Forecast autopsy after actual outcomes arrive.
6. Cost-sensitive model selection under asymmetric decision losses.
7. Scenario-conditioned forecasting with explicit future trajectories.
8. Foundation-model adapters and domain-specific LoRA/adapters.
9. Human-subject evaluation of trust calibration and explanation comprehension.

## Project structure

```text
forecastos/
├── .github/workflows/ci.yml
├── .streamlit/config.toml
├── data/
├── src/forecastos/
│   ├── analyst.py
│   ├── data.py
│   ├── deep_models.py
│   ├── engine.py
│   ├── features.py
│   ├── human_factors.py
│   ├── metrics.py
│   ├── models.py
│   ├── profile.py
│   ├── sample.py
│   └── types.py
├── tests/test_core.py
├── streamlit_app.py
├── HUMAN_FACTORS.md
├── WINDOWING.md
├── ARCHITECTURE.md
├── DEPLOYMENT.md
├── TEST_REPORT.md
├── requirements.txt
├── requirements-lite.txt
├── requirements-deep.txt
└── requirements-foundation.txt
```

## License

MIT
