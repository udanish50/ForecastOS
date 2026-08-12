# ForecastOS v0.3 test report

## Automated suite

`PYTHONPATH=src pytest -q`

**Result: 10 passed.**

Coverage includes:

- timestamp/target inference and time-series profiling;
- original metric sanity checks;
- end-to-end fast forecasting;
- exogenous covariate forecasting;
- Deep MLP tournament participation;
- optional LSTM/TCN/Transformer smoke tests when PyTorch is available;
- CSV header detection below metadata rows;
- automatic feature audit;
- causal missing-value preparation;
- categorical encoding;
- Standard, Robust, and Min-Max scaler availability;
- extended validation metrics in the final result/leaderboard.

## Dirty-data integration smoke test

A 28-day hourly synthetic energy series was deliberately modified with missing timestamps, duplicate timestamps, missing target observations, missing temperature, an added schedule feature, and a categorical site feature.

Observed preprocessing behavior:

- duplicate timestamps aggregated;
- missing values reduced to zero in the model-ready frame;
- categorical feature encoded;
- target and timestamp detected correctly;
- the experiment successfully trained under all three supported scaler modes.

Example fast-mode outcomes from this synthetic run:

| Scaler | Selected model | RMSE | WAPE | R² | Trust |
|---|---|---:|---:|---:|---:|
| Standard | Ridge AR | 2.9389 | 2.11% | 0.513 | 97.9 |
| Robust | Ridge AR | 3.2080 | 2.28% | 0.402 | 97.1 |
| Min-Max | Ridge AR | 3.3253 | 2.34% | 0.358 | 97.2 |

These numbers validate pipeline execution; they are not product-performance claims because the dataset is synthetic.

## Static validation

The Streamlit entrypoint and all package modules compile successfully with Python `py_compile`.
