# ForecastOS preprocessing pipeline

ForecastOS v0.3 treats preprocessing as part of the forecasting evidence chain rather than a hidden cleanup step.

## Automatic ingestion

- CSV delimiter detection for comma, semicolon, tab, and pipe-like exports.
- Header-row inference across the first rows of CSV and Excel files, including files with title/metadata lines above the table.
- Duplicate column names are made unique and blank/`Unnamed` columns receive stable names.
- Completely empty rows and columns are removed.
- Numeric-looking strings are considered when detecting targets.

## Automatic schema and feature audit

Every non-target column is classified as numeric, categorical/boolean, high-cardinality text, constant, sparse, identifier-like, or possible leakage. ForecastOS separately marks conservative candidates that *may* be known at forecast time (for example schedule/holiday/promotion columns), but the user remains responsible for confirming future availability.

This distinction is intentional: a variable can be a useful historical feature and still be invalid as a future covariate.

## Missing values and time-grid repair

- Invalid timestamps are removed.
- Duplicate timestamps are aggregated.
- Manageable gaps in an otherwise regular time grid can be reconstructed automatically.
- Missing targets are filled **causally** using prior values; leading target gaps are removed instead of interpolating from future targets.
- Numeric future-known features use causal forward fill plus a compact leading fallback.
- Categorical future-known features use forward fill and stable ordinal encoding.
- The preprocessing audit reports rows inserted, duplicates aggregated, missing values before/after, and encoding actions.

The default avoids bidirectional target interpolation because using a later target to fill an earlier missing target can contaminate temporal validation.

## Scaling

ForecastOS exposes three fitted-inside-the-training-fold alternatives:

1. **Standard / z-score** — mean and standard deviation.
2. **Robust** — median and interquartile range; preferred when outliers/skew are material.
3. **Min-Max** — maps observed training values to `[0, 1]`; useful for bounded signals.

`Auto` chooses a scaler from distribution diagnostics. Users can override it. Gradient-boosted trees do not require scaling, while Ridge, MLP, LSTM, TCN, and Transformer models use the configured scaler.

## Validation metrics

ForecastOS reports complementary metrics rather than relying on one score:

- MAE
- Median absolute error (MdAE)
- RMSE
- NRMSE normalized by target standard deviation
- MAPE (undefined near an all-zero target is shown as unavailable)
- sMAPE
- WAPE
- MASE
- R²
- signed forecast bias (`actual - prediction`)
- directional accuracy
- skill versus the seasonal-naïve baseline

The model tournament still selects by rolling-backtest RMSE in v0.3 so the decision rule remains explicit and reproducible.
