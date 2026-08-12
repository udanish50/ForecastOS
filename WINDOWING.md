# ForecastOS windowing and future-feature policy

ForecastOS v0.4 makes the forecasting geometry explicit.

## Historical table

Keep the historical data in ordinary long form. Do **not** manually create lag columns or sliding-window tensors.

```text
timestamp,target,temperature,humidity
2026-01-01 00:00,100.2,3.1,71
2026-01-01 01:00,102.8,3.0,72
...
```

The user chooses `history_window = N`. ForecastOS then constructs the windows internally:

```text
[t-N, ..., t-2, t-1] -> predict t
```

For supervised autoregressive models, this exposes `lag_1 ... lag_N` for the target. For LSTM, TCN and Transformer, the same choice becomes the exact sequence length.

## Forecast horizon

The user separately chooses `horizon = H`:

```text
previous N observations -> forecast next H observations
```

The final forecast therefore covers `t+1 ... t+H`.

## Weather and other external variables

Only select an external feature if its future values will genuinely be available at forecast time. Examples include numerical weather prediction, holiday calendars, tariffs, planned promotions and schedules.

If future weather is selected, ForecastOS requires a separate future-feature table with one row per forecast step. It will not silently repeat the last observed weather value.

Recommended future-weather format:

```text
timestamp,temperature,humidity
2026-01-08 00:00,-3.1,77
2026-01-08 01:00,-3.4,79
...
```

The UI generates a downloadable template containing the exact timestamps required for the selected horizon.

## Alignment

The user explicitly chooses either:

- timestamp alignment (recommended), or
- row-order alignment.

Timestamp alignment requires all requested forecast timestamps to be present. Missing timestamps cause validation to fail rather than being guessed.

## Deep sequence models

LSTM, TCN and Transformer use two information blocks:

1. the exact historical sliding window selected by the user;
2. calendar/time information plus explicitly supplied future-known covariates for the step being predicted.

This prevents a future weather value for `t+h` from being shifted to the wrong forecast step.
