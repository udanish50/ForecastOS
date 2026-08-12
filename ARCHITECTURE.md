# ForecastOS Architecture

## Product architecture

```text
User dataset
    │
    ▼
Data-role inference + explicit user confirmation
    │
    ▼
Time-series fingerprint + readiness checks
    │
    ▼
Model tournament router
    ├── Naïve / seasonal / drift
    ├── Ridge / gradient boosting
    └── Deep MLP / LSTM / TCN / Transformer
    │
    ▼
Rolling temporal backtests
    │
    ├── model leaderboard
    ├── baseline skill
    ├── fold stability
    └── residual distribution
    │
    ▼
Winning model fit on complete history
    │
    ├── point forecast
    ├── empirical uncertainty band
    ├── trust decomposition
    └── predictive-sensitivity explanation
    │
    ▼
Decision layer
    ├── forecast summary
    ├── stress testing
    ├── scenario lab
    ├── evidence-grounded LLM analyst
    └── export / reproducibility packet
```

## Deep-learning architecture

The deep models intentionally remain compact enough for interactive CPU testing.

### Deep MLP AR

Lag, calendar and optional known-at-forecast-time covariates are converted to a supervised feature vector and passed through a three-hidden-layer MLP with standardization, Adam optimization and early stopping.

### LSTM

A rolling context window contains:

- target history
- cyclical calendar features
- optional known-at-forecast-time numeric covariates

A two-layer LSTM encodes the context, followed by layer normalization and a nonlinear regression head.

### TCN

The same context tensor is processed through stacked 1D convolutions with increasing dilation, then a compact nonlinear prediction head.

### Transformer

Input channels are projected into a compact embedding, combined with sinusoidal positional encoding, passed through a two-layer Transformer encoder, normalized, and mapped to the next-step target.

All sequence models forecast recursively and are trained with standardized inputs/target, AdamW, SmoothL1 loss, gradient clipping and validation-tail early stopping.

## Explainability

- Ridge/HGB/MLP: permutation sensitivity on fitted supervised features.
- LSTM/TCN/Transformer: channel-level permutation sensitivity on held-out sequence examples.

These methods quantify predictive dependence and are never presented as causal effects.

## Selection policy

Every model is evaluated using rolling-origin temporal folds. ForecastOS selects the lowest aggregate RMSE among successful candidates and uses model runtime only as a tie-breaker. Model complexity is not a selection criterion.

## Human-factors layer

The UI is a separate decision-support layer over the forecasting engine. It provides:

- readiness before action
- explicit model effort/deep-learning controls
- visible training status
- failure disclosure
- decision summary before technical details
- progressive disclosure through focused result tabs
- plain-language model labels
- trust-calibration language
- explanation boundaries
- accessible focus and target sizing

See `HUMAN_FACTORS.md`.
