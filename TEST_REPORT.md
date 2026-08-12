# ForecastOS v0.4 test report

## Automated suite

```text
13 passed
```

Coverage includes:

- timestamp/target advisory inference
- messy CSV header detection
- causal missing-value preprocessing
- categorical feature encoding
- Standard / Robust / Min-Max scaling
- expanded metric suite
- end-to-end classical forecasting
- explicit external-feature forecasting
- Deep MLP tournament completion
- LSTM / TCN / Transformer smoke tests when PyTorch is installed
- exact user-selected history-window propagation
- explicit future-feature timestamp alignment
- prevention of silent future-covariate invention

## Integrated neural tournament smoke test

A synthetic hourly energy series was trained with an explicit 12-step history window, 4-step forecast horizon and explicit future temperature trajectory.

Models attempted:

```text
Naive
Seasonal Naive
Drift
Ridge AR
Deep MLP AR
LSTM
TCN
Transformer
```

Result:

```text
All model failures: none
History window in diagnostics: 12
Future temperature baseline preserved exactly
```

The final winner was selected only by rolling temporal backtest performance.

## UI validation limitation

The complete Streamlit source compiles successfully. The current execution sandbox does not have Streamlit installed, so browser rendering was not executed here. GitHub/Streamlit deployment installs the declared dependencies from `requirements.txt`.
