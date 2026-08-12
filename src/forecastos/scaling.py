from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

SCALER_LABELS = {
    "standard": "Standard (z-score)",
    "robust": "Robust (median/IQR)",
    "minmax": "Min-Max [0, 1]",
}


def make_scaler(kind: str):
    kind = (kind or "standard").lower().replace("-", "").replace("_", "")
    if kind in {"standard", "zscore", "z"}:
        return StandardScaler()
    if kind in {"robust", "iqr"}:
        return RobustScaler(quantile_range=(25.0, 75.0))
    if kind in {"minmax", "minmax01", "01"}:
        return MinMaxScaler(feature_range=(0.0, 1.0), clip=True)
    raise ValueError(f"Unknown scaler: {kind}")


def recommend_scaler(df: pd.DataFrame, columns: list[str]) -> tuple[str, dict[str, float | str]]:
    """Choose a scaler from distribution diagnostics without peeking at the future target.

    The recommendation is heuristic. Actual scaler fitting remains inside each training fold.
    """
    cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        return "standard", {"reason": "No numeric feature distribution was available; Standard scaling is the safe default."}

    outlier_rates, skews, bounded = [], [], []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) < 8:
            continue
        q1, q3 = np.quantile(s, [0.25, 0.75])
        iqr = float(q3 - q1)
        if iqr > 1e-12:
            outlier_rates.append(float(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean()))
        else:
            outlier_rates.append(0.0)
        sk = float(s.skew()) if len(s) >= 3 else 0.0
        skews.append(abs(sk) if np.isfinite(sk) else 0.0)
        bounded.append(bool(float(s.min()) >= 0.0 and float(s.max()) <= 1.0))

    outlier_rate = float(np.median(outlier_rates)) if outlier_rates else 0.0
    abs_skew = float(np.median(skews)) if skews else 0.0
    bounded_ratio = float(np.mean(bounded)) if bounded else 0.0

    if outlier_rate >= 0.025 or abs_skew >= 1.5:
        kind = "robust"
        reason = "Robust scaling selected because the numeric data shows material outliers or skewness."
    elif bounded_ratio >= 0.7:
        kind = "minmax"
        reason = "Min-Max scaling selected because most numeric signals are already naturally bounded near [0, 1]."
    else:
        kind = "standard"
        reason = "Standard scaling selected because numeric distributions are not strongly outlier-dominated or bounded."
    return kind, {
        "reason": reason,
        "median_outlier_rate": outlier_rate,
        "median_abs_skew": abs_skew,
        "bounded_feature_ratio": bounded_ratio,
    }
