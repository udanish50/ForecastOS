from __future__ import annotations

import numpy as np
import pandas as pd


def make_sample(n: int = 24 * 120, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-01-01", periods=n, freq="h")
    hour = ts.hour.to_numpy()
    dow = ts.dayofweek.to_numpy()
    temperature = 8 + 9 * np.sin(2 * np.pi * (hour - 7) / 24) + rng.normal(0, 1.5, n)
    humidity = 62 - 0.8 * temperature + rng.normal(0, 3, n)
    weekend = (dow >= 5).astype(float)
    load = 110 + 18 * np.sin(2 * np.pi * (hour - 15) / 24) - 9 * weekend + 1.4 * temperature + rng.normal(0, 3.5, n)
    # Smooth autoregressive component.
    for i in range(1, n):
        load[i] = 0.72 * load[i] + 0.28 * load[i - 1]
    return pd.DataFrame({"timestamp": ts, "load": load, "temperature": temperature, "humidity": humidity})
