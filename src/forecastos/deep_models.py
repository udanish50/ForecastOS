from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import time_features
from .models import BaseForecaster
from .scaling import make_scaler


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _context_length(n: int, seasonal_period: int) -> int:
    preferred = 48
    if seasonal_period > 1:
        preferred = max(preferred, min(int(seasonal_period), 168))
    return int(max(12, min(preferred, 168, max(12, n // 5))))


def _row_features(target_value: float, ts: pd.Timestamp, exog: dict[str, float] | None) -> list[float]:
    tf = time_features(pd.Timestamp(ts))
    vals = [float(target_value)] + [float(tf[k]) for k in sorted(tf)]
    if exog:
        for k in sorted(exog):
            try:
                vals.append(float(exog[k]))
            except (TypeError, ValueError):
                vals.append(0.0)
    return vals


class TorchSequenceForecaster(BaseForecaster):
    supports_exog = True

    def __init__(self, kind: str, training_level: str = "balanced", scaler_kind: str = "standard"):
        self.kind = kind
        self.training_level = training_level.lower()
        self.scaler_kind = scaler_kind
        self.name = {"lstm": "LSTM", "tcn": "TCN", "transformer": "Transformer"}[kind]

    def fit(self, df, timestamp_col, target_col, exog_cols, seasonal_period):
        try:
            import torch
            import torch.nn as nn
        except Exception as exc:
            raise RuntimeError("PyTorch is not installed. Install requirements-deep.txt to enable LSTM, TCN and Transformer models.") from exc

        torch.manual_seed(42)
        np.random.seed(42)
        try:
            torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
        except Exception:
            pass

        self.timestamp_col = timestamp_col
        self.target_col = target_col
        self.exog_cols = sorted(list(exog_cols or []))
        self.context = _context_length(len(df), seasonal_period)
        self.history_y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=np.float32).tolist()
        self.history_ts = [pd.Timestamp(x) for x in df[timestamp_col]]
        self.history_exog = []
        for _, row in df.iterrows():
            self.history_exog.append({c: float(row[c]) if pd.notna(row[c]) else 0.0 for c in self.exog_cols})

        matrix = np.asarray([
            _row_features(y, ts, ex)
            for y, ts, ex in zip(self.history_y, self.history_ts, self.history_exog)
        ], dtype=np.float32)
        if len(matrix) <= self.context + 8:
            raise ValueError("Not enough observations for the selected deep sequence model.")

        xs, ys = [], []
        for i in range(self.context, len(matrix)):
            xs.append(matrix[i-self.context:i])
            ys.append(self.history_y[i])
        X = np.asarray(xs, dtype=np.float32)
        y = np.asarray(ys, dtype=np.float32)

        max_samples = {"fast": 2500, "balanced": 6000, "maximum accuracy": 10000}.get(self.training_level, 6000)
        if len(X) > max_samples:
            X, y = X[-max_samples:], y[-max_samples:]

        self.x_scaler = make_scaler(self.scaler_kind)
        flat = X.reshape(-1, X.shape[-1])
        flat_n = self.x_scaler.fit_transform(flat).astype(np.float32)
        Xn = flat_n.reshape(X.shape)
        self.y_scaler = make_scaler(self.scaler_kind)
        yn = self.y_scaler.fit_transform(y.reshape(-1, 1)).reshape(-1).astype(np.float32)

        in_dim = X.shape[-1]

        class LSTMNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = nn.LSTM(in_dim, 40, num_layers=2, batch_first=True, dropout=0.12)
                self.head = nn.Sequential(nn.LayerNorm(40), nn.Linear(40, 24), nn.GELU(), nn.Linear(24, 1))
            def forward(self, x):
                z, _ = self.rnn(x)
                return self.head(z[:, -1]).squeeze(-1)

        class TCNNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv1d(in_dim, 32, 5, padding=2), nn.GELU(),
                    nn.Conv1d(32, 32, 5, padding=4, dilation=2), nn.GELU(),
                    nn.Conv1d(32, 24, 3, padding=2, dilation=2), nn.GELU(),
                )
                self.head = nn.Sequential(nn.Linear(24, 24), nn.GELU(), nn.Linear(24, 1))
            def forward(self, x):
                z = self.net(x.transpose(1, 2))
                return self.head(z[:, :, -1]).squeeze(-1)

        context_len = self.context

        class TransformerNet(nn.Module):
            def __init__(self):
                super().__init__()
                d_model = 32
                self.proj = nn.Linear(in_dim, d_model)
                layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=96, dropout=0.10, batch_first=True, activation="gelu", norm_first=True)
                self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
                self.norm = nn.LayerNorm(d_model)
                self.head = nn.Linear(d_model, 1)
                pe = self._pe(context_len, d_model)
                self.register_buffer("pe", pe, persistent=False)
            @staticmethod
            def _pe(length, d_model):
                pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
                div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
                pe = torch.zeros(length, d_model)
                pe[:, 0::2] = torch.sin(pos * div)
                pe[:, 1::2] = torch.cos(pos * div)
                return pe.unsqueeze(0)
            def forward(self, x):
                z = self.proj(x) + self.pe[:, :x.shape[1]]
                z = self.encoder(z)
                return self.head(self.norm(z[:, -1])).squeeze(-1)

        self.model = {"lstm": LSTMNet, "tcn": TCNNet, "transformer": TransformerNet}[self.kind]()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1.8e-3, weight_decay=1e-4)
        loss_fn = nn.SmoothL1Loss()
        epochs = {"fast": 6, "balanced": 12, "maximum accuracy": 20}.get(self.training_level, 12)
        batch = min(128, max(16, len(Xn) // 12))

        split = max(8, int(len(Xn) * 0.85))
        if split >= len(Xn):
            split = len(Xn) - 4
        X_train = torch.from_numpy(Xn[:split])
        y_train = torch.from_numpy(yn[:split])
        X_val = torch.from_numpy(Xn[split:])
        y_val = torch.from_numpy(yn[split:])

        best_state, best_val, patience = None, float("inf"), 0
        self.model.train()
        for _ in range(epochs):
            order = torch.randperm(len(X_train))
            for start in range(0, len(order), batch):
                idx = order[start:start+batch]
                optimizer.zero_grad(set_to_none=True)
                pred = self.model(X_train[idx])
                loss = loss_fn(pred, y_train[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
            self.model.eval()
            with torch.no_grad():
                val = float(loss_fn(self.model(X_val), y_val).item()) if len(X_val) else 0.0
            self.model.train()
            if val < best_val - 1e-4:
                best_val = val
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 4:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        self.feature_names_ = ["target"] + sorted(time_features(pd.Timestamp("2026-01-01")).keys()) + [f"exog__{c}" for c in self.exog_cols]

        # Channel-level permutation sensitivity on the held-out tail. This is
        # predictive sensitivity, not a causal attribution.
        if len(X_val):
            with torch.no_grad():
                base_pred = self.y_scaler.inverse_transform(self.model(X_val).cpu().numpy().reshape(-1, 1)).reshape(-1)
            actual = self.y_scaler.inverse_transform(y_val.cpu().numpy().reshape(-1, 1)).reshape(-1)
            base_mae = float(np.mean(np.abs(actual - base_pred)))
            rng = np.random.default_rng(42)
            rows = []
            for j, fname in enumerate(self.feature_names_):
                pert = X_val.clone()
                order = torch.from_numpy(rng.permutation(len(pert))).long()
                pert[:, :, j] = pert[order, :, j]
                with torch.no_grad():
                    pp = self.y_scaler.inverse_transform(self.model(pert).cpu().numpy().reshape(-1, 1)).reshape(-1)
                delta = max(0.0, float(np.mean(np.abs(actual - pp))) - base_mae)
                rows.append({"feature": fname, "importance": delta})
            imp = pd.DataFrame(rows)
            total = float(imp["importance"].sum())
            if total > 0:
                imp["importance"] = 100.0 * imp["importance"] / total
            self.feature_importance_df_ = imp.sort_values("importance", ascending=False).reset_index(drop=True)
        else:
            self.feature_importance_df_ = pd.DataFrame(columns=["feature", "importance"])
        return self

    def predict(self, horizon, future_timestamps, future_exog=None):
        import torch

        ys = list(self.history_y)
        ts_hist = list(self.history_ts)
        ex_hist = list(self.history_exog)
        preds: list[float] = []
        for step in range(horizon):
            ex = {}
            for c in self.exog_cols:
                if future_exog is not None and c in future_exog.columns and step < len(future_exog):
                    val = future_exog.iloc[step][c]
                    ex[c] = float(val) if pd.notna(val) else 0.0
                else:
                    ex[c] = ex_hist[-1].get(c, 0.0) if ex_hist else 0.0
            rows = [_row_features(y, ts, e) for y, ts, e in zip(ys[-self.context:], ts_hist[-self.context:], ex_hist[-self.context:])]
            arr = np.asarray(rows, dtype=np.float32)
            if len(arr) < self.context:
                pad = np.repeat(arr[:1], self.context - len(arr), axis=0)
                arr = np.vstack([pad, arr])
            arr = self.x_scaler.transform(arr).astype(np.float32)
            x = torch.from_numpy(arr[None, :, :])
            with torch.no_grad():
                pred_n = float(self.model(x).item())
            pred = float(self.y_scaler.inverse_transform(np.array([[pred_n]], dtype=float))[0, 0])
            pred = float(np.nan_to_num(pred, nan=ys[-1], posinf=ys[-1], neginf=ys[-1]))
            preds.append(pred)
            ys.append(pred)
            ts_hist.append(pd.Timestamp(future_timestamps[step]))
            ex_hist.append(ex)
        return np.asarray(preds, dtype=float)


def build_torch_models(names: list[str], mode: str, scaler_kind: str = "standard") -> list[BaseForecaster]:
    keymap = {"LSTM": "lstm", "TCN": "tcn", "Transformer": "transformer"}
    return [TorchSequenceForecaster(keymap[n], mode, scaler_kind=scaler_kind) for n in names if n in keymap]
