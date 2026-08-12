from __future__ import annotations

import csv
from dataclasses import dataclass, field
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TIMESTAMP_HINTS = ("timestamp", "datetime", "date", "time", "ds")
TARGET_HINTS = ("target", "y", "value", "load", "demand", "sales", "ghi", "power", "price")
ID_HINTS = ("id", "index", "key", "uuid", "code")
LEAKAGE_HINTS = ("future", "lead", "next", "actual", "label", "ground_truth", "prediction", "forecast")
FUTURE_KNOWN_HINTS = ("holiday", "calendar", "tariff", "schedule", "promotion", "promo", "planned", "forecast_")


@dataclass
class ReadMetadata:
    detected_header_row: int | None = 0
    delimiter: str | None = None
    source_type: str = "unknown"
    notes: list[str] = field(default_factory=list)


@dataclass
class PreprocessingReport:
    original_rows: int
    final_rows: int
    removed_invalid_timestamps: int
    duplicate_timestamps_aggregated: int
    target_missing_before: int
    target_missing_after: int
    feature_missing_before: int
    feature_missing_after: int
    regularized: bool
    inserted_time_rows: int
    feature_table: pd.DataFrame
    numeric_features: list[str]
    categorical_features: list[str]
    ignored_features: list[str]
    suggested_future_features: list[str]
    model_features: list[str]
    categorical_encoding: dict[str, dict[str, int]] = field(default_factory=dict)


def _as_bytes(source) -> tuple[bytes | None, str]:
    name = getattr(source, "name", str(source))
    if isinstance(source, (str, Path)):
        return None, str(source)
    if hasattr(source, "getvalue"):
        value = source.getvalue()
        return value if isinstance(value, bytes) else bytes(value), name
    if hasattr(source, "read"):
        try:
            pos = source.tell()
        except Exception:
            pos = None
        value = source.read()
        if pos is not None:
            try:
                source.seek(pos)
            except Exception:
                pass
        if isinstance(value, str):
            value = value.encode("utf-8")
        return value, name
    return None, name


def _looks_number(value: Any) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    try:
        float(str(value).strip().replace(",", ""))
        return True
    except Exception:
        return False


def _looks_datetime(value: Any) -> bool:
    if value is None or str(value).strip() == "":
        return False
    text = str(value).strip()
    if len(text) < 6:
        return False
    try:
        parsed = pd.to_datetime([text], errors="coerce")
        return bool(parsed.notna()[0])
    except Exception:
        return False


def _detect_header_row(preview: pd.DataFrame, max_rows: int = 8) -> int | None:
    if preview.empty:
        return 0
    best: tuple[float, int] | None = None
    limit = min(max_rows, len(preview))
    semantic = set(TIMESTAMP_HINTS + TARGET_HINTS + ID_HINTS)
    for i in range(limit):
        vals = [v for v in preview.iloc[i].tolist() if pd.notna(v) and str(v).strip() != ""]
        if len(vals) < 2:
            continue
        texts = [str(v).strip() for v in vals]
        lower = [x.lower() for x in texts]
        unique_ratio = len(set(lower)) / max(1, len(lower))
        text_ratio = np.mean([not _looks_number(x) and not _looks_datetime(x) for x in texts])
        semantic_hits = sum(any(h == x or (len(h) >= 4 and h in x) for h in semantic) for x in lower)
        next_data = 0.0
        if i + 1 < len(preview):
            nxt = [v for v in preview.iloc[i + 1].tolist() if pd.notna(v) and str(v).strip() != ""]
            if nxt:
                next_data = float(np.mean([_looks_number(v) or _looks_datetime(v) for v in nxt]))
        score = 1.4 * unique_ratio + 1.4 * float(text_ratio) + 1.2 * next_data + 0.7 * min(semantic_hits, 2) - 0.04 * i
        if best is None or score > best[0]:
            best = (score, i)
    if best is None:
        return None
    row = best[1]
    vals = [v for v in preview.iloc[row].tolist() if pd.notna(v) and str(v).strip() != ""]
    text_ratio = float(np.mean([not _looks_number(v) and not _looks_datetime(v) for v in vals])) if vals else 0.0
    semantic_hits = sum(any(h == str(v).strip().lower() or (len(h) >= 4 and h in str(v).strip().lower()) for h in semantic) for v in vals)
    return row if (text_ratio >= 0.45 or semantic_hits > 0) else None


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    seen: dict[str, int] = {}
    names = []
    for i, col in enumerate(out.columns):
        raw = str(col).strip() if col is not None else ""
        base = raw if raw and not raw.lower().startswith("unnamed") else f"column_{i+1}"
        if base in seen:
            seen[base] += 1
            base = f"{base}_{seen[base]}"
        else:
            seen[base] = 1
        names.append(base)
    out.columns = names
    return out


def read_table(source, return_metadata: bool = False):
    """Read CSV/Excel/Parquet and automatically detect messy header rows when possible."""
    raw_bytes, name = _as_bytes(source)
    lower = name.lower()
    meta = ReadMetadata()

    if lower.endswith(".csv"):
        meta.source_type = "csv"
        if raw_bytes is None:
            raw_bytes = Path(name).read_bytes()
        text = raw_bytes.decode("utf-8-sig", errors="replace")
        sample = text[:65536]
        lines = [line for line in sample.splitlines()[:20] if line.strip()]
        delimiter = max([",", ";", "\t", "|"], key=lambda d: max([line.count(d) for line in lines] or [0]))
        if max([line.count(delimiter) for line in lines] or [0]) == 0:
            delimiter = ","
        meta.delimiter = delimiter
        parsed_rows = list(csv.reader(lines[:12], delimiter=delimiter))
        width = max([len(r) for r in parsed_rows] or [1])
        padded = [r + [None] * (width - len(r)) for r in parsed_rows]
        preview = pd.DataFrame(padded)
        header_row = _detect_header_row(preview)
        meta.detected_header_row = header_row
        if header_row is None:
            df = pd.read_csv(StringIO(text), sep=delimiter, header=None, engine="python", on_bad_lines="skip")
            df.columns = [f"column_{i+1}" for i in range(df.shape[1])]
            meta.notes.append("No reliable header row was detected; generic column names were assigned.")
        else:
            df = pd.read_csv(StringIO(text), sep=delimiter, skiprows=header_row, header=0, engine="python")
            if header_row > 0:
                meta.notes.append(f"Detected the table header on row {header_row + 1} and skipped preceding metadata rows.")
    elif lower.endswith((".xlsx", ".xls")):
        meta.source_type = "excel"
        src = BytesIO(raw_bytes) if raw_bytes is not None else name
        preview = pd.read_excel(src, header=None, nrows=12)
        header_row = _detect_header_row(preview)
        meta.detected_header_row = header_row
        src = BytesIO(raw_bytes) if raw_bytes is not None else name
        if header_row is None:
            df = pd.read_excel(src, header=None)
            df.columns = [f"column_{i+1}" for i in range(df.shape[1])]
            meta.notes.append("No reliable header row was detected; generic column names were assigned.")
        else:
            df = pd.read_excel(src, header=header_row)
            if header_row > 0:
                meta.notes.append(f"Detected the table header on row {header_row + 1} and skipped preceding metadata rows.")
    elif lower.endswith(".parquet"):
        meta.source_type = "parquet"
        src = BytesIO(raw_bytes) if raw_bytes is not None else name
        df = pd.read_parquet(src)
        meta.detected_header_row = 0
    else:
        raise ValueError("Unsupported file type. Use CSV, Excel, or Parquet.")

    df = _clean_column_names(df)
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    return (df, meta) if return_metadata else df


def infer_timestamp_column(df: pd.DataFrame) -> str | None:
    cols = list(df.columns)
    lowered = {c: str(c).lower().strip() for c in cols}
    hinted = [c for c in cols if any(h == lowered[c] or (len(h) >= 4 and h in lowered[c]) for h in TIMESTAMP_HINTS)]
    candidates = hinted + [c for c in cols if c not in hinted]
    best, best_score = None, -1.0
    for col in candidates:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) and col not in hinted:
            # Unix-like integer timestamps are considered only when magnitudes are plausible.
            nums = pd.to_numeric(s, errors="coerce").dropna()
            if not len(nums) or not (nums.abs().median() > 1e8):
                continue
            parsed = pd.to_datetime(nums, unit="s", errors="coerce")
        else:
            parsed = pd.to_datetime(s, errors="coerce", utc=False)
        rate = float(parsed.notna().mean())
        unique = float(parsed.nunique(dropna=True) / max(1, parsed.notna().sum()))
        hint_bonus = 0.25 if col in hinted else 0.0
        score = rate + 0.2 * unique + hint_bonus
        if rate >= 0.75 and score > best_score:
            best, best_score = col, score
    return best


def _numeric_coercion_rate(s: pd.Series) -> float:
    if pd.api.types.is_numeric_dtype(s):
        return 1.0
    return float(pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce").notna().mean())


def infer_target_column(df: pd.DataFrame, timestamp_col: str | None = None) -> str | None:
    candidates = []
    for c in df.columns:
        if c == timestamp_col:
            continue
        rate = _numeric_coercion_rate(df[c])
        if rate >= 0.8:
            candidates.append(c)
    if not candidates:
        return None
    lowered = {c: str(c).lower().strip() for c in candidates}
    for hint in TARGET_HINTS:
        for c in candidates:
            if lowered[c] == hint:
                return c
    for hint in (h for h in TARGET_HINTS if len(h) >= 4):
        for c in candidates:
            if hint in lowered[c]:
                return c
    scored = []
    for c in candidates:
        s = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
        completeness = float(s.notna().mean())
        variability = float(np.log1p(max(s.nunique(dropna=True), 1)))
        id_penalty = 0.5 if any(h in lowered[c] for h in ID_HINTS) else 0.0
        scored.append((completeness * variability - id_penalty, c))
    return max(scored)[1]


def analyze_features(df: pd.DataFrame, timestamp_col: str, target_col: str) -> pd.DataFrame:
    rows = []
    n = max(1, len(df))
    target_name = str(target_col).lower()
    for c in df.columns:
        if c in {timestamp_col, target_col}:
            continue
        s = df[c]
        missing = float(100 * s.isna().mean())
        unique = int(s.nunique(dropna=True))
        unique_ratio = unique / n
        numeric_rate = _numeric_coercion_rate(s)
        name = str(c).lower().strip()
        if numeric_rate >= 0.8:
            kind = "numeric"
        elif pd.api.types.is_bool_dtype(s) or unique <= 2:
            kind = "boolean/categorical"
        elif unique <= min(100, max(12, int(0.1 * n))):
            kind = "categorical"
        else:
            kind = "high-cardinality text"

        reasons = []
        recommended = True
        role = "candidate feature"
        if unique <= 1:
            role, recommended = "constant / ignore", False
            reasons.append("no variation")
        if missing >= 60:
            role, recommended = "sparse / review", False
            reasons.append(f"{missing:.0f}% missing")
        if any(h == name or name.endswith(f"_{h}") for h in ID_HINTS) and unique_ratio > 0.8:
            role, recommended = "identifier / ignore", False
            reasons.append("looks like a row identifier")
        if kind == "high-cardinality text":
            role, recommended = "text / ignore", False
            reasons.append("too many categories for this release")
        if any(h in name for h in LEAKAGE_HINTS) and "lag" not in name:
            role, recommended = "possible leakage / review", False
            reasons.append("name suggests future or post-outcome information")
        if target_name and target_name in name and "lag" not in name and name != target_name:
            role, recommended = "possible leakage / review", False
            reasons.append("name overlaps the target")

        future_suggested = bool(recommended and any(h in name for h in FUTURE_KNOWN_HINTS))
        rows.append({
            "feature": c,
            "detected_type": kind,
            "role": role,
            "missing_%": missing,
            "unique_values": unique,
            "recommended_usable": recommended,
            "suggested_known_future": future_suggested,
            "reason": "; ".join(reasons) if reasons else "usable signal; future availability must be confirmed",
        })
    return pd.DataFrame(rows)


def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")


def _causal_fill_numeric(s: pd.Series) -> pd.Series:
    # Forward fill is causal. Leading gaps get the first finite value only as a final fallback;
    # this is flagged by preprocessing notes and affects only initial context.
    out = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).ffill()
    if out.isna().any():
        finite = out.dropna()
        fallback = float(finite.iloc[0]) if len(finite) else 0.0
        out = out.fillna(fallback)
    return out


def prepare_frame(
    df: pd.DataFrame,
    timestamp_col: str,
    target_col: str,
    known_future_cols: list[str] | None = None,
    *,
    regularize: bool = True,
    return_report: bool = False,
):
    known_future_cols = [c for c in (known_future_cols or []) if c in df.columns and c not in {timestamp_col, target_col}]
    feature_table = analyze_features(df, timestamp_col, target_col)
    original_rows = len(df)
    keep = [timestamp_col, target_col] + known_future_cols
    work = df[keep].copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="coerce")
    invalid_ts = int(work[timestamp_col].isna().sum())
    work[target_col] = _coerce_numeric_series(work[target_col])
    target_missing_before = int(work[target_col].isna().sum())
    feature_missing_before = int(work[known_future_cols].isna().sum().sum()) if known_future_cols else 0
    work = work.dropna(subset=[timestamp_col]).sort_values(timestamp_col)

    dupes = int(work.duplicated(timestamp_col).sum())
    if dupes:
        agg: dict[str, str] = {target_col: "mean"}
        for c in known_future_cols:
            agg[c] = "mean" if _numeric_coercion_rate(work[c]) >= 0.8 else "last"
        work = work.groupby(timestamp_col, as_index=False).agg(agg)

    # Optional bounded regularization. Avoid exploding memory across very large time gaps.
    inserted = 0
    regularized = False
    deltas = work[timestamp_col].diff().dt.total_seconds().dropna()
    med = float(deltas.median()) if len(deltas) else float("nan")
    if regularize and np.isfinite(med) and med > 0 and len(work) >= 3:
        regularity = float(np.mean(np.abs(deltas - med) <= max(1.0, med * 0.02)))
        expected = int(round((work[timestamp_col].iloc[-1] - work[timestamp_col].iloc[0]).total_seconds() / med)) + 1
        has_detected_gaps = expected > len(work)
        if (regularity < 0.98 or has_detected_gaps) and expected <= min(250_000, max(len(work) + 5000, int(len(work) * 1.75))):
            freq = pd.to_timedelta(med, unit="s")
            idx = pd.date_range(work[timestamp_col].iloc[0], work[timestamp_col].iloc[-1], freq=freq)
            before = len(work)
            work = work.set_index(timestamp_col).reindex(idx).rename_axis(timestamp_col).reset_index()
            inserted = max(0, len(work) - before)
            regularized = inserted > 0

    # Target: causal fill only; leading missing target rows are removed rather than using future target values.
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill()
    work = work.loc[work[target_col].notna()].copy()

    categorical_encoding: dict[str, dict[str, int]] = {}
    model_features: list[str] = []
    numeric_features: list[str] = []
    categorical_features: list[str] = []
    for c in known_future_cols:
        if c not in work.columns:
            continue
        if _numeric_coercion_rate(work[c]) >= 0.8:
            work[c] = _causal_fill_numeric(_coerce_numeric_series(work[c]))
            numeric_features.append(c)
            model_features.append(c)
        else:
            # Stable ordinal encoding keeps the model matrix compact for high-row Streamlit demos.
            vals = work[c].astype("string").replace("<NA>", np.nan).ffill().fillna("__MISSING__")
            categories = sorted(str(x) for x in vals.dropna().unique())
            mapping = {v: i for i, v in enumerate(categories)}
            work[c] = vals.map(mapping).fillna(-1).astype(float)
            categorical_encoding[c] = mapping
            categorical_features.append(c)
            model_features.append(c)

    warnings: list[str] = []
    if len(work) < 30:
        warnings.append("Very short series: fewer than 30 usable observations.")
    if work[target_col].nunique(dropna=True) < 3:
        warnings.append("Target has very low variability; forecasting may be uninformative.")
    if target_missing_before:
        warnings.append(f"Filled {target_missing_before} missing target observations causally with prior values; leading missing target rows were dropped.")
    if feature_missing_before:
        warnings.append("Missing feature values were filled with time-safe forward filling and compact fallbacks; review sparse features before operational use.")
    if regularized:
        warnings.append(f"Regularized the time grid and inserted {inserted} missing timestamp rows using the inferred interval.")
    if categorical_encoding:
        warnings.append(f"Encoded {len(categorical_encoding)} categorical feature(s) into stable numeric codes for model compatibility.")

    ignored = feature_table.loc[~feature_table["recommended_usable"], "feature"].astype(str).tolist() if not feature_table.empty else []
    suggested = feature_table.loc[feature_table["suggested_known_future"], "feature"].astype(str).tolist() if not feature_table.empty else []
    report = PreprocessingReport(
        original_rows=original_rows,
        final_rows=len(work),
        removed_invalid_timestamps=invalid_ts,
        duplicate_timestamps_aggregated=dupes,
        target_missing_before=target_missing_before,
        target_missing_after=int(work[target_col].isna().sum()),
        feature_missing_before=feature_missing_before,
        feature_missing_after=int(work[model_features].isna().sum().sum()) if model_features else 0,
        regularized=regularized,
        inserted_time_rows=inserted,
        feature_table=feature_table,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        ignored_features=ignored,
        suggested_future_features=suggested,
        model_features=model_features,
        categorical_encoding=categorical_encoding,
    )
    out = work.reset_index(drop=True)
    return (out, warnings, report) if return_report else (out, warnings)
