"""Strict CSV validation; never silently discard or reorder records."""

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd


def validate(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = frame.copy()
    if not df.columns.is_unique:
        raise ValueError("Duplicate column names")
    if df.empty or len(df) > 1_000_000:
        raise ValueError("Expected 1..1,000,000 rows")
    if not {"timestamp", "asset_id"}.issubset(df.columns):
        raise ValueError("timestamp and asset_id are required")
    if df.asset_id.isna().any() or df.asset_id.astype(str).str.strip().eq("").any():
        raise ValueError("asset_id must be nonempty")
    # Require explicit offsets: naive local times and DST ambiguity are unsafe.
    raw = df.timestamp.astype(str)
    if not raw.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True).all():
        raise ValueError("Every timestamp must include an explicit timezone offset")
    df["timestamp"] = pd.to_datetime(raw, utc=True, errors="raise", format="mixed")
    df["asset_id"] = df.asset_id.astype(str)
    if df.duplicated(["asset_id", "timestamp"]).any():
        raise ValueError("Duplicate asset/timestamp")
    sensors = [c for c in df.columns if c not in {"timestamp", "asset_id", "label"}]
    if not sensors or len(sensors) > 64:
        raise ValueError("Expected 1..64 numeric sensors")
    for c in sensors:
        df[c] = pd.to_numeric(df[c], errors="raise")
        observed = df[c].dropna().to_numpy(dtype=float)
        if not np.isfinite(observed).all() or (np.abs(observed) > 1e12).any():
            raise ValueError(f"Nonfinite or out-of-range sensor: {c}")
    if "label" in df and (df.label.isna().any() or not df.label.isin([0, 1]).all()):
        raise ValueError("Labels must be complete binary 0/1")
    irregular = {}
    for asset, group in df.groupby("asset_id", sort=True):
        if not group.timestamp.is_monotonic_increasing:
            raise ValueError(f"Timestamps must increase within asset {asset}")
        gaps = group.timestamp.diff().dropna().dt.total_seconds()
        irregular[str(asset)] = int((gaps != gaps.median()).sum())
    return df, {
        "rows": len(df),
        "sensors": sensors,
        "missing": {c: int(df[c].isna().sum()) for c in sensors},
        "irregular_intervals": irregular,
        "timezone": "UTC",
    }


def read_csv(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if Path(path).stat().st_size > 100_000_000:
        raise ValueError("CSV exceeds 100 MB")
    return validate(pd.read_csv(path, nrows=1_000_001))
