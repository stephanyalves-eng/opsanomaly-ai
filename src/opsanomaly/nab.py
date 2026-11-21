"""Strict local adapter for NAB window labels; never claims official NAB scoring."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any
import numpy as np
import pandas as pd
from .ingest import validate


class SourceValidationError(ValueError):
    def __init__(self, message: str, diagnostics: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


def safe_path(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts or "\\" in relative:
        raise ValueError("Unsafe source path")
    path = root.joinpath(*parts.parts).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Source path escapes dataset root")
    return path


def verify_sources(root: Path, lock: dict[str, Any], revision: str) -> None:
    if lock["revision"] != revision or lock["repository"] != "https://github.com/numenta/NAB":
        raise ValueError("Source revision/repository mismatch")
    for name, expected in lock["files"].items():
        path = safe_path(root, name)
        if path.stat().st_size > 10_000_000:
            raise ValueError("Oversized source file")
        payload = path.read_bytes()
        if (
            len(payload) != expected["bytes"]
            or hashlib.sha256(payload).hexdigest() != expected["sha256"]
        ):
            raise ValueError(f"Source checksum mismatch: {name}")


def window_labels(timestamps: pd.Series[Any], windows: list[list[str]]) -> np.ndarray[Any, Any]:
    labels = np.zeros(len(timestamps), dtype=int)
    previous_end = None
    for window in windows:
        if len(window) != 2:
            raise ValueError("Window must have two endpoints")
        start, end = [pd.Timestamp(v) for v in window]
        if pd.isna(start) or pd.isna(end):
            raise ValueError("Missing window endpoint")
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        if end < start or previous_end is not None and start <= previous_end:
            raise ValueError("Windows must be ordered, non-overlapping and valid")
        selected = ((timestamps >= start) & (timestamps <= end)).to_numpy()
        if not selected.any():
            raise ValueError("Published window has no observations in source series")
        labels[selected] = 1
        previous_end = end
    return labels


def load_series(
    path: Path, windows: list[list[str]], asset: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(path, nrows=1_000_001)
    if list(raw.columns) != ["timestamp", "value"]:
        raise ValueError("Expected exact NAB timestamp,value schema")
    clock = pd.to_datetime(raw.timestamp, errors="raise", format="mixed")
    if clock.isna().any():
        raise ValueError("Missing source timestamps")
    naive = clock.dt.tz is None
    timestamps = clock.dt.tz_localize("UTC") if naive else clock.dt.tz_convert("UTC")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "asset_id": asset,
            "sensor_value": raw.value,
            "label": window_labels(timestamps, windows),
        }
    )
    audit = {
        "rows": len(raw),
        "duplicate_timestamp_rows_beyond_first": int(raw.timestamp.duplicated().sum()),
        "exact_duplicate_rows_beyond_first": int(raw.duplicated().sum()),
        "conflicting_timestamp_groups": int(
            (raw.groupby("timestamp").value.nunique(dropna=False) > 1).sum()
        ),
        "duplicate_timestamp_examples": raw.loc[raw.timestamp.duplicated(), "timestamp"]
        .astype(str)
        .head(20)
        .tolist(),
    }
    try:
        clean, quality = validate(frame)
    except ValueError as error:
        raise SourceValidationError(str(error), audit) from error
    return clean, {
        "validation": quality,
        "source_rows": len(raw),
        "source_windows": windows,
        "source_timezone": "unspecified naive clock" if naive else "explicit source offset",
        "timezone_assumption": "UTC ordering coordinate only; publisher physical timezone not verified",
        "label_semantics": "inclusive published anomaly-window membership, not pointwise fault truth",
    }


def read_labels(root: Path, relative: str, series: str) -> list[list[str]]:
    content = json.loads(safe_path(root, relative).read_text(encoding="utf-8"))
    if series not in content:
        raise ValueError(
            "No published label entry for chosen series; do not assume unlabeled negatives"
        )
    windows: list[list[str]] = content[series]
    return windows
