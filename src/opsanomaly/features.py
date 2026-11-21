"""Deterministic chronological partitions and training-only preprocessing."""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from .ingest import validate


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df, _ = validate(df)
    for _, group in df.groupby("asset_id", sort=True):
        if len(group) < 15:
            raise ValueError("Each asset needs >=15 ordered observations")
    # Shared cutoffs: a pooled model must never train on another asset's future.
    times = df.timestamp.drop_duplicates().sort_values()
    if len(times) < 15:
        raise ValueError("Need >=15 distinct UTC timestamps")
    a, b = int(len(times) * 0.6), int(len(times) * 0.8)
    calibration_start, test_start = times.iloc[a], times.iloc[b]
    train = df[df.timestamp < calibration_start].copy()
    calibration = df[(df.timestamp >= calibration_start) & (df.timestamp < test_start)].copy()
    test = df[df.timestamp >= test_start].copy()
    assets = set(df.asset_id)
    for name, part in (("train", train), ("calibration", calibration), ("test", test)):
        if set(part.asset_id) != assets:
            raise ValueError(
                f"Every asset must occur in shared {name} period; timelines incompatible"
            )
    return train, calibration, test


class Preprocessor:
    def fit(self, df: pd.DataFrame, sensors: list[str]) -> "Preprocessor":
        if not sensors or len(set(sensors)) != len(sensors):
            raise ValueError("Features must be nonempty and unique")
        if any(c in {"label", "timestamp", "asset_id"} for c in sensors):
            raise ValueError("Identifiers and labels cannot be features")
        self.sensors = sensors.copy()
        self.medians = df[sensors].median().to_numpy(dtype=float)
        if np.isnan(self.medians).any():
            raise ValueError("A training sensor is entirely missing")
        x = self.transform(df)
        self.scale = np.median(np.abs(x - self.medians), axis=0) * 1.4826
        self.scale = np.where(self.scale < 1e-9, 1.0, self.scale)
        return self

    def transform(self, df: pd.DataFrame) -> NDArray[np.float64]:
        x = df[self.sensors].to_numpy(dtype=float)
        if np.isinf(x).any() or (np.abs(x[~np.isnan(x)]) > 1e12).any():
            raise ValueError("Nonfinite or out-of-range inference sensor")
        return np.asarray(np.where(np.isnan(x), self.medians, x), dtype=np.float64)

    def deviations(self, df: pd.DataFrame) -> NDArray[np.float64]:
        return np.asarray(
            np.abs((self.transform(df) - self.medians) / self.scale), dtype=np.float64
        )
