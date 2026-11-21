"""Frozen and past-only rolling robust statistics; no label-based fitting."""

from collections import deque
from typing import Any
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from .features import Preprocessor


def score(deviations: Any) -> Any:
    return np.max(deviations, axis=1)


class RollingMAD:
    """Pure batch replay with an immutable initial history per known asset.

    Each row is scored before it enters the window. Missing values use the
    frozen training imputer. A zero local MAD falls back to training scale.
    """

    def __init__(self, window: int = 31) -> None:
        if isinstance(window, bool) or not isinstance(window, int) or not 3 <= window <= 1000:
            raise ValueError("Rolling window must be an integer in 3..1000")
        self.window = window

    def history(self, df: pd.DataFrame, pre: Preprocessor) -> dict[str, NDArray[np.float64]]:
        return {
            str(asset): pre.transform(g.tail(self.window)).copy()
            for asset, g in df.groupby("asset_id", sort=True)
        }

    def deviations(
        self, df: pd.DataFrame, pre: Preprocessor, history: dict[str, NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        result = np.empty((len(df), len(pre.sensors)), dtype=np.float64)
        x = pre.transform(df)
        # Positional indices support duplicate user DataFrame indices safely.
        for asset, positions in df.groupby("asset_id", sort=True).indices.items():
            key = str(asset)
            if key not in history or len(history[key]) < 3:
                raise ValueError("Rolling score needs >=3 prior training rows per known asset")
            past = deque(history[key].copy(), maxlen=self.window)
            for position in positions:
                values = np.asarray(past)
                center = np.median(values, axis=0)
                scale = 1.4826 * np.median(np.abs(values - center), axis=0)
                scale = np.where(scale < 1e-9, pre.scale, scale)
                result[position] = np.abs((x[position] - center) / scale)
                past.append(x[position].copy())
        return result
