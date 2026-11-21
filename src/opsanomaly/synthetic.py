"""Original synthetic data; no external dataset or implied real failure labels."""

import numpy as np
import pandas as pd


def generate(seed: int = 42, rows: int = 1000, assets: int = 3) -> pd.DataFrame:
    if rows < 15 or assets < 1:
        raise ValueError("Need >=15 rows per asset and >=1 asset")
    rng = np.random.default_rng(seed)
    frames = []
    for asset in range(assets):
        t = np.arange(rows)
        temperature = 50 + 2 * np.sin(t / 30) + rng.normal(0, 0.5, rows)
        vibration = 5 + rng.normal(0, 0.3, rows)
        labels = rng.random(rows) < 0.04
        temperature[labels] += rng.choice([-1, 1], labels.sum()) * rng.uniform(6, 12, labels.sum())
        vibration[labels] += rng.uniform(2, 4, labels.sum())
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": pd.date_range(
                        "2025-01-01", periods=rows, freq="min", tz="UTC"
                    ).astype(str),
                    "asset_id": f"asset-{asset}",
                    "temperature": temperature,
                    "vibration": vibration,
                    "label": labels.astype(int),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
