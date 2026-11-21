"""Original paired synthetic stress cases; fixed interventions, no physical claims."""

from typing import Any
import numpy as np
import pandas as pd

SCENARIOS = (
    "clean",
    "benign_drift",
    "gradual_drift",
    "sustained_shift",
    "stuck_sensor",
    "missing_random",
    "missing_block",
    "spikes_with_missing",
)
SENSORS = ["temperature", "vibration"]


def generate_stress(
    scenario: str,
    seed: int = 42,
    rows: int = 1000,
    assets: int = 3,
    missing_probability: float = 0.1,
    start_fraction: float = 0.82,
    end_fraction: float = 0.94,
) -> pd.DataFrame:
    if scenario not in SCENARIOS:
        raise ValueError("Unknown stress scenario")
    if rows < 100 or assets < 1 or seed < 0:
        raise ValueError("Need >=100 rows, >=1 asset and nonnegative seed")
    if not 0 <= missing_probability <= 1 or not 0.8 <= start_fraction < end_fraction <= 1:
        raise ValueError("Invalid missing probability or held-out event boundaries")
    start, end = int(rows * start_fraction), int(rows * end_fraction)
    if end <= start:
        raise ValueError("Event interval must contain observations")
    rng = np.random.default_rng(seed)
    frames = []
    for asset in range(assets):
        t = np.arange(rows)
        temperature = 50 + 2 * np.sin(t / 30) + rng.normal(0, 0.5, rows)
        vibration = 5 + rng.normal(0, 0.3, rows)
        labels = np.zeros(rows, dtype=int)
        if scenario == "benign_drift":
            temperature[start:end] += np.linspace(0, 1.5, end - start)
        elif scenario == "gradual_drift":
            temperature[start:end] += np.linspace(0, 12, end - start)
            vibration[start:end] += np.linspace(0, 3, end - start)
            labels[start:end] = 1
        elif scenario == "sustained_shift":
            temperature[start:end] += 8
            vibration[start:end] += 2.5
            labels[start:end] = 1
        elif scenario == "stuck_sensor":
            temperature[start:end] = temperature[start - 1]
            labels[start:end] = 1
        elif scenario == "missing_block":
            temperature[start:end] = np.nan
            vibration[start:end] = np.nan
            labels[start:end] = 1
        elif scenario == "spikes_with_missing":
            spike_rng = np.random.default_rng(np.random.SeedSequence([seed, asset, 502]))
            injected = spike_rng.random(rows - int(0.8 * rows)) < 0.04
            positions = np.flatnonzero(injected) + int(0.8 * rows)
            labels[positions] = 1
            temperature[positions] += spike_rng.choice([-1, 1], len(positions)) * spike_rng.uniform(
                6, 12, len(positions)
            )
            vibration[positions] += spike_rng.uniform(2, 4, len(positions))
        if scenario in {"missing_random", "spikes_with_missing"}:
            missing_rng = np.random.default_rng(np.random.SeedSequence([seed, asset, 501]))
            positions = np.arange(int(0.8 * rows), rows)
            missing = missing_rng.random((len(positions), 2)) < missing_probability
            temperature[positions[missing[:, 0]]] = np.nan
            vibration[positions[missing[:, 1]]] = np.nan
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": pd.date_range(
                        "2025-01-01", periods=rows, freq="min", tz="UTC"
                    ).astype(str),
                    "asset_id": f"asset-{asset}",
                    "temperature": temperature,
                    "vibration": vibration,
                    "label": labels,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def event_diagnostics(scored: pd.DataFrame, method: str, window: int) -> dict[str, Any]:
    """Raw contiguous-positive episode diagnostics; never adjust point predictions."""
    events = []
    early_hits = early_rows = late_hits = late_rows = 0
    for asset, group in scored.groupby("asset_id", sort=True):
        y = group.label.to_numpy(dtype=int)
        edges = np.diff(np.r_[0, y, 0])
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
            alerts = group[f"{method}_alert"].to_numpy(dtype=bool)[start:end]
            hits = np.flatnonzero(alerts)
            boundary = min(window, len(alerts))
            early_hits += int(alerts[:boundary].sum())
            early_rows += boundary
            late_hits += int(alerts[boundary:].sum())
            late_rows += len(alerts) - boundary
            events.append(
                {
                    "asset_id": str(asset),
                    "start": str(group.timestamp.iloc[start]),
                    "end": str(group.timestamp.iloc[end - 1]),
                    "observations": int(end - start),
                    "alerted_observations": int(alerts.sum()),
                    "hit": bool(len(hits)),
                    "first_alert_delay_observations": int(hits[0]) if len(hits) else None,
                    "point_coverage": float(alerts.mean()),
                }
            )
    return {
        "definition": "contiguous positive labels within asset; no point adjustment",
        "events": events,
        "event_count": len(events),
        "events_hit": sum(e["hit"] for e in events),
        "early_positive_rows": early_rows,
        "late_positive_rows": late_rows,
        "early_point_recall": early_hits / early_rows if early_rows else None,
        "late_point_recall": late_hits / late_rows if late_rows else None,
    }
