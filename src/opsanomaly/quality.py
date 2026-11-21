"""Separate raw-data quality rules: descriptive, past-only, never a diagnosis."""

from dataclasses import dataclass, replace
import math
from typing import Any
import numpy as np
import pandas as pd
from .ingest import validate


@dataclass(frozen=True)
class QualityRules:
    stuck_run_length: int = 8
    missing_run_length: int = 3
    repeat_tolerance: float = 1e-8
    gap_multiplier: float = 1.5

    def __post_init__(self) -> None:
        for value in (self.stuck_run_length, self.missing_run_length):
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError("Run lengths must be integers >=2")
        if not math.isfinite(self.repeat_tolerance) or self.repeat_tolerance < 0:
            raise ValueError("Repeat tolerance must be finite and nonnegative")
        if not math.isfinite(self.gap_multiplier) or self.gap_multiplier <= 1:
            raise ValueError("Gap multiplier must be finite and >1")


@dataclass
class _SensorState:
    anchor: float | None = None
    repeat_count: int = 0
    missing_count: int = 0


class QualityMonitor:
    def __init__(self, rules: QualityRules | None = None) -> None:
        self.rules = rules or QualityRules()

    def fit(
        self, train: pd.DataFrame, calibration: pd.DataFrame, sensors: list[str]
    ) -> "QualityMonitor":
        train, tq = validate(train)
        calibration, cq = validate(calibration)
        if tq["sensors"] != sensors or cq["sensors"] != sensors:
            raise ValueError("Exact sensor schema/order required")
        if set(train.asset_id) != set(calibration.asset_id):
            raise ValueError("Training/calibration assets must match")
        if train.timestamp.max() >= calibration.timestamp.min():
            raise ValueError("Calibration must follow all training")
        if any(s.startswith("quality_") for s in sensors):
            raise ValueError("quality_ is a reserved output prefix")
        self.sensors = sensors.copy()
        self.assets = set(train.asset_id)
        self.cadence: dict[str, float] = {}
        self.assessable: dict[str, dict[str, bool]] = {}
        for asset, group in train.groupby("asset_id", sort=True):
            key = str(asset)
            gaps = group.timestamp.diff().dropna().dt.total_seconds()
            if len(group) < 3 or gaps.empty:
                raise ValueError("Need >=3 training observations per asset")
            self.cadence[key] = float(gaps.median())
            self.assessable[key] = {}
            for sensor in sensors:
                observed = group[sensor].dropna()
                self.assessable[key][sensor] = bool(
                    len(observed) >= 3
                    and float(observed.max() - observed.min()) > self.rules.repeat_tolerance
                )
        initial = {asset: {s: _SensorState() for s in sensors} for asset in self.assessable}
        _, state, last = self._replay(train, initial, {})
        _, state, last = self._replay(calibration, state, last)
        self.history = state
        self.last_times = last
        self.calibration_end = calibration.timestamp.max()
        self.schema_version = 1
        return self

    def _replay(
        self,
        frame: pd.DataFrame,
        initial: dict[str, dict[str, _SensorState]],
        previous: dict[str, pd.Timestamp],
    ) -> tuple[pd.DataFrame, dict[str, dict[str, _SensorState]], dict[str, pd.Timestamp]]:
        state = {a: {s: replace(v) for s, v in channels.items()} for a, channels in initial.items()}
        last = previous.copy()
        flags: dict[str, Any] = {}
        for sensor in self.sensors:
            for field in ("missing", "missing_alert", "repeat_alert", "repeat_assessable"):
                flags[f"quality_{sensor}_{field}"] = np.zeros(len(frame), dtype=bool)
            for field in ("missing_run", "repeat_run"):
                flags[f"quality_{sensor}_{field}"] = np.zeros(len(frame), dtype=int)
        evidence = [""] * len(frame)
        gap_reset = np.zeros(len(frame), dtype=bool)
        for asset, positions in frame.groupby("asset_id", sort=True).indices.items():
            key = str(asset)
            for position in positions:
                row = frame.iloc[position]
                timestamp = row.timestamp
                gap = (
                    key in last
                    and (timestamp - last[key]).total_seconds()
                    > self.rules.gap_multiplier * self.cadence[key]
                )
                if gap:
                    state[key] = {s: _SensorState() for s in self.sensors}
                    gap_reset[position] = True
                notes = []
                for sensor in self.sensors:
                    current = state[key][sensor]
                    missing = bool(pd.isna(row[sensor]))
                    allowed = self.assessable[key][sensor]
                    if missing:
                        current.missing_count += 1
                        current.anchor = None
                        current.repeat_count = 0
                        notes.append(f"{sensor}: missing raw observation")
                    else:
                        value = float(row[sensor])
                        current.missing_count = 0
                        if (
                            current.anchor is not None
                            and abs(value - current.anchor) <= self.rules.repeat_tolerance
                        ):
                            current.repeat_count += 1
                        else:
                            current.anchor = value
                            current.repeat_count = 1
                    repeat_alert = allowed and current.repeat_count >= self.rules.stuck_run_length
                    missing_alert = current.missing_count >= self.rules.missing_run_length
                    if repeat_alert:
                        notes.append(
                            f"{sensor}: {current.repeat_count} repeated values; review, not fault diagnosis"
                        )
                    if missing_alert:
                        notes.append(
                            f"{sensor}: {current.missing_count} consecutive missing observations"
                        )
                    if not allowed:
                        notes.append(
                            f"{sensor}: repeat rule not assessable from training variability"
                        )
                    flags[f"quality_{sensor}_missing"][position] = missing
                    flags[f"quality_{sensor}_missing_alert"][position] = missing_alert
                    flags[f"quality_{sensor}_repeat_alert"][position] = repeat_alert
                    flags[f"quality_{sensor}_repeat_assessable"][position] = allowed
                    flags[f"quality_{sensor}_missing_run"][position] = current.missing_count
                    flags[f"quality_{sensor}_repeat_run"][position] = current.repeat_count
                evidence[position] = (
                    "; ".join(notes) or "No quality rule triggered; not a health guarantee"
                )
                last[key] = timestamp
        out = frame.copy()
        for column, values in flags.items():
            out[column] = values
        out["quality_missing_any"] = out[[f"quality_{s}_missing" for s in self.sensors]].any(axis=1)
        out["quality_missing_alert"] = out[
            [f"quality_{s}_missing_alert" for s in self.sensors]
        ].any(axis=1)
        out["quality_repeat_alert"] = out[[f"quality_{s}_repeat_alert" for s in self.sensors]].any(
            axis=1
        )
        out["quality_review_alert"] = out.quality_missing_alert | out.quality_repeat_alert
        out["quality_gap_reset"] = gap_reset
        out["quality_evidence"] = evidence
        return out, state, last

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if getattr(self, "schema_version", None) != 1:
            raise ValueError("Fit quality monitor first")
        clean, quality = validate(frame)
        if quality["sensors"] != self.sensors or not set(clean.asset_id).issubset(self.assets):
            raise ValueError("Unknown asset or different sensor schema")
        if clean.timestamp.min() <= self.calibration_end:
            raise ValueError("Quality inference must follow entire calibration period")
        return self._replay(clean, self.history, self.last_times)[0]


def quality_summary(scored: pd.DataFrame, sensors: list[str]) -> dict[str, Any]:
    return {
        "scope": "raw-data quality flags, not operational anomaly performance or diagnosis",
        "observations": len(scored),
        "missing_rows": int(scored.quality_missing_any.sum()),
        "review_alerts": int(scored.quality_review_alert.sum()),
        "review_alert_rate": float(scored.quality_review_alert.mean()),
        "repeat_alerts": int(scored.quality_repeat_alert.sum()),
        "persistent_missing_alerts": int(scored.quality_missing_alert.sum()),
        "gap_resets": int(scored.quality_gap_reset.sum()),
        "by_sensor": {
            s: {
                "missing_values": int(scored[f"quality_{s}_missing"].sum()),
                "repeat_alerts": int(scored[f"quality_{s}_repeat_alert"].sum()),
                "repeat_assessable_rows": int(scored[f"quality_{s}_repeat_assessable"].sum()),
            }
            for s in sensors
        },
    }
