"""Chronologically guarded scoring with calibration-only thresholds."""

from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    precision_recall_curve,
    auc,
)
from . import baseline, model
from .explain import evidence
from .features import Preprocessor
from .ingest import validate

DETECTORS = ("mad", "rolling_mad", "isolation_forest")
ARTIFACT_SCHEMA = 2


class Detector:
    def fit(
        self,
        train: pd.DataFrame,
        calibration: pd.DataFrame,
        sensors: list[str],
        rate: float = 0.02,
        seed: int = 42,
        window: int = 31,
    ) -> "Detector":
        if not 0 < rate < 1:
            raise ValueError("Calibration rate must be between zero and one")
        if any(c in {"label", "timestamp", "asset_id"} for c in sensors):
            raise ValueError("Identifiers and labels cannot be features")
        train, train_quality = validate(train)
        calibration, cal_quality = validate(calibration)
        if train_quality["sensors"] != sensors or cal_quality["sensors"] != sensors:
            raise ValueError("Sensor schema/order must match the explicit feature list")
        if set(train.asset_id) != set(calibration.asset_id):
            raise ValueError("Training and calibration assets must match")
        if train.timestamp.max() >= calibration.timestamp.min():
            raise ValueError("Calibration must follow all training without overlap, across assets")
        if train.groupby("asset_id").size().min() < 3:
            raise ValueError("Need >=3 training rows per asset for rolling history")
        self.artifact_schema = ARTIFACT_SCHEMA
        self.pre = Preprocessor().fit(train, sensors)
        self.forest = model.fit(self.pre.transform(train), seed)
        self.rolling = baseline.RollingMAD(window)
        self.train_history = self.rolling.history(train, self.pre)
        self.inference_history = self.rolling.history(pd.concat([train, calibration]), self.pre)
        self.training_end = train.timestamp.max()
        self.calibration_end = calibration.timestamp.max()
        self.assets = set(train.asset_id)
        scores = self._scores(calibration, self.train_history)
        self.thresholds = {
            k: float(np.quantile(v, 1 - rate, method="higher")) for k, v in scores.items()
        }
        self.calibration_rate = rate
        return self

    def _scores(self, df: pd.DataFrame, history: dict[str, Any]) -> dict[str, Any]:
        return {
            "mad": baseline.score(self.pre.deviations(df)),
            "rolling_mad": baseline.score(self.rolling.deviations(df, self.pre, history)),
            "isolation_forest": model.score(self.forest, self.pre.transform(df)),
        }

    def _validate_future(self, df: pd.DataFrame) -> pd.DataFrame:
        if getattr(self, "artifact_schema", None) != ARTIFACT_SCHEMA:
            raise ValueError("Unsupported artifact schema; retrain with the current pipeline")
        clean, quality = validate(df)
        if quality["sensors"] != self.pre.sensors:
            raise ValueError("Sensor schema/order differs from training")
        if not set(clean.asset_id).issubset(self.assets):
            raise ValueError("Unknown asset has no rolling history; retrain explicitly")
        if clean.timestamp.min() <= self.calibration_end:
            raise ValueError("Inference must be strictly after the entire calibration period")
        return clean

    def scores(self, df: pd.DataFrame) -> dict[str, Any]:
        return self._scores(self._validate_future(df), self.inference_history)

    def predict(self, df: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
        """Replay a complete future batch, without mutating fitted history or thresholds.

        A manual threshold is exploratory and applies only to MAD; method score
        scales are incompatible. Prefer the calibration-rate parameter at fit.
        """
        if threshold is not None and not np.isfinite(threshold):
            raise ValueError("Threshold must be finite")
        clean = self._validate_future(df)
        out = clean.copy()
        frozen = self.pre.deviations(clean)
        rolling = self.rolling.deviations(clean, self.pre, self.inference_history)
        values = {
            "mad": baseline.score(frozen),
            "rolling_mad": baseline.score(rolling),
            "isolation_forest": model.score(self.forest, self.pre.transform(clean)),
        }
        for name, scores in values.items():
            cutoff = threshold if threshold is not None and name == "mad" else self.thresholds[name]
            out[f"{name}_score"] = scores
            out[f"{name}_threshold"] = cutoff
            out[f"{name}_alert"] = scores > cutoff
        out["evidence"] = evidence(frozen, self.pre.sensors)
        out["rolling_mad_evidence"] = evidence(rolling, self.pre.sensors)
        out["isolation_forest_evidence"] = out.evidence.map(
            lambda x: "Training-MAD contextual reference only, NOT forest attribution: " + x
        )
        out["threshold_source"] = (
            "manual MAD override; exploratory"
            if threshold is not None
            else ("unlabeled calibration quantile; fixed before this batch")
        )
        return out


def _unit_interval(value: float) -> float:
    """Correct only floating-point endpoint roundoff, never a genuinely invalid metric."""
    if not np.isfinite(value) or not -1e-12 <= value <= 1 + 1e-12:
        raise ValueError("Metric outside probability bounds")
    return float(np.clip(value, 0, 1))


def metrics(scored: pd.DataFrame) -> dict[str, Any]:
    if scored.empty:
        raise ValueError("Cannot evaluate empty observations")
    labeled = "label" in scored
    if labeled and (scored.label.isna().any() or not scored.label.isin([0, 1]).all()):
        raise ValueError("Evaluation labels must be complete binary 0/1")
    result = {}
    for name in DETECTORS:
        if f"{name}_score" not in scored or f"{name}_alert" not in scored:
            raise ValueError(f"Missing scores or alerts for {name}")
        values = scored[f"{name}_score"].to_numpy(dtype=float)
        alerts = scored[f"{name}_alert"].to_numpy()
        if not np.isfinite(values).all() or alerts.dtype != np.dtype(bool):
            raise ValueError("Scores must be finite; alerts must be complete booleans")
        item: dict[str, Any] = {
            "labeled": labeled,
            "threshold_sources": sorted(scored.threshold_source.unique().tolist())
            if "threshold_source" in scored
            else ["caller-supplied alerts; provenance unspecified"],
            "observations": len(scored),
            "alerts": int(alerts.sum()),
            "alert_rate": float(alerts.mean()),
            "score_quantiles": [float(v) for v in np.quantile(values, [0, 0.5, 0.95, 1])],
        }
        if labeled:
            y = scored.label.to_numpy(dtype=int)
            positive, negative = int(y.sum()), int((y == 0).sum())
            tp, fp = int(((y == 1) & alerts).sum()), int(((y == 0) & alerts).sum())
            p, r, f, _ = precision_recall_fscore_support(
                y, alerts, average="binary", zero_division=0
            )
            both_classes = positive > 0 and negative > 0
            pr_precision, pr_recall, _ = (
                precision_recall_curve(y, values)
                if both_classes
                else (np.array([]), np.array([]), np.array([]))
            )
            item.update(
                precision=float(p),
                recall=float(r),
                f1=float(f),
                pr_auc_average_precision=_unit_interval(float(average_precision_score(y, values)))
                if both_classes
                else None,
                pr_auc_trapezoid=_unit_interval(float(auc(pr_recall, pr_precision)))
                if both_classes
                else None,
                false_alarms_per_1000_observations=float(fp * 1000 / len(y)),
                positives=positive,
                negatives=negative,
                confusion_matrix={"tp": tp, "fp": fp, "tn": negative - fp, "fn": positive - tp},
                pr_auc_status="defined" if both_classes else "undefined: single-class labels",
                undefined_metrics=(["precision"] if not alerts.any() else [])
                + (["recall"] if not positive else [])
                + (["f1"] if not (positive + int(alerts.sum())) else []),
                zero_division_policy="undefined precision/recall/F1 represented as 0; see undefined_metrics",
            )
        result[name] = item
    return result
