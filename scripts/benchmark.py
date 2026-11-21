"""Predeclared synthetic comparison; never choose settings from test labels."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
from typing import Any
import numpy as np
from opsanomaly.features import temporal_split
from opsanomaly.ingest import validate
from opsanomaly.inference import Detector, metrics, DETECTORS
from opsanomaly.protocol import SPLIT_PROTOCOL, split_manifest
from opsanomaly.synthetic import generate


def run(seed: int, rows: int, rate: float, window: int) -> dict[str, Any]:
    frame, _ = validate(generate(seed=seed, rows=rows))
    parts = temporal_split(frame)
    train, cal, test = parts
    started = time.perf_counter()
    detector = Detector().fit(train, cal, ["temperature", "vibration"], rate, seed, window)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    scored = detector.predict(test)
    inference_seconds = time.perf_counter() - started
    return {
        "seed": seed,
        "dataset_sha256_canonical_csv": hashlib.sha256(
            frame.to_csv(index=False, lineterminator="\n").encode()
        ).hexdigest(),
        "splits": split_manifest(parts),
        "thresholds": detector.thresholds,
        "calibration_positives": int(cal.label.sum()),
        "calibration_rows": len(cal),
        "metrics": metrics(scored),
        "unlabeled_metrics": metrics(scored.drop(columns="label")),
        "by_asset": {str(asset): metrics(g) for asset, g in scored.groupby("asset_id", sort=True)},
        "timing": {"fit_seconds": fit_seconds, "batch_inference_seconds": inference_seconds},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--rows", type=int, default=1000)
    p.add_argument("--alert-rate", type=float, default=0.02)
    p.add_argument("--window", type=int, default=31)
    p.add_argument("--output", default="docs/benchmark.json")
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be nonempty and distinct")
    runs = [run(seed, args.rows, args.alert_rate, args.window) for seed in seeds]
    summary = {
        name: {
            key: {
                "mean": float(np.mean([r["metrics"][name][key] for r in runs])),
                "std_population": float(np.std([r["metrics"][name][key] for r in runs])),
            }
            for key in ("precision", "recall", "f1", "false_alarms_per_1000_observations")
        }
        for name in DETECTORS
    }
    report = {
        "schema_version": 2,
        "scope": "original synthetic point injections only",
        "evaluation_mode": "labeled; independent unlabeled views also included",
        "protocol": {
            "split": SPLIT_PROTOCOL,
            "seeds": seeds,
            "rows_per_asset": args.rows,
            "assets": 3,
            "calibration_alert_rate": args.alert_rate,
            "rolling_window": args.window,
            "test_labels_used_for_tuning": False,
            "rolling_updates": "past sensor values only, including prior test observations; no labels",
            "thresholds": "unlabeled calibration higher quantiles, strict greater-than",
        },
        "versions": {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "scikit-learn")},
        "runs": runs,
        "equal_weight_seed_summary": summary,
        "limitations": [
            "Only three dependent simulation runs; no confidence intervals or generalization claim",
            "Seed averages are not pooled PR curves or independent industrial replications",
            "Positive calibration contamination is retained; labels are diagnostic only",
            "No test-based tuning or event-level point adjustment",
            "Timing is single-run local CPU timing",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
