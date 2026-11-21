"""Offline, fixed-protocol NAB subset evaluation; not the official NAB score."""

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any
from opsanomaly.features import temporal_split
from opsanomaly.inference import Detector, DETECTORS, metrics
from opsanomaly.nab import (
    load_series,
    read_labels,
    verify_sources,
    safe_path,
    SourceValidationError,
)
from opsanomaly.protocol import split_manifest
from opsanomaly.quality import QualityMonitor, quality_summary
from opsanomaly.stress import event_diagnostics


def evaluate(
    plan: dict[str, Any], root: Path, lock: dict[str, Any], output: Path
) -> dict[str, Any]:
    if (
        plan["protocol_version"] != "nab-local-v1"
        or not plan["series"]
        or len(set(plan["series"])) != len(plan["series"])
    ):
        raise ValueError("Invalid NAB protocol")
    verify_sources(root, lock, plan["revision"])
    required = {
        plan["labels_path"],
        plan["license_path"],
        *plan["metadata_paths"],
        *("data/" + s for s in plan["series"]),
    }
    if set(lock["files"]) != required:
        raise ValueError("Protocol/source-lock file coverage differs")
    output.mkdir(parents=True, exist_ok=True)
    runs = []
    for series in plan["series"]:
        windows = read_labels(root, plan["labels_path"], series)
        try:
            frame, provenance = load_series(
                safe_path(root, "data/" + series), windows, Path(series).stem
            )
        except SourceValidationError as error:
            runs.append(
                {
                    "series": series,
                    "status": "source_validation_rejected",
                    "reason": str(error),
                    "diagnostics": error.diagnostics,
                    "metrics": None,
                    "policy": "No silent correction, metric-based exclusion or replacement",
                }
            )
            continue
        train, cal, test = temporal_split(frame)
        detector = Detector().fit(
            train,
            cal,
            ["sensor_value"],
            plan["calibration_alert_rate"],
            plan["seed"],
            plan["rolling_window"],
        )
        scored = detector.predict(test)
        quality = QualityMonitor().fit(train, cal, ["sensor_value"])
        flags = quality.predict(test)
        for column in flags:
            if isinstance(column, str) and column.startswith("quality_"):
                scored[column] = flags[column]
        scored.to_csv(output / f"{Path(series).stem}-scored.csv", index=False)
        # Disclose windows intersecting/truncated at test boundaries without moving them.
        test_windows = []
        import pandas as pd

        for start, end in windows:
            a, b = pd.Timestamp(start), pd.Timestamp(end)
            a = a.tz_localize("UTC") if a.tzinfo is None else a.tz_convert("UTC")
            b = b.tz_localize("UTC") if b.tzinfo is None else b.tz_convert("UTC")
            if b >= test.timestamp.min() and a <= test.timestamp.max():
                test_windows.append(
                    {
                        "published_start": str(a),
                        "published_end": str(b),
                        "clipped_at_test_boundary": bool(
                            a < test.timestamp.min() or b > test.timestamp.max()
                        ),
                    }
                )
        runs.append(
            {
                "series": series,
                "status": "evaluated",
                "provenance": provenance,
                "splits": split_manifest((train, cal, test)),
                "window_membership_counts": {
                    name: {"rows": len(part), "positive_rows": int(part.label.sum())}
                    for name, part in (("train", train), ("calibration", cal), ("test", test))
                },
                "thresholds": detector.thresholds,
                "test_windows": test_windows,
                "metrics": metrics(scored),
                "unlabeled_metrics": metrics(scored.drop(columns="label")),
                "events": {
                    name: event_diagnostics(scored, name, plan["rolling_window"])
                    for name in DETECTORS
                },
                "quality_summary": quality_summary(flags, ["sensor_value"]),
                "quality_rules": asdict(quality.rules),
                "scored_csv_sha256": hashlib.sha256(
                    (output / f"{Path(series).stem}-scored.csv").read_bytes()
                ).hexdigest(),
            }
        )
    evaluated = [run for run in runs if run["metrics"] is not None]
    summary = {
        "series_count": len(runs),
        "evaluated_series": len(evaluated),
        "source_rejections": len(runs) - len(evaluated),
        "two_class_test_series": sum(
            run["metrics"]["mad"]["positives"] > 0 and run["metrics"]["mad"]["negatives"] > 0
            for run in evaluated
        ),
        "warning": "No pooled metric or ranking; inspect per-series support and label-window semantics",
    }
    sources = [
        Path(__file__),
        *sorted((Path(__file__).resolve().parents[1] / "src/opsanomaly").glob("*.py")),
    ]
    report = {
        "schema_version": 1,
        "scope": "real-data NAB convenience subset, local point-window evaluation; NOT official NAB scoring",
        "protocol": plan,
        "protocol_sha256": hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
        "source_lock": lock,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "versions": {
            n: importlib.metadata.version(n)
            for n in ("numpy", "pandas", "scikit-learn", "opsanomaly-ai")
        },
        "runs": runs,
        "descriptive_summary": summary,
        "limitations": [
            "Publisher-released real data; full independent provenance and source clock timezone not verified",
            "MIT is publisher repository declaration, not an invented per-file data license; raw files not redistributed",
            "Broad window membership is not per-point fault truth; no official NAB score or leaderboard comparison",
            "Known public convenience subset; source research exposed some label metadata; not a blind trial",
            "60/20/20 split may contain anomalies in training/calibration or no final-test positives; no filtering or exclusion",
            "One predeclared model seed, three series; no confidence intervals, parameter search or generalization claim",
            "No independent quality-rule target exists; quality outputs descriptive only",
            "Once inspected these test periods cannot be reused as untouched final validation for tuned models",
        ],
    }
    (output / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", default="configs/nab_protocol.json")
    p.add_argument("--data", default="data/external/nab")
    p.add_argument("--lock", default="configs/nab_source_lock.json")
    p.add_argument("--output", default="artifacts/nab")
    a = p.parse_args()
    result = evaluate(
        json.loads(Path(a.protocol).read_text(encoding="utf-8")),
        Path(a.data),
        json.loads(Path(a.lock).read_text(encoding="utf-8")),
        Path(a.output),
    )
    print(
        json.dumps(
            {"series": len(result["runs"]), "output": str(Path(a.output) / "results.json")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
