"""Evaluate only held-out rows from the exact training dataset."""

import argparse
import os
import tempfile
import json
import time
from pathlib import Path
import joblib

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "opsanomaly-mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from opsanomaly.features import temporal_split
from opsanomaly.ingest import read_csv
from opsanomaly.inference import metrics, DETECTORS
from opsanomaly.protocol import verify_run


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/synthetic.csv")
    p.add_argument("--artifacts", default="artifacts")
    a = p.parse_args()
    out = Path(a.artifacts)
    metadata = json.loads((out / "metadata.json").read_text())
    df, quality = read_csv(a.data)
    parts = temporal_split(df)
    verify_run(metadata, a.data, out / "model.joblib", parts)
    _, _, test = parts
    detector = joblib.load(out / "model.joblib")  # Trusted local artifacts only.
    start = time.perf_counter()
    scored = detector.predict(test)
    elapsed = time.perf_counter() - start
    scored.to_csv(out / "scored.csv", index=False)
    report = {
        "schema_version": 2,
        "scope": "held-out point observations; no failure-diagnosis or real-world validity claim",
        "evaluation_mode": "labeled" if "label" in test else "unlabeled",
        "protocol": {
            "split": metadata["split_protocol"],
            "dataset_sha256": metadata["dataset_sha256"],
            "model_sha256": metadata["model_sha256"],
            "thresholds": detector.thresholds,
            "calibration_alert_rate": detector.calibration_rate,
            "threshold_labels_used": False,
            "rolling_window": detector.rolling.window,
            "test_labels_used_for_tuning": False,
            "calibration_label_diagnostics": metadata["calibration_label_diagnostics"],
        },
        "quality": quality,
        "by_asset": {str(asset): metrics(g) for asset, g in scored.groupby("asset_id", sort=True)},
        "metrics": metrics(scored),
        "benchmark": {
            "inference_seconds": elapsed,
            "observations": len(test),
            "rows_per_second": len(test) / elapsed,
        },
    }
    (out / "evaluation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    for ax, name in zip(axes, DETECTORS, strict=True):
        for asset, group in scored.groupby("asset_id"):
            ax.plot(group.timestamp, group[f"{name}_score"], label=str(asset), alpha=0.7)
        ax.axhline(detector.thresholds[name], color="red", label="fixed calibration threshold")
        ax.set(ylabel=f"{name} score", title=f"Held-out {name} (not failure prediction)")
        ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "scores.png", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
