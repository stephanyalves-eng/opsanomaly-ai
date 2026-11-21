"""Train locally and preserve immutable split membership and metadata."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import joblib
import yaml
from opsanomaly.features import temporal_split
from opsanomaly.ingest import read_csv
from opsanomaly.inference import Detector, ARTIFACT_SCHEMA
from opsanomaly.protocol import SPLIT_PROTOCOL, split_manifest, sha256


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/synthetic.csv")
    p.add_argument("--output", default="artifacts")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--alert-rate", type=float, default=None)
    p.add_argument("--window", type=int, default=None)
    a = p.parse_args()
    config = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    seed = a.seed if a.seed is not None else int(config["seed"])
    rate = a.alert_rate if a.alert_rate is not None else float(config["calibration_alert_rate"])
    window = a.window if a.window is not None else int(config["rolling_window"])
    df, report = read_csv(a.data)
    train, calibration, test = temporal_split(df)
    detector = Detector().fit(train, calibration, report["sensors"], rate, seed, window)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(detector, out / "model.joblib")
    metadata = {
        "schema_version": ARTIFACT_SCHEMA,
        "split_protocol": SPLIT_PROTOCOL,
        "model_sha256": sha256(out / "model.joblib"),
        "rolling_window": window,
        "threshold_method": "higher calibration quantile; strict greater-than",
        "threshold_labels_used": False,
        "calibration_label_diagnostics": {
            "rows": len(calibration),
            "injected_or_positive_rows": int(calibration.label.sum())
            if "label" in calibration
            else None,
        },
        "seed": seed,
        "quality": report,
        "splits": split_manifest((train, calibration, test)),
        "dataset_sha256": hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),
        "thresholds": detector.thresholds,
        "calibration_alert_rate": rate,
        "versions": {n: importlib.metadata.version(n) for n in ["numpy", "pandas", "scikit-learn"]},
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
