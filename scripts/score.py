"""Score a new CSV with a trusted local fitted detector, without refitting."""

import argparse
import json
from pathlib import Path
import joblib
from opsanomaly.ingest import read_csv
from opsanomaly.inference import metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="artifacts/model.joblib", help="Trusted local model only")
    p.add_argument("--output", default="artifacts/new")
    a = p.parse_args()
    frame, quality = read_csv(a.data)
    detector = joblib.load(a.model)
    if getattr(detector, "artifact_schema", None) != 2:
        raise ValueError("Unsupported artifact schema; retrain")
    if quality["sensors"] != detector.pre.sensors:
        raise ValueError("Sensor schema/order differs from training")
    scored = detector.predict(frame)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out / "scored.csv", index=False)
    (out / "evaluation.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "scope": "future batch replay from calibration history; not an independent benchmark",
                "evaluation_mode": "labeled" if "label" in scored else "unlabeled",
                "quality": quality,
                "thresholds": detector.thresholds,
                "metrics": metrics(scored),
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
