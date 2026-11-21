"""Fit raw quality context on shared train/calibration periods, export future flags."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from opsanomaly.features import temporal_split
from opsanomaly.ingest import read_csv
from opsanomaly.quality import QualityMonitor, quality_summary
from opsanomaly.protocol import split_manifest, sha256


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--output", default="artifacts/quality")
    args = p.parse_args()
    frame, info = read_csv(args.data)
    train, cal, test = temporal_split(frame)
    monitor = QualityMonitor().fit(train, cal, info["sensors"])
    scored = monitor.predict(test)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output / "quality.csv", index=False)
    report = {
        "schema_version": 1,
        "rules": asdict(monitor.rules),
        "dataset_sha256": sha256(args.data),
        "splits": split_manifest((train, cal, test)),
        "training_repeat_assessability": monitor.assessable,
        "cadence_seconds": monitor.cadence,
        "summary": quality_summary(scored, info["sensors"]),
        "scope": "raw quality rules only; labels are not quality ground truth and are not scored",
    }
    (output / "quality.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(output / "quality.json")


if __name__ == "__main__":
    main()
