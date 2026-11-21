"""Execute the predeclared paired stress suite with unchanged detector parameters."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
from typing import Any
import numpy as np
import pandas as pd
from opsanomaly.features import temporal_split
from opsanomaly.inference import Detector, DETECTORS, metrics
from opsanomaly.ingest import validate
from opsanomaly.protocol import SPLIT_PROTOCOL, split_manifest
from opsanomaly.stress import SCENARIOS, SENSORS, generate_stress, event_diagnostics


def execute(protocol: dict[str, Any], output: Path, export_csv: bool = False) -> dict[str, Any]:
    if protocol["protocol_version"] != "synthetic-stress-v1":
        raise ValueError("Unsupported protocol")
    seeds = protocol["seeds"]
    scenarios = protocol["scenarios"]
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or not scenarios
        or len(set(scenarios)) != len(scenarios)
    ):
        raise ValueError("Seeds and scenarios must be nonempty and unique")
    if any(s not in SCENARIOS for s in scenarios):
        raise ValueError("Unknown stress scenario")
    output.mkdir(parents=True, exist_ok=True)
    runs = []

    def generate(scenario: str, seed: int) -> pd.DataFrame:
        return validate(
            generate_stress(
                scenario,
                seed,
                protocol["rows_per_asset"],
                protocol["assets"],
                protocol["random_missing_probability"],
                protocol["event_start_fraction"],
                protocol["event_end_fraction"],
            )
        )[0]

    for seed in seeds:
        clean = generate("clean", seed)
        train, cal, _ = temporal_split(clean)
        started = time.perf_counter()
        detector = Detector().fit(
            train,
            cal,
            SENSORS,
            protocol["calibration_alert_rate"],
            seed,
            protocol["rolling_window"],
        )
        fit_seconds = time.perf_counter() - started
        original_thresholds = detector.thresholds.copy()
        for scenario in scenarios:
            frame = generate(scenario, seed)
            parts = temporal_split(frame)
            # Executable pairing invariant: every case shares exact clean fit periods.
            pd.testing.assert_frame_equal(train, parts[0])
            pd.testing.assert_frame_equal(cal, parts[1])
            test = parts[2]
            started = time.perf_counter()
            scored = detector.predict(test)
            seconds = time.perf_counter() - started
            if detector.thresholds != original_thresholds:
                raise RuntimeError("Threshold mutation during stress evaluation")
            missing_rows = test[SENSORS].isna().any(axis=1).to_numpy()
            reports = metrics(scored)
            missingness = {
                "rows_with_missing": int(missing_rows.sum()),
                "sensor_values_missing": int(test[SENSORS].isna().sum().sum()),
                "positive_rows_with_missing": int(
                    ((test.label.to_numpy() == 1) & missing_rows).sum()
                ),
                "by_method": {
                    name: {
                        "alerts_on_rows_with_missing": int(
                            scored[f"{name}_alert"].to_numpy()[missing_rows].sum()
                        ),
                        "alert_rate_on_rows_with_missing": float(
                            scored[f"{name}_alert"].to_numpy()[missing_rows].mean()
                        )
                        if missing_rows.any()
                        else None,
                    }
                    for name in DETECTORS
                },
            }
            runs.append(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "dataset_sha256_canonical_csv": hashlib.sha256(
                        frame.to_csv(index=False, lineterminator="\n").encode()
                    ).hexdigest(),
                    "splits": split_manifest(parts),
                    "thresholds": detector.thresholds.copy(),
                    "training_positives": int(train.label.sum()),
                    "calibration_positives": int(cal.label.sum()),
                    "metrics": reports,
                    "unlabeled_metrics": metrics(scored.drop(columns="label")),
                    "by_asset": {
                        str(asset): metrics(g) for asset, g in scored.groupby("asset_id", sort=True)
                    },
                    "events": {
                        name: event_diagnostics(scored, name, protocol["rolling_window"])
                        for name in DETECTORS
                    },
                    "missingness": missingness,
                    "timing": {
                        "shared_seed_fit_seconds": fit_seconds,
                        "batch_inference_seconds": seconds,
                    },
                }
            )
            if export_csv:
                scored.to_csv(output / f"{scenario}-seed-{seed}-scored.csv", index=False)
    summaries = {}
    for scenario in scenarios:
        cases = [r for r in runs if r["scenario"] == scenario]
        summaries[scenario] = {
            name: {
                key: {
                    "mean": float(np.mean([c["metrics"][name][key] for c in cases])),
                    "std_population": float(np.std([c["metrics"][name][key] for c in cases])),
                }
                for key in (
                    "precision",
                    "recall",
                    "f1",
                    "alert_rate",
                    "false_alarms_per_1000_observations",
                )
            }
            for name in DETECTORS
        }
    sources = [
        Path(__file__),
        *sorted((Path(__file__).resolve().parents[1] / "src/opsanomaly").glob("*.py")),
    ]
    report = {
        "schema_version": 1,
        "scope": "original paired synthetic stress injections; no real-world validity claim",
        "protocol": protocol,
        "split_protocol": SPLIT_PROTOCOL,
        "protocol_sha256": hashlib.sha256(
            json.dumps(protocol, sort_keys=True).encode()
        ).hexdigest(),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "versions": {
            n: importlib.metadata.version(n)
            for n in ("numpy", "pandas", "scikit-learn", "opsanomaly-ai")
        },
        "runs": runs,
        "by_scenario_equal_weight_seed_summary": summaries,
        "summary_warning": "No overall ranking: control F1 is undefined/zero-denominator and must not be averaged into anomaly comparisons",
        "limitations": [
            "Synthetic label policy is a declared convention, not fault diagnosis",
            "Only three predeclared seeds; no confidence intervals, parameter search or significance claim",
            "Clean fit periods differ from the earlier contaminated spike benchmark; compare within this suite only",
            "Event hits never replace raw point recall; short alert bursts can hide poor sustained coverage",
            "Missingness is imputed, not a dedicated anomaly feature; outages may be invisible",
            "Timings reuse seed fit across scenarios; do not sum repeated fit_seconds fields",
        ],
    }
    (output / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", default="configs/stress_protocol.json")
    p.add_argument("--output", default="artifacts/stress")
    p.add_argument("--export-csv", action="store_true")
    args = p.parse_args()
    report = execute(
        json.loads(Path(args.protocol).read_text(encoding="utf-8")),
        Path(args.output),
        args.export_csv,
    )
    print(
        json.dumps(
            {"runs": len(report["runs"]), "output": str(Path(args.output) / "results.json")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
