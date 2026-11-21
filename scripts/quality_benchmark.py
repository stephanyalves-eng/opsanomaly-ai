"""New-seed validation of separate quality rules; preserve numeric model decisions."""

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from opsanomaly.features import temporal_split
from opsanomaly.ingest import validate
from opsanomaly.inference import Detector, metrics
from opsanomaly.protocol import split_manifest
from opsanomaly.quality import QualityMonitor, QualityRules, quality_summary
from opsanomaly.quality_scenarios import QUALITY_SCENARIOS, generate_quality_case
from opsanomaly.stress import SENSORS, event_diagnostics


def intervention_metrics(
    target: np.ndarray[Any, Any], alerts: np.ndarray[Any, Any]
) -> dict[str, Any]:
    """Point counts against a declared intervention, not model probabilities or diagnosis."""
    tp = int((target & alerts).sum())
    fp = int((~target & alerts).sum())
    fn = int((target & ~alerts).sum())
    tn = int((~target & ~alerts).sum())
    return {
        "observations": len(target),
        "positives": int(target.sum()),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "false_alerts_per_1000_observations": fp * 1000 / len(target),
    }


def execute(plan: dict[str, Any], output: Path, export_csv: bool = False) -> dict[str, Any]:
    if plan["protocol_version"] != "quality-validation-v1":
        raise ValueError("Unsupported quality protocol")
    seeds, scenarios = plan["seeds"], plan["scenarios"]
    if not seeds or len(set(seeds)) != len(seeds) or set(seeds) & set(plan["old_seeds_excluded"]):
        raise ValueError("Seeds must be nonempty, unique and exclude previous development seeds")
    if (
        not scenarios
        or len(set(scenarios)) != len(scenarios)
        or not set(scenarios).issubset(QUALITY_SCENARIOS)
    ):
        raise ValueError("Invalid scenarios")
    rules = QualityRules(**plan["quality_rules"])
    output.mkdir(parents=True, exist_ok=True)
    runs = []

    def generate(scenario: str, seed: int) -> pd.DataFrame:
        return validate(
            generate_quality_case(
                scenario,
                seed,
                plan["rows_per_asset"],
                plan["assets"],
                plan["random_missing_probability"],
                plan["event_start_fraction"],
                plan["event_end_fraction"],
            )
        )[0]

    for seed in seeds:
        clean = temporal_split(generate("clean", seed))
        cache: dict[str, tuple[Detector, QualityMonitor]] = {}
        for scenario in scenarios:
            frame = generate(scenario, seed)
            train, cal, test = temporal_split(frame)
            fit_group = "constant_sensor" if scenario == "constant_sensor" else "paired_clean"
            if fit_group == "paired_clean":
                pd.testing.assert_frame_equal(train, clean[0])
                pd.testing.assert_frame_equal(cal, clean[1])
            if fit_group not in cache:
                cache[fit_group] = (
                    Detector().fit(
                        train,
                        cal,
                        SENSORS,
                        plan["calibration_alert_rate"],
                        seed,
                        plan["rolling_window"],
                    ),
                    QualityMonitor(rules).fit(train, cal, SENSORS),
                )
            detector, quality = cache[fit_group]
            thresholds = detector.thresholds.copy()
            numeric = detector.predict(test)
            flags = quality.predict(test)
            if detector.thresholds != thresholds:
                raise RuntimeError("Numeric thresholds changed")
            missing_truth = test[SENSORS].isna()
            mismatches = {
                s: int(
                    (flags[f"quality_{s}_missing"].to_numpy() != missing_truth[s].to_numpy()).sum()
                )
                for s in SENSORS
            }
            if any(mismatches.values()):
                raise RuntimeError("Raw missingness flags inconsistent")
            repeat_target = (
                test.label.eq(1).to_numpy()
                if scenario == "stuck_sensor"
                else np.zeros(len(test), dtype=bool)
            )
            missing_target = (
                test.label.eq(1).to_numpy()
                if scenario == "missing_block"
                else np.zeros(len(test), dtype=bool)
            )
            quality_reports = {}
            for name, target in (("repeat", repeat_target), ("missing", missing_target)):
                alerts = flags[f"quality_{name}_alert"].to_numpy(dtype=bool)
                episode = test[["timestamp", "asset_id"]].copy()
                episode["label"] = target.astype(int)
                episode["rule_alert"] = alerts
                quality_reports[name] = {
                    "target_definition": "declared stuck injection"
                    if name == "repeat"
                    else "declared block outage",
                    "point_metrics": intervention_metrics(target, alerts),
                    "events": event_diagnostics(
                        episode,
                        "rule",
                        rules.stuck_run_length if name == "repeat" else rules.missing_run_length,
                    ),
                }
            summary = quality_summary(flags, SENSORS)
            summary["informational_missing_consistency"] = {
                "mismatches_by_sensor": mismatches,
                "description": "direct NaN-mask plumbing check; not predictive accuracy",
            }
            runs.append(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "fit_group": fit_group,
                    "dataset_sha256_canonical_csv": hashlib.sha256(
                        frame.to_csv(index=False, lineterminator="\n").encode()
                    ).hexdigest(),
                    "splits": split_manifest((train, cal, test)),
                    "numeric_thresholds": thresholds,
                    "numeric_metrics": metrics(numeric),
                    "numeric_unlabeled_metrics": metrics(numeric.drop(columns="label")),
                    "quality_summary": summary,
                    "quality_intervention_metrics": quality_reports,
                    "quality_unlabeled_summary": quality_summary(
                        flags.drop(columns="label"), SENSORS
                    ),
                    "training_repeat_assessability": quality.assessable,
                    "cadence_seconds": quality.cadence,
                }
            )
            if export_csv:
                combined = numeric.copy()
                for column in flags:
                    if isinstance(column, str) and column.startswith("quality_"):
                        combined[column] = flags[column]
                combined.to_csv(output / f"{scenario}-seed-{seed}-scored.csv", index=False)
    aggregate = {}
    for scenario in scenarios:
        cases = [r for r in runs if r["scenario"] == scenario]
        aggregate[scenario] = {
            "quality_review_alert_rate_mean": float(
                np.mean([r["quality_summary"]["review_alert_rate"] for r in cases])
            ),
            "repeat_injection_recall_mean": float(
                np.mean(
                    [
                        r["quality_intervention_metrics"]["repeat"]["point_metrics"]["recall"]
                        for r in cases
                    ]
                )
            )
            if scenario == "stuck_sensor"
            else None,
            "block_outage_recall_mean": float(
                np.mean(
                    [
                        r["quality_intervention_metrics"]["missing"]["point_metrics"]["recall"]
                        for r in cases
                    ]
                )
            )
            if scenario == "missing_block"
            else None,
        }
    sources = [
        Path(__file__),
        *sorted((Path(__file__).resolve().parents[1] / "src/opsanomaly").glob("*.py")),
    ]
    report = {
        "schema_version": 1,
        "scope": "separate data-quality rules on predeclared new synthetic seeds; no diagnosis",
        "protocol": plan,
        "rules": asdict(rules),
        "protocol_sha256": hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "versions": {
            n: importlib.metadata.version(n)
            for n in ("numpy", "pandas", "scikit-learn", "opsanomaly-ai")
        },
        "runs": runs,
        "by_scenario": aggregate,
        "limitations": [
            "Same synthetic family, only three new RNG seeds; no real-world validity",
            "Identical measurements cannot distinguish legitimate plateau from injected stuck sensor",
            "Direct missingness checks are not ML performance; quality target differs from operational labels",
            "No OR-combined model decisions or claimed overall ML accuracy gain",
            "Count-based latency, absolute tolerance, training-variability gate and cadence assumptions require domain review",
            "constant_sensor has separate fitting data; do not confuse it with paired_clean fits",
            "Inspected new seeds are development evidence for any future changes",
        ],
    }
    (output / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", default="configs/quality_protocol.json")
    p.add_argument("--output", default="artifacts/quality-validation")
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
