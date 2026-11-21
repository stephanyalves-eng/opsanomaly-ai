import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
from opsanomaly.features import temporal_split
from opsanomaly.ingest import validate
from opsanomaly.inference import Detector, metrics
from opsanomaly.stress import SCENARIOS, SENSORS, generate_stress, event_diagnostics


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_deterministic_paired_clean_fit_periods_and_no_mutation(scenario):
    frame = validate(generate_stress(scenario, rows=100, assets=2))[0]
    pd.testing.assert_frame_equal(frame, validate(generate_stress(scenario, rows=100, assets=2))[0])
    clean = validate(generate_stress("clean", rows=100, assets=2))[0]
    parts = temporal_split(frame)
    nominal_parts = temporal_split(clean)
    pd.testing.assert_frame_equal(parts[0], nominal_parts[0])
    pd.testing.assert_frame_equal(parts[1], nominal_parts[1])
    assert parts[0].label.sum() == parts[1].label.sum() == 0
    assert not parts[0][SENSORS].isna().any().any()
    assert not parts[1][SENSORS].isna().any().any()
    assert frame.timestamp.equals(clean.timestamp)
    assert frame.asset_id.equals(clean.asset_id)


@pytest.mark.parametrize(
    "scenario", ["gradual_drift", "sustained_shift", "stuck_sensor", "missing_block"]
)
def test_declared_event_labels_and_exact_boundaries(scenario):
    frame = generate_stress(scenario, assets=1)
    assert frame.label.sum() == 120
    assert frame.label.iloc[819] == 0
    assert frame.label.iloc[820:940].eq(1).all()
    assert frame.label.iloc[940] == 0
    clean = generate_stress("clean", assets=1)
    pd.testing.assert_frame_equal(frame.iloc[:820], clean.iloc[:820])
    pd.testing.assert_frame_equal(frame.iloc[940:], clean.iloc[940:])
    if scenario == "stuck_sensor":
        assert frame.temperature.iloc[820:940].eq(frame.temperature.iloc[819]).all()
        np.testing.assert_array_equal(frame.vibration, clean.vibration)
    if scenario == "missing_block":
        assert frame[SENSORS].iloc[820:940].isna().all().all()


def test_gradual_drift_labels_include_unobservable_onset_and_benign_control():
    clean = generate_stress("clean", assets=1)
    gradual = generate_stress("gradual_drift", assets=1)
    benign = generate_stress("benign_drift", assets=1)
    assert gradual.label.iloc[820] == 1
    assert gradual.temperature.iloc[820] == clean.temperature.iloc[820]
    assert gradual.temperature.iloc[939] - clean.temperature.iloc[939] == pytest.approx(12)
    assert benign.temperature.iloc[939] - clean.temperature.iloc[939] == pytest.approx(1.5)
    assert benign.label.sum() == 0


def test_random_missingness_is_paired_and_does_not_erase_spike_labels():
    missing = generate_stress("missing_random", assets=1, missing_probability=1)
    combined = generate_stress("spikes_with_missing", assets=1, missing_probability=1)
    observed = generate_stress("spikes_with_missing", assets=1, missing_probability=0)
    assert missing.label.sum() == 0
    assert combined.label.sum() > 0
    np.testing.assert_array_equal(combined.label, observed.label)
    assert combined[SENSORS].iloc[800:].isna().all().all()
    assert not combined[SENSORS].iloc[:800].isna().any().any()
    pd.testing.assert_frame_equal(missing[SENSORS].isna(), combined[SENSORS].isna())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scenario": "unknown"},
        {"rows": 99},
        {"assets": 0},
        {"seed": -1},
        {"missing_probability": 1.1},
        {"start_fraction": 0.7},
        {"start_fraction": 0.94, "end_fraction": 0.82},
    ],
)
def test_invalid_stress_generation_rejected(kwargs):
    arguments = {"scenario": "clean", **kwargs}
    with pytest.raises(ValueError):
        generate_stress(**arguments)


def test_event_hit_never_expands_point_alerts_and_miss_is_explicit():
    frame = validate(generate_stress("sustained_shift", assets=1, rows=100))[0].iloc[80:].copy()
    frame["mad_alert"] = False
    frame.loc[frame.index[2], "mad_alert"] = True
    before = frame.mad_alert.copy()
    result = event_diagnostics(frame, "mad", 3)
    assert result["event_count"] == result["events_hit"] == 1
    event = result["events"][0]
    assert event["observations"] == 12
    assert event["first_alert_delay_observations"] == 0
    assert event["point_coverage"] == pytest.approx(1 / 12)
    assert result["early_point_recall"] == pytest.approx(1 / 3)
    assert result["late_point_recall"] == 0
    pd.testing.assert_series_equal(before, frame.mad_alert)
    frame["mad_alert"] = False
    missed = event_diagnostics(frame, "mad", 3)["events"][0]
    assert missed["hit"] is False
    assert missed["first_alert_delay_observations"] is None


def test_same_fitted_detector_replays_cases_without_threshold_or_history_change():
    clean = validate(generate_stress("clean", rows=100, assets=1))[0]
    train, cal, test = temporal_split(clean)
    detector = Detector().fit(train, cal, SENSORS)
    thresholds = detector.thresholds.copy()
    history = {a: x.copy() for a, x in detector.inference_history.items()}
    reference = detector.predict(test)
    for scenario in SCENARIOS:
        future = temporal_split(validate(generate_stress(scenario, rows=100, assets=1))[0])[2]
        scored = detector.predict(future)
        assert all(np.isfinite(scored[f"{name}_score"]).all() for name in thresholds)
        assert detector.thresholds == thresholds
        for asset in history:
            np.testing.assert_array_equal(history[asset], detector.inference_history[asset])
    pd.testing.assert_frame_equal(reference, detector.predict(test))


def test_stress_cli_exports_machine_readable_results_and_preserves_plan(tmp_path):
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "configs/stress_protocol.json").read_text())
    protocol.update(seeds=[42], rows_per_asset=100, assets=1)
    plan = tmp_path / "protocol.json"
    plan.write_text(json.dumps(protocol))
    original = plan.read_bytes()
    out = tmp_path / "report"
    subprocess.run(
        [
            sys.executable,
            "scripts/stress_benchmark.py",
            "--protocol",
            str(plan),
            "--output",
            str(out),
            "--export-csv",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=120,
    )
    result = json.loads((out / "results.json").read_text())
    assert plan.read_bytes() == original
    assert len(result["runs"]) == 8
    thresholds = result["runs"][0]["thresholds"]
    for run in result["runs"]:
        assert run["thresholds"] == thresholds
        assert run["training_positives"] == run["calibration_positives"] == 0
        for method in run["unlabeled_metrics"].values():
            assert {"precision", "recall", "f1", "false_alarms_per_1000_observations"}.isdisjoint(
                method
            )
        exported = pd.read_csv(out / f"{run['scenario']}-seed-42-scored.csv")
        assert len(exported) == 20
        assert (
            metrics(exported)["mad"]["confusion_matrix"]
            == run["metrics"]["mad"]["confusion_matrix"]
        )
    clean = next(r for r in result["runs"] if r["scenario"] == "clean")
    assert clean["metrics"]["mad"]["pr_auc_average_precision"] is None
    assert clean["events"]["mad"]["event_count"] == 0


def test_stress_cli_rejects_duplicate_seeds_before_experiment(tmp_path):
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / "configs/stress_protocol.json").read_text())
    plan["seeds"] = [42, 42]
    source = tmp_path / "invalid.json"
    source.write_text(json.dumps(plan))
    result = subprocess.run(
        [
            sys.executable,
            "scripts/stress_benchmark.py",
            "--protocol",
            str(source),
            "--output",
            str(tmp_path / "report"),
        ],
        cwd=root,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode != 0
    assert b"unique" in result.stderr
    assert not (tmp_path / "report/results.json").exists()
