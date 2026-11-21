import copy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
from opsanomaly.features import temporal_split
from opsanomaly.ingest import validate
from opsanomaly.inference import Detector
from opsanomaly.quality import QualityMonitor, QualityRules, quality_summary
from opsanomaly.quality_scenarios import generate_quality_case
from opsanomaly.stress import SENSORS


def dataset(scenario="clean", seed=1051, rows=100, assets=1):
    return validate(generate_quality_case(scenario, seed, rows, assets))[0]


def monitor_and_future():
    train, cal, test = temporal_split(dataset())
    return QualityMonitor().fit(train, cal, SENSORS), test


def test_repeat_threshold_uses_only_current_and_prior_rows():
    monitor, test = monitor_and_future()
    test = test.copy()
    test["temperature"] = 123.0
    scored = monitor.predict(test)
    np.testing.assert_array_equal(scored.quality_temperature_repeat_run, np.arange(1, 21))
    assert not scored.quality_repeat_alert.iloc[:7].any()
    assert scored.quality_repeat_alert.iloc[7:].all()
    assert scored.quality_evidence.iloc[7].find("not fault diagnosis") >= 0


def test_missing_is_raw_immediate_and_persistent_after_three():
    monitor, test = monitor_and_future()
    test = test.copy()
    test[SENSORS] = np.nan
    scored = monitor.predict(test)
    assert scored.quality_missing_any.all()
    assert not scored.quality_missing_alert.iloc[:2].any()
    assert scored.quality_missing_alert.iloc[2:].all()
    assert not scored.quality_repeat_alert.any()
    assert scored.temperature.isna().all()
    assert scored.quality_temperature_repeat_run.eq(0).all()


def test_missing_interrupts_repetition_and_observation_interrupts_missing():
    monitor, test = monitor_and_future()
    test = test.copy()
    test["temperature"] = 123.0
    test.loc[test.index[8:10], "temperature"] = np.nan
    scored = monitor.predict(test)
    np.testing.assert_array_equal(scored.quality_temperature_repeat_run.iloc[8:12], [0, 0, 1, 2])
    np.testing.assert_array_equal(scored.quality_temperature_missing_run.iloc[8:12], [1, 2, 0, 0])
    assert not scored.quality_temperature_missing_alert.any()


def test_fixed_anchor_prevents_chained_tolerance_drift():
    train, cal, test = temporal_split(dataset())
    monitor = QualityMonitor(QualityRules(stuck_run_length=3, repeat_tolerance=0.1)).fit(
        train, cal, SENSORS
    )
    test = test.copy()
    test["temperature"] = 100 + np.arange(len(test)) * 0.09
    scored = monitor.predict(test)
    assert not scored.quality_temperature_repeat_alert.any()
    assert scored.quality_temperature_repeat_run.max() <= 2


def test_time_gap_resets_both_types_of_runs():
    monitor, test = monitor_and_future()
    test = test.copy()
    test["temperature"] = 123.0
    test["vibration"] = np.nan
    test.loc[test.index[10:], "timestamp"] += pd.Timedelta(hours=1)
    scored = monitor.predict(test)
    assert scored.quality_gap_reset.sum() == 1
    assert scored.quality_temperature_repeat_run.iloc[10] == 1
    assert scored.quality_vibration_missing_run.iloc[10] == 1
    assert not scored.quality_review_alert.iloc[10]


def test_training_constant_not_mislabeled_as_stuck():
    train, cal, test = temporal_split(dataset("constant_sensor"))
    monitor = QualityMonitor().fit(train, cal, SENSORS)
    scored = monitor.predict(test)
    assert not monitor.assessable["asset-0"]["temperature"]
    assert not scored.quality_temperature_repeat_assessable.any()
    assert not scored.quality_temperature_repeat_alert.any()
    assert scored.quality_evidence.str.contains("not assessable").all()


def test_calibration_cannot_enable_training_constant_gate():
    train, cal, test = temporal_split(dataset("constant_sensor"))
    cal = cal.copy()
    cal["temperature"] = np.arange(len(cal))
    monitor = QualityMonitor().fit(train, cal, SENSORS)
    assert not monitor.assessable["asset-0"]["temperature"]
    assert not monitor.predict(test).quality_temperature_repeat_alert.any()


def test_legitimate_plateau_is_observationally_indistinguishable_from_stuck():
    stuck = dataset("stuck_sensor", rows=1000)
    legitimate = dataset("legitimate_plateau", rows=1000)
    pd.testing.assert_frame_equal(stuck.drop(columns="label"), legitimate.drop(columns="label"))
    train, cal, test = temporal_split(stuck)
    other = temporal_split(legitimate)[2]
    monitor = QualityMonitor().fit(train, cal, SENSORS)
    first = monitor.predict(test).filter(regex="^quality_")
    second = monitor.predict(other).filter(regex="^quality_")
    pd.testing.assert_frame_equal(first, second)
    assert first.quality_repeat_alert.any()
    assert other.label.sum() == 0


def test_quantized_control_preserves_paired_fit_periods():
    clean = temporal_split(dataset())
    quantized = temporal_split(dataset("quantized_clean"))
    pd.testing.assert_frame_equal(clean[0], quantized[0])
    pd.testing.assert_frame_equal(clean[1], quantized[1])
    np.testing.assert_allclose(
        quantized[2].temperature * 10, (quantized[2].temperature * 10).round()
    )


def test_predict_is_immutable_label_independent_and_no_future_leakage():
    train, cal, test = temporal_split(dataset(assets=2))
    monitor = QualityMonitor().fit(train, cal, SENSORS)
    history = copy.deepcopy(monitor.history)
    last = monitor.last_times.copy()
    original = monitor.predict(test)
    changed = test.copy()
    changed["label"] = 1 - changed.label
    pd.testing.assert_frame_equal(
        original.filter(regex="^quality_"), monitor.predict(changed).filter(regex="^quality_")
    )
    positions = np.zeros(len(test), dtype=bool)
    for group in test.groupby("asset_id").indices.values():
        positions[group[10:]] = True
    changed.loc[changed.index[positions], SENSORS] = np.nan
    altered = monitor.predict(changed)
    pd.testing.assert_frame_equal(
        original.filter(regex="^quality_").iloc[~positions],
        altered.filter(regex="^quality_").iloc[~positions],
    )
    assert history == monitor.history and last == monitor.last_times
    pd.testing.assert_frame_equal(original, monitor.predict(test))


def test_quality_never_changes_numeric_models_or_raw_data():
    train, cal, test = temporal_split(dataset("missing_block"))
    numeric = Detector().fit(train, cal, SENSORS)
    reference = numeric.predict(test)
    before = test.copy(deep=True)
    flags = QualityMonitor().fit(train, cal, SENSORS).predict(test)
    pd.testing.assert_frame_equal(test, before)
    pd.testing.assert_frame_equal(numeric.predict(test), reference)
    assert flags.quality_missing_any.any()
    assert flags.temperature.isna().sum() == test.temperature.isna().sum()


def test_asset_histories_and_assessability_are_separate():
    train, cal, test = temporal_split(dataset(assets=2))
    train = train.copy()
    cal = cal.copy()
    train.loc[train.asset_id == "asset-1", "temperature"] = 100.0
    cal.loc[cal.asset_id == "asset-1", "temperature"] = 100.0
    monitor = QualityMonitor().fit(train, cal, SENSORS)
    test = test.copy()
    test["temperature"] = 100.0
    scored = monitor.predict(test)
    assert (
        scored.loc[scored.asset_id == "asset-0", "quality_temperature_repeat_alert"].iloc[7:].all()
    )
    assert not scored.loc[scored.asset_id == "asset-1", "quality_temperature_repeat_alert"].any()


def test_unlabeled_summary_has_no_quality_performance_claim():
    monitor, test = monitor_and_future()
    scored = monitor.predict(test.drop(columns="label"))
    summary = quality_summary(scored, SENSORS)
    assert {"precision", "recall", "f1", "accuracy", "confusion_matrix"}.isdisjoint(summary)
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize(
    "rule",
    [
        {"stuck_run_length": 1},
        {"missing_run_length": True},
        {"repeat_tolerance": float("nan")},
        {"repeat_tolerance": -1},
        {"gap_multiplier": 1},
    ],
)
def test_invalid_rules_fail(rule):
    with pytest.raises(ValueError):
        QualityRules(**rule)


def test_rules_are_frozen():
    rules = QualityRules()
    with pytest.raises(FrozenInstanceError):
        rules.stuck_run_length = 2


@pytest.mark.parametrize("kind", ["past", "unknown", "schema", "unordered"])
def test_invalid_inference_rejected(kind):
    train, cal, test = temporal_split(dataset())
    monitor = QualityMonitor().fit(train, cal, SENSORS)
    frame = test.copy()
    if kind == "past":
        frame = cal
    elif kind == "unknown":
        frame["asset_id"] = "new"
    elif kind == "schema":
        frame = frame.drop(columns="vibration")
    else:
        frame = frame.iloc[::-1]
    with pytest.raises(ValueError):
        monitor.predict(frame)


def test_quality_cli_labeled_and_unlabeled_equivalent(tmp_path):
    root = Path(__file__).resolve().parents[1]
    frame = dataset("missing_block")
    reports = []
    for labeled in (True, False):
        source = tmp_path / f"{labeled}.csv"
        out = tmp_path / f"out-{labeled}"
        (frame if labeled else frame.drop(columns="label")).to_csv(source, index=False)
        subprocess.run(
            [
                sys.executable,
                "scripts/quality_score.py",
                "--data",
                str(source),
                "--output",
                str(out),
            ],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=120,
        )
        reports.append(json.loads((out / "quality.json").read_text())["summary"])
        assert pd.read_csv(out / "quality.csv").quality_missing_alert.sum() > 0
    assert reports[0] == reports[1]


def test_new_seed_benchmark_cli_and_old_seed_rejection(tmp_path):
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / "configs/quality_protocol.json").read_text())
    plan.update(seeds=[1051], rows_per_asset=100, assets=1)
    source = tmp_path / "plan.json"
    source.write_text(json.dumps(plan))
    out = tmp_path / "benchmark"
    subprocess.run(
        [
            sys.executable,
            "scripts/quality_benchmark.py",
            "--protocol",
            str(source),
            "--output",
            str(out),
            "--export-csv",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=120,
    )
    report = json.loads((out / "results.json").read_text())
    assert len(report["runs"]) == 11
    runs = {r["scenario"]: r for r in report["runs"]}
    assert runs["stuck_sensor"]["quality_summary"] == runs["legitimate_plateau"]["quality_summary"]
    for run in report["runs"]:
        assert all(
            v == 0
            for v in run["quality_summary"]["informational_missing_consistency"][
                "mismatches_by_sensor"
            ].values()
        )
        for entry in run["numeric_unlabeled_metrics"].values():
            assert "precision" not in entry
        assert (
            run["quality_unlabeled_summary"]["review_alerts"]
            == run["quality_summary"]["review_alerts"]
        )
    plan["seeds"] = [42]
    source.write_text(json.dumps(plan))
    result = subprocess.run(
        [
            sys.executable,
            "scripts/quality_benchmark.py",
            "--protocol",
            str(source),
            "--output",
            str(tmp_path / "invalid"),
        ],
        cwd=root,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode != 0 and b"exclude previous" in result.stderr
