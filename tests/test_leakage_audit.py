"""Adversarial invariance tests for temporal and threshold leakage."""

import copy
import json
import numpy as np
import pandas as pd
import pytest
from opsanomaly.baseline import RollingMAD
from opsanomaly.features import temporal_split, Preprocessor
from opsanomaly.ingest import validate
from opsanomaly.inference import Detector, metrics
from opsanomaly.synthetic import generate

SENSORS = ["temperature", "vibration"]


def test_shared_cutoffs_on_staggered_assets():
    frame = validate(generate(rows=100, assets=2))[0]
    frame.loc[frame.asset_id == "asset-1", "timestamp"] += pd.Timedelta(minutes=10)
    train, cal, test = temporal_split(frame)
    assert train.timestamp.max() < cal.timestamp.min()
    assert cal.timestamp.max() < test.timestamp.min()
    # Equal row-ratio splitting would incorrectly train B through minute 69
    # while calibrating A from minute 60.
    assert train.timestamp.max() < frame.timestamp.min() + pd.Timedelta(minutes=66)
    assert len(train) + len(cal) + len(test) == len(frame)
    assert all(set(p.asset_id) == set(frame.asset_id) for p in (train, cal, test))
    detector = Detector().fit(train, cal, SENSORS)
    assert len(detector.predict(test)) == len(test)


def test_nonoverlapping_assets_rejected():
    frame = validate(generate(rows=100, assets=2))[0]
    frame.loc[frame.asset_id == "asset-1", "timestamp"] += pd.Timedelta(days=2)
    with pytest.raises(ValueError, match="timelines incompatible"):
        temporal_split(frame)


def test_same_utc_time_never_crosses_partitions(frame):
    train, cal, test = temporal_split(frame)
    assert set(train.timestamp).isdisjoint(cal.timestamp)
    assert set(cal.timestamp).isdisjoint(test.timestamp)
    assert set(train.timestamp).isdisjoint(test.timestamp)
    for period in (train, cal, test):
        assert period.timestamp.value_counts().eq(3).all()


def test_raw_split_rejects_duplicate_time_and_unsorted_rows(frame):
    invalid = frame.copy()
    invalid.loc[1, "timestamp"] = invalid.loc[0, "timestamp"]
    with pytest.raises(ValueError, match="Duplicate"):
        temporal_split(invalid)
    with pytest.raises(ValueError, match="increase"):
        temporal_split(frame.iloc[::-1])


def test_fit_rejects_cross_asset_future_even_if_each_asset_ordered(frame):
    train, cal, _ = temporal_split(frame)
    train = train.copy()
    cal = cal.copy()
    train.loc[train.asset_id == "asset-1", "timestamp"] += pd.Timedelta(days=1)
    cal.loc[cal.asset_id == "asset-1", "timestamp"] += pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="across assets"):
        Detector().fit(train, cal, SENSORS)


@pytest.mark.parametrize("period", ["train", "calibration"])
def test_all_label_values_irrelevant_to_fit(frame, period):
    train, cal, test = temporal_split(frame)
    reference = Detector().fit(train, cal, SENSORS)
    changed = (train if period == "train" else cal).copy()
    changed["label"] = 1 - changed.label
    altered = Detector().fit(
        changed if period == "train" else train,
        changed if period == "calibration" else cal,
        SENSORS,
    )
    assert reference.thresholds == altered.thresholds
    np.testing.assert_array_equal(reference.pre.medians, altered.pre.medians)
    np.testing.assert_array_equal(reference.pre.scale, altered.pre.scale)
    pd.testing.assert_frame_equal(reference.predict(test), altered.predict(test))


def test_calibration_changes_do_not_refit_imputer_scale_or_forest(frame):
    train, cal, test = temporal_split(frame)
    reference = Detector().fit(train, cal, SENSORS)
    changed = cal.copy()
    changed[SENSORS] += 1000
    altered = Detector().fit(train, changed, SENSORS)
    np.testing.assert_array_equal(reference.pre.medians, altered.pre.medians)
    np.testing.assert_array_equal(reference.pre.scale, altered.pre.scale)
    np.testing.assert_array_equal(
        reference.scores(test)["isolation_forest"], altered.scores(test)["isolation_forest"]
    )
    assert reference.thresholds != altered.thresholds


def test_threshold_matches_calibration_only_higher_quantile(frame):
    train, cal, _ = temporal_split(frame)
    detector = Detector().fit(train, cal, SENSORS, rate=0.1)
    calibration_scores = detector._scores(cal, detector.train_history)
    for name, values in calibration_scores.items():
        assert detector.thresholds[name] == float(np.quantile(values, 0.9, method="higher"))


def test_test_labels_never_change_scores_and_future_features_never_change_past(frame):
    train, cal, test = temporal_split(frame)
    detector = Detector().fit(train, cal, SENSORS)
    before = copy.deepcopy(detector.__dict__)
    reference = detector.predict(test)
    changed = test.copy()
    changed["label"] = 1 - changed.label
    only_labels = detector.predict(changed)
    for name in detector.thresholds:
        np.testing.assert_array_equal(reference[f"{name}_score"], only_labels[f"{name}_score"])
    # Change the second half within every asset, retaining the first half.
    earlier = np.zeros(len(test), dtype=bool)
    for positions in test.groupby("asset_id").indices.values():
        earlier[positions[:10]] = True
    changed.loc[changed.index[~earlier], SENSORS] = 1e6
    altered = detector.predict(changed)
    for name in detector.thresholds:
        np.testing.assert_array_equal(
            reference[f"{name}_score"].to_numpy()[earlier],
            altered[f"{name}_score"].to_numpy()[earlier],
        )
    assert detector.thresholds == before["thresholds"]
    np.testing.assert_array_equal(detector.pre.medians, before["pre"].medians)
    np.testing.assert_array_equal(detector.pre.scale, before["pre"].scale)
    for asset in detector.inference_history:
        np.testing.assert_array_equal(
            detector.inference_history[asset], before["inference_history"][asset]
        )
    pd.testing.assert_frame_equal(reference, detector.predict(test))


def test_rolling_exact_prior_window_and_no_current_point_self_inclusion():
    raw = generate(rows=30, assets=1).drop(columns="label")
    raw["temperature"] = np.arange(30, dtype=float)
    raw["vibration"] = np.arange(30, dtype=float)
    frame = validate(raw)[0]
    train, batch = frame.iloc[:10], frame.iloc[10:]
    pre = Preprocessor().fit(train, SENSORS)
    rolling = RollingMAD(3)
    deviations = rolling.deviations(batch, pre, rolling.history(train, pre))
    np.testing.assert_allclose(deviations, np.full((20, 2), 2 / 1.4826))
    changed = batch.copy()
    changed.loc[changed.index[0], "temperature"] = 100
    altered = rolling.deviations(changed, pre, rolling.history(train, pre))
    assert altered[0, 0] == pytest.approx((100 - 8) / 1.4826)


def test_rolling_never_pools_asset_history(frame):
    train, cal, test = temporal_split(frame)
    detector = Detector().fit(train, cal, SENSORS)
    original = detector.predict(test)
    changed = test.copy()
    changed.loc[changed.asset_id == "asset-1", SENSORS] += 1000
    scored = detector.predict(changed)
    asset = original.asset_id == "asset-0"
    np.testing.assert_array_equal(
        original.loc[asset, "rolling_mad_score"], scored.loc[asset, "rolling_mad_score"]
    )


@pytest.mark.parametrize("bad", [0, 2, 1.5, True, 1001])
def test_invalid_rolling_window(bad):
    with pytest.raises(ValueError, match="window"):
        RollingMAD(bad)


@pytest.mark.parametrize("kind", ["train", "calibration", "unordered", "unknown_asset", "schema"])
def test_predict_guards_against_retrospective_or_invalid_context(frame, kind):
    train, cal, test = temporal_split(frame)
    detector = Detector().fit(train, cal, SENSORS)
    invalid = {"train": train, "calibration": cal, "unordered": test.iloc[::-1]}.get(
        kind, test.copy()
    )
    if kind == "unknown_asset":
        invalid["asset_id"] = "unseen"
    if kind == "schema":
        invalid = invalid.drop(columns="vibration")
    with pytest.raises(ValueError):
        detector.predict(invalid)


def test_explanation_is_explicitly_descriptive_not_forest_attribution(frame):
    train, cal, test = temporal_split(frame)
    scored = Detector().fit(train, cal, SENSORS).predict(test)
    assert scored.evidence.str.contains("not causality or model attribution").all()
    assert scored.rolling_mad_evidence.str.contains("not causality or model attribution").all()
    assert scored.isolation_forest_evidence.str.contains("NOT forest attribution").all()
    assert scored.threshold_source.str.contains("fixed before this batch").all()


def test_missing_values_use_training_medians_not_future_values(frame):
    train, cal, test = temporal_split(frame)
    detector = Detector().fit(train, cal, SENSORS)
    test = test.copy()
    test.loc[test.index[0], "temperature"] = np.nan
    actual = detector.predict(test)
    explicit = test.copy()
    explicit.loc[explicit.index[0], "temperature"] = detector.pre.medians[0]
    expected = detector.predict(explicit)
    for name in detector.thresholds:
        np.testing.assert_array_equal(actual[f"{name}_score"], expected[f"{name}_score"])


def test_serializable_unlabeled_metrics_never_emit_supervised_fields(frame):
    train, cal, test = temporal_split(frame)
    scored = Detector().fit(train, cal, SENSORS).predict(test.drop(columns="label"))
    report = metrics(scored)
    json.dumps(report, allow_nan=False)
    for entry in report.values():
        assert entry["labeled"] is False
        assert {
            "precision",
            "recall",
            "f1",
            "pr_auc_average_precision",
            "confusion_matrix",
            "false_alarms_per_1000_observations",
        }.isdisjoint(entry)


def test_manual_mad_override_is_exploratory_and_never_crosses_model_scales(frame):
    train, cal, test = temporal_split(frame)
    detector = Detector().fit(train, cal, SENSORS)
    default = detector.predict(test)
    override = detector.predict(test, threshold=-1)
    assert override.mad_alert.all()
    for name in ("rolling_mad", "isolation_forest"):
        np.testing.assert_array_equal(default[f"{name}_alert"], override[f"{name}_alert"])
    assert metrics(override)["mad"]["threshold_sources"] == ["manual MAD override; exploratory"]
