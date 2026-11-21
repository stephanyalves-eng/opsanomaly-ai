import numpy as np
import pandas as pd
from opsanomaly.features import temporal_split
from opsanomaly.inference import Detector


def test_deterministic_roundtrip_labels_and_missing(frame, tmp_path):
    import joblib

    train, cal, test = temporal_split(frame)
    train = train.copy()
    train.loc[train.index[0], "temperature"] = np.nan
    first = Detector().fit(train, cal, ["temperature", "vibration"])
    second = Detector().fit(
        train.drop(columns="label"), cal.drop(columns="label"), ["temperature", "vibration"]
    )
    pd.testing.assert_frame_equal(first.predict(test), second.predict(test))
    path = tmp_path / "model.joblib"
    joblib.dump(first, path)
    pd.testing.assert_frame_equal(first.predict(test), joblib.load(path).predict(test))
    original = first.thresholds.copy()
    test = test.copy()
    test["temperature"] = 1e8
    first.predict(test)
    assert first.thresholds == original


def test_constant_sensors_and_missing_inference(frame):
    frame["temperature"] = 5.0
    frame["vibration"] = 5.0
    train, cal, test = temporal_split(frame)
    model = Detector().fit(train, cal, ["temperature", "vibration"])
    scored = model.predict(test)
    assert not scored.mad_alert.any()
    assert np.isfinite(scored.mad_score).all()
    test = test.copy()
    test["temperature"] = np.nan
    assert np.isfinite(model.predict(test).mad_score).all()
    assert model.predict(test, threshold=-1).mad_alert.all()


def test_fit_rejects_overlap_and_label_features(frame):
    import pytest

    train, cal, _ = temporal_split(frame)
    with pytest.raises(ValueError, match="without overlap"):
        Detector().fit(train, train, ["temperature", "vibration"])
    with pytest.raises(ValueError, match="cannot be features"):
        Detector().fit(train, cal, ["label"])


def test_inference_rejects_infinity(frame):
    import pytest

    train, cal, test = temporal_split(frame)
    detector = Detector().fit(train, cal, ["temperature", "vibration"])
    test = test.copy()
    test["temperature"] = np.inf
    with pytest.raises(ValueError, match="Nonfinite"):
        detector.predict(test)
