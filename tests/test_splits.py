import numpy as np
import pandas as pd
import pytest
from opsanomaly.features import temporal_split, Preprocessor
from opsanomaly.synthetic import generate


def test_chronological_disjoint_and_complete(frame):
    train, cal, test = temporal_split(frame)
    assert len(train) + len(cal) + len(test) == len(frame)
    assert not (set(train.index) & set(cal.index) | set(train.index) & set(test.index))
    for asset in frame.asset_id.unique():
        a, b, c = [p[p.asset_id == asset].timestamp for p in (train, cal, test)]
        assert a.max() < b.min() < c.min()
        assert b.max() < c.min()


def test_preprocessing_never_fits_test(frame):
    train, _, test = temporal_split(frame)
    pre = Preprocessor().fit(train, ["temperature", "vibration"])
    medians = pre.medians.copy()
    test = test.copy()
    test["temperature"] = 1e10
    pre.transform(test)
    np.testing.assert_array_equal(pre.medians, medians)
    np.testing.assert_array_equal(pre.medians, train[["temperature", "vibration"]].median())


def test_all_missing_training_rejected(frame):
    frame["temperature"] = np.nan
    with pytest.raises(ValueError, match="entirely missing"):
        Preprocessor().fit(frame, ["temperature"])


def test_small_asset(frame):
    with pytest.raises(ValueError, match=">=15"):
        temporal_split(frame.iloc[:10])


def test_synthetic_deterministic():
    pd.testing.assert_frame_equal(generate(42), generate(42))
    assert not generate(42).equals(generate(43))
