import numpy as np
import pytest
from opsanomaly.ingest import validate, read_csv


def test_missing_report(frame):
    frame.loc[0, "temperature"] = np.nan
    _, report = validate(frame)
    assert report["missing"]["temperature"] == 1


@pytest.mark.parametrize("value", [np.inf, -np.inf, 1e13, "broken"])
def test_invalid_numeric(frame, value):
    frame["temperature"] = frame.temperature.astype(object)
    frame.loc[0, "temperature"] = value
    with pytest.raises((ValueError, TypeError)):
        validate(frame)


def test_duplicates(frame):
    frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
    with pytest.raises(ValueError, match="Duplicate"):
        validate(frame)


def test_naive_timezone(frame):
    frame["timestamp"] = frame.timestamp.dt.tz_localize(None)
    with pytest.raises(ValueError, match="timezone"):
        validate(frame)


def test_timezone_equivalent_duplicate(frame):
    frame["timestamp"] = frame.timestamp.astype(str)
    frame.loc[1, "timestamp"] = "2024-12-31T19:00:00-05:00"
    with pytest.raises(ValueError, match="Duplicate"):
        validate(frame)


def test_sampling_and_order(frame):
    frame = frame.drop(index=1)
    assert validate(frame)[1]["irregular_intervals"]["asset-0"] == 1
    with pytest.raises(ValueError, match="increase"):
        validate(frame.iloc[::-1])


def test_csv_round_trip(frame, tmp_path):
    path = tmp_path / "input.csv"
    frame.to_csv(path, index=False)
    assert len(read_csv(path)[0]) == len(frame)


def test_invalid_label_and_empty_asset(frame):
    frame.loc[0, "label"] = 2
    with pytest.raises(ValueError, match="Labels"):
        validate(frame)
    frame.loc[0, "label"] = 0
    frame.loc[0, "asset_id"] = ""
    with pytest.raises(ValueError, match="asset_id"):
        validate(frame)


def test_sensor_count_limit(frame):
    for i in range(65):
        frame[f"extra_{i}"] = 0
    with pytest.raises(ValueError, match="1..64"):
        validate(frame)
