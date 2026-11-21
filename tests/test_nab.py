"""Offline adapter/protocol tests use original tiny fixtures, never external data."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
from opsanomaly.nab import (
    load_series,
    verify_sources,
    safe_path,
    read_labels,
    SourceValidationError,
)
from opsanomaly.features import temporal_split
from opsanomaly.inference import Detector


def source_csv(tmp_path, rows=100):
    path = tmp_path / "original-fixture.csv"
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=rows, freq="min").astype(str),
            "value": np.random.default_rng(7).normal(50, 1, rows),
        }
    )
    frame.to_csv(path, index=False)
    return path, frame


def test_inclusive_windows_and_explicit_utc_assumption(tmp_path):
    path, raw = source_csv(tmp_path)
    windows = [[raw.timestamp.iloc[82], raw.timestamp.iloc[90]]]
    frame, info = load_series(path, windows, "fixture")
    assert frame.label.sum() == 9
    assert frame.label.iloc[82] == frame.label.iloc[90] == 1
    assert frame.label.iloc[81] == frame.label.iloc[91] == 0
    assert str(frame.timestamp.dt.tz) == "UTC"
    assert "unspecified" in info["source_timezone"]
    assert "not verified" in info["timezone_assumption"]
    train, cal, test = temporal_split(frame)
    assert train.timestamp.max() < cal.timestamp.min() < test.timestamp.min()
    assert len(test) == 20


@pytest.mark.parametrize("kind", ["reversed", "overlap", "no_observations", "missing_endpoint"])
def test_invalid_published_windows_rejected(tmp_path, kind):
    path, raw = source_csv(tmp_path)
    a, b = raw.timestamp.iloc[20], raw.timestamp.iloc[30]
    windows = {
        "reversed": [[b, a]],
        "overlap": [[a, b], [b, raw.timestamp.iloc[40]]],
        "no_observations": [["2030-01-01", "2030-01-02"]],
        "missing_endpoint": [[a]],
    }[kind]
    with pytest.raises(ValueError):
        load_series(path, windows, "fixture")


def test_missing_entry_does_not_become_all_negative(tmp_path):
    (tmp_path / "labels.json").write_text(json.dumps({"other": []}))
    with pytest.raises(ValueError, match="No published label"):
        read_labels(tmp_path, "labels.json", "chosen")


def test_no_labels_entry_empty_windows_is_explicit_negative_fixture(tmp_path):
    path, _ = source_csv(tmp_path)
    frame, _ = load_series(path, [], "fixture")
    assert frame.label.sum() == 0


@pytest.mark.parametrize("name", ["../private.csv", "/outside.csv", "a/../../b", "a\\..\\b"])
def test_unsafe_paths_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        safe_path(tmp_path, name)


def test_source_hash_and_revision_guards(tmp_path):
    content = b"original fixture"
    (tmp_path / "file.csv").write_bytes(content)
    lock = {
        "repository": "https://github.com/numenta/NAB",
        "revision": "a" * 40,
        "files": {
            "file.csv": {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        },
    }
    verify_sources(tmp_path, lock, "a" * 40)
    with pytest.raises(ValueError, match="revision"):
        verify_sources(tmp_path, lock, "b" * 40)
    (tmp_path / "file.csv").write_bytes(content + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        verify_sources(tmp_path, lock, "a" * 40)


def test_source_duplicates_and_unsorted_rows_are_not_silently_repaired(tmp_path):
    path, raw = source_csv(tmp_path)
    raw.loc[1, "timestamp"] = raw.timestamp.iloc[0]
    raw.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        load_series(path, [], "fixture")
    raw.iloc[::-1].drop_duplicates("timestamp").to_csv(path, index=False)
    with pytest.raises(ValueError, match="increase"):
        load_series(path, [], "fixture")


def test_incorrect_schema_rejected(tmp_path):
    path, raw = source_csv(tmp_path)
    raw["unexpected"] = 0
    raw.to_csv(path, index=False)
    with pytest.raises(ValueError, match="schema"):
        load_series(path, [], "fixture")


def test_window_labels_do_not_change_numeric_fit_or_thresholds(tmp_path):
    path, raw = source_csv(tmp_path)
    labeled, _ = load_series(
        path,
        [
            [raw.timestamp.iloc[10], raw.timestamp.iloc[30]],
            [raw.timestamp.iloc[82], raw.timestamp.iloc[90]],
        ],
        "fixture",
    )
    unlabeled = labeled.drop(columns="label")
    first_parts = temporal_split(labeled)
    second_parts = temporal_split(unlabeled)
    first = Detector().fit(first_parts[0], first_parts[1], ["sensor_value"])
    second = Detector().fit(second_parts[0], second_parts[1], ["sensor_value"])
    assert first.thresholds == second.thresholds
    for name in first.thresholds:
        np.testing.assert_array_equal(
            first.predict(first_parts[2])[f"{name}_score"],
            second.predict(second_parts[2])[f"{name}_score"],
        )


def test_offline_cli_on_original_fixture_and_no_official_score(tmp_path):
    root = Path(__file__).resolve().parents[1]
    path, raw = source_csv(tmp_path)
    data = tmp_path / "data"
    (data / "data").mkdir(parents=True)
    (data / "data/fixture.csv").write_bytes(path.read_bytes())
    broken = raw.copy()
    broken.loc[1, "timestamp"] = broken.timestamp.iloc[0]
    broken.to_csv(data / "data/rejected.csv", index=False)
    windows = [[raw.timestamp.iloc[75], raw.timestamp.iloc[90]]]
    (data / "labels.json").write_text(json.dumps({"fixture.csv": windows, "rejected.csv": []}))
    (data / "LICENSE.txt").write_text(
        "Original unit-test fixture, not upstream data or an upstream license"
    )
    lock = {"revision": "a" * 40, "repository": "https://github.com/numenta/NAB", "files": {}}
    for relative in ("data/fixture.csv", "data/rejected.csv", "labels.json", "LICENSE.txt"):
        content = (data / relative).read_bytes()
        lock["files"][relative] = {
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    plan = {
        "protocol_version": "nab-local-v1",
        "revision": "a" * 40,
        "series": ["fixture.csv", "rejected.csv"],
        "labels_path": "labels.json",
        "license_path": "LICENSE.txt",
        "metadata_paths": [],
        "seed": 42,
        "calibration_alert_rate": 0.02,
        "rolling_window": 31,
    }
    protocol = tmp_path / "plan.json"
    protocol.write_text(json.dumps(plan))
    source_lock = tmp_path / "lock.json"
    source_lock.write_text(json.dumps(lock))
    output = tmp_path / "results"
    command = [
        sys.executable,
        "scripts/evaluate_nab.py",
        "--protocol",
        str(protocol),
        "--data",
        str(data),
        "--lock",
        str(source_lock),
        "--output",
        str(output),
    ]
    subprocess.run(command, cwd=root, check=True, capture_output=True, timeout=120)
    result = json.loads((output / "results.json").read_text())
    assert "NOT official" in result["scope"]
    assert result["descriptive_summary"]["source_rejections"] == 1
    assert result["runs"][1]["status"] == "source_validation_rejected"
    assert result["runs"][1]["metrics"] is None
    run = result["runs"][0]
    assert run["test_windows"][0]["clipped_at_test_boundary"] is True
    assert run["window_membership_counts"]["test"]["positive_rows"] == 11
    assert run["window_membership_counts"]["calibration"]["positive_rows"] == 5
    assert "precision" not in run["unlabeled_metrics"]["mad"]
    assert "precision" not in run["quality_summary"]
    scored = pd.read_csv(output / "fixture-scored.csv")
    assert len(scored) == 20 and scored.label.sum() == 11
    (data / "data/fixture.csv").write_text("changed")
    failure = subprocess.run(command, cwd=root, capture_output=True, timeout=120)
    assert failure.returncode != 0 and b"checksum mismatch" in failure.stderr


def test_conflicting_duplicates_have_structured_rejection_diagnostics(tmp_path):
    path, raw = source_csv(tmp_path)
    raw.loc[1, "timestamp"] = raw.timestamp.iloc[0]
    raw.loc[1, "value"] = raw.value.iloc[0] + 1
    raw.to_csv(path, index=False)
    with pytest.raises(SourceValidationError) as captured:
        load_series(path, [], "fixture")
    assert captured.value.diagnostics["conflicting_timestamp_groups"] == 1
    assert captured.value.diagnostics["exact_duplicate_rows_beyond_first"] == 0


def test_downloader_rejects_mutable_revision_or_unknown_paths_without_network(tmp_path):
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "nab_downloader", root / "scripts/download_nab.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = json.loads((root / "configs/nab_protocol.json").read_text())
    plan["revision"] = "master"
    with pytest.raises(ValueError, match="immutable"):
        module.download(plan, tmp_path / "data", tmp_path / "lock.json")
    plan["revision"] = "a" * 40
    plan["series"] = ["../../private.csv"]
    with pytest.raises(ValueError, match="Unexpected"):
        module.download(plan, tmp_path / "data", tmp_path / "lock.json")
    assert not (tmp_path / "data").exists()
