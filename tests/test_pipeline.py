import json
from pathlib import Path
import subprocess
import sys


def test_commands_and_dataset_guard(tmp_path):
    root = Path(__file__).resolve().parents[1]
    data = tmp_path / "synthetic.csv"
    output = tmp_path / "artifacts"
    commands = [
        ["scripts/generate_synthetic.py", "--output", str(data), "--rows", "100"],
        ["scripts/train.py", "--data", str(data), "--output", str(output)],
        ["scripts/evaluate.py", "--data", str(data), "--artifacts", str(output)],
    ]
    for command in commands:
        subprocess.run(
            [sys.executable, *command], cwd=root, check=True, capture_output=True, timeout=120
        )
    report = json.loads((output / "evaluation.json").read_text())
    assert report["metrics"]["mad"]["observations"] == 60
    assert (output / "scores.png").stat().st_size > 0

    unlabeled = tmp_path / "unlabeled.csv"
    from opsanomaly.ingest import read_csv
    from opsanomaly.features import temporal_split

    held_out = temporal_split(read_csv(data)[0])[2]
    held_out.drop(columns="label").to_csv(unlabeled, index=False)
    new_output = tmp_path / "new-scores"
    subprocess.run(
        [
            sys.executable,
            "scripts/score.py",
            "--data",
            str(unlabeled),
            "--model",
            str(output / "model.joblib"),
            "--output",
            str(new_output),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=120,
    )
    unlabeled_report = json.loads((new_output / "evaluation.json").read_text())
    assert "precision" not in unlabeled_report["metrics"]["mad"]
    assert unlabeled_report["metrics"]["mad"]["observations"] == 60
    # Pairing and partition guards run before trusted local deserialization.
    metadata_path = output / "metadata.json"
    original_metadata = metadata_path.read_text()
    changed_metadata = json.loads(original_metadata)
    changed_metadata["splits"]["test"]["membership_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(changed_metadata))
    result = subprocess.run(
        [sys.executable, *commands[-1]], cwd=root, capture_output=True, timeout=120
    )
    assert result.returncode != 0 and b"Partition membership" in result.stderr
    metadata_path.write_text(original_metadata)
    model_path = output / "model.joblib"
    original_model = model_path.read_bytes()
    model_path.write_bytes(original_model + b"changed")
    result = subprocess.run(
        [sys.executable, *commands[-1]], cwd=root, capture_output=True, timeout=120
    )
    assert result.returncode != 0 and b"paired run metadata" in result.stderr
    model_path.write_bytes(original_model)
    data.write_text(data.read_text() + "\n")
    result = subprocess.run(
        [sys.executable, *commands[-1]], cwd=root, capture_output=True, timeout=120
    )
    assert result.returncode != 0
    assert b"immutable" in result.stderr


def test_app_build_and_callback(tmp_path):
    import importlib.util
    from opsanomaly.synthetic import generate

    path = Path(__file__).resolve().parents[1] / "app/gradio_app.py"
    spec = importlib.util.spec_from_file_location("local_demo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.build_app() is not None
    csv = tmp_path / "sample.csv"
    generate(rows=100).to_csv(csv, index=False)
    scored, text, download, report = module.analyze(str(csv), 0.02)
    assert len(scored) == 60
    assert "thresholds" in json.loads(text)
    assert "data_quality_rules" in json.loads(text)
    assert "quality_review_alert" in scored
    assert "quality_evidence" in scored
    assert "precision" not in json.loads(text)["data_quality_rules"]
    assert Path(download).exists() and Path(report).exists()
    # Explicitly remove only the callback's own files and now-empty directory.
    Path(download).unlink()
    Path(report).unlink()
    Path(download).parent.rmdir()


def test_fully_unlabeled_training_and_evaluation_commands(tmp_path):
    import pandas as pd
    from opsanomaly.synthetic import generate

    root = Path(__file__).resolve().parents[1]
    data = tmp_path / "unlabeled.csv"
    generate(rows=100).drop(columns="label").to_csv(data, index=False)
    out = tmp_path / "run"
    for command in (
        ["scripts/train.py", "--data", str(data), "--output", str(out)],
        ["scripts/evaluate.py", "--data", str(data), "--artifacts", str(out)],
    ):
        subprocess.run(
            [sys.executable, *command], cwd=root, capture_output=True, check=True, timeout=120
        )
    report = json.loads((out / "evaluation.json").read_text())
    assert report["evaluation_mode"] == "unlabeled"
    assert set(report["metrics"]) == {"mad", "rolling_mad", "isolation_forest"}
    for method in report["metrics"].values():
        assert {"precision", "recall", "f1", "false_alarms_per_1000_observations"}.isdisjoint(
            method
        )
    assert report["protocol"]["calibration_label_diagnostics"]["injected_or_positive_rows"] is None
    assert len(pd.read_csv(out / "scored.csv")) == 60


def test_machine_readable_benchmark_command(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "benchmark.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/benchmark.py",
            "--seeds",
            "42",
            "--rows",
            "100",
            "--output",
            str(output),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=120,
    )
    report = json.loads(output.read_text())
    assert report["protocol"]["test_labels_used_for_tuning"] is False
    assert len(report["runs"]) == 1
    run = report["runs"][0]
    assert run["metrics"]["rolling_mad"]["observations"] == 60
    assert "precision" not in run["unlabeled_metrics"]["rolling_mad"]
