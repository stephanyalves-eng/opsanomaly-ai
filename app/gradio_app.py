"""Ephemeral local CSV analysis; uploaded content never leaves the process."""

import os

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
import json
import tempfile
from typing import cast
from pathlib import Path
import gradio as gr
import pandas as pd
from opsanomaly.ingest import read_csv
from opsanomaly.features import temporal_split
from opsanomaly.inference import Detector, metrics
from opsanomaly.quality import QualityMonitor, quality_summary


def analyze(path: str, rate: float) -> tuple[pd.DataFrame, str, str, str]:
    if not path:
        raise gr.Error("Choose a CSV first")
    try:
        df, quality = read_csv(path)
        train, cal, test = temporal_split(df)
        detector = Detector().fit(train, cal, quality["sensors"], rate=rate)
        scored = detector.predict(test)
        quality_rows = QualityMonitor().fit(train, cal, quality["sensors"]).predict(test)
        for column in quality_rows.columns:
            if column.startswith("quality_"):
                scored[column] = quality_rows[column]
    except (ValueError, KeyError) as error:
        raise gr.Error(str(error)) from error
    folder = Path(tempfile.mkdtemp(prefix="opsanomaly-"))
    csv = folder / "scored.csv"
    report = folder / "evaluation.json"
    scored.to_csv(csv, index=False)
    text = json.dumps(
        {
            "evaluation_mode": "labeled" if "label" in scored else "unlabeled",
            "scope": "exploratory; changing settings after viewing test labels invalidates confirmatory evaluation",
            "quality": quality,
            "data_quality_rules": quality_summary(scored, quality["sensors"]),
            "thresholds": detector.thresholds,
            "metrics": metrics(scored),
        },
        indent=2,
    )
    report.write_text(text, encoding="utf-8")
    return scored, text, str(csv), str(report)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="OpsAnomaly AI", delete_cache=(3600, 3600)) as demo:
        gr.Markdown(
            "# OpsAnomaly AI — local experimental analysis\n"
            "Shared UTC 60/20/20 periods; pooled training ends before calibration. "
            "Rolling MAD uses prior observations only (31-row window). "
            "Evidence describes deviations, not causes or forest attribution. "
            "This UI is exploratory: tuning after viewing results invalidates independent evaluation. "
            "Raw missing/repeated-value review flags are separate from numeric model decisions. "
            "They are not fault diagnoses. Synthetic labels do not establish real failure detection. Downloads expire after one hour."
        )
        upload = gr.File(file_types=[".csv"], type="filepath")
        rate = gr.Slider(0.001, 0.2, value=0.02, label="Calibration alert budget (not probability)")
        sample = Path(__file__).resolve().parents[1] / "data/synthetic.csv"
        if sample.exists():
            gr.Examples(
                examples=[[str(sample), 0.02]],
                inputs=[upload, rate],
                label="Original labeled synthetic example",
            )
        button = gr.Button("Fit on training period and score held-out period")
        table = gr.Dataframe(label="Held-out scores and per-row evidence")
        chart = gr.LinePlot(
            x="timestamp",
            y="rolling_mad_score",
            color="asset_id",
            title="Past-only rolling MAD scores",
        )
        report = gr.Textbox(label="Quality report, actual thresholds and evaluation", lines=15)
        csv = gr.File(label="Scored CSV")
        export = gr.File(label="Evaluation JSON")

        def run(path: str, rate: float) -> tuple[pd.DataFrame, pd.DataFrame, str, str, str]:
            frame, text, csv_path, json_path = analyze(path, rate)
            return frame, frame, text, csv_path, json_path

        button.click(run, [upload, rate], [table, chart, report, csv, export])
    return cast(gr.Blocks, demo)


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", share=False, max_file_size="100mb")
