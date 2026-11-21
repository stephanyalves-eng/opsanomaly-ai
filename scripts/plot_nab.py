"""Plot verified local NAB exports without fetching or retuning."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "opsanomaly-mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from opsanomaly.inference import DETECTORS


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="artifacts/nab/results.json")
    p.add_argument("--output", default="artifacts/nab/plots")
    a = p.parse_args()
    source = Path(a.results)
    report = json.loads(source.read_text(encoding="utf-8"))
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    for run in report["runs"]:
        if run["metrics"] is None:
            continue
        name = Path(run["series"]).stem
        path = source.parent / f"{name}-scored.csv"
        if hashlib.sha256(path.read_bytes()).hexdigest() != run["scored_csv_sha256"]:
            raise ValueError("Scored CSV does not match report")
        frame = pd.read_csv(path)
        times = pd.to_datetime(frame.timestamp, utc=True)
        gaps = times.diff().dt.total_seconds()
        break_line = gaps > 1.5 * gaps.dropna().median()
        fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True, layout="constrained")
        axes[0].plot(times, frame.sensor_value.mask(break_line), color="#334155", linewidth=0.8)
        axes[0].set_ylabel("Source value")
        for axis, method in zip(axes[1:], DETECTORS, strict=True):
            axis.plot(times, frame[f"{method}_score"].mask(break_line), linewidth=0.8)
            axis.axhline(
                run["thresholds"][method], color="#dc2626", label="Fixed calibration threshold"
            )
            axis.set_ylabel(method)
            axis.legend(loc="upper left", fontsize=8)
        for axis in axes:
            for window in run["test_windows"]:
                axis.axvspan(
                    pd.Timestamp(window["published_start"]),
                    pd.Timestamp(window["published_end"]),
                    color="#f59e0b",
                    alpha=0.16,
                )
            axis.set_xlim(times.min(), times.max())
        axes[-1].tick_params(axis="x", labelrotation=20, labelsize=8)
        axes[-1].set_xlabel(
            "Source clock localized to UTC for ordering; physical timezone unverified"
        )
        fig.suptitle(
            name + "\nShading = published anomaly-window membership; not official NAB scoring",
            fontsize=12,
        )
        fig.savefig(out / f"{name}.png", dpi=120)
        plt.close(fig)
        print(out / f"{name}.png")


if __name__ == "__main__":
    main()
