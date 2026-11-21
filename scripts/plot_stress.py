"""Plot executed stress results without rerunning models or changing thresholds."""

import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "opsanomaly-mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from opsanomaly.inference import DETECTORS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="artifacts/stress/results.json")
    parser.add_argument("--output", default="docs/stress-comparison.png")
    args = parser.parse_args()
    report = json.loads(Path(args.results).read_text(encoding="utf-8"))
    summary = report["by_scenario_equal_weight_seed_summary"]
    positive = [
        s
        for s in summary
        if any(r["metrics"]["mad"]["positives"] > 0 for r in report["runs"] if r["scenario"] == s)
    ]
    scenarios = list(summary)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), layout="constrained")
    colors = ["#334155", "#0891b2", "#d97706"]
    labels = ["Frozen MAD", "Rolling MAD (31 prior rows)", "Isolation Forest"]
    for ax, names, key, title in zip(
        axes,
        [positive, scenarios],
        ["recall", "false_alarms_per_1000_observations"],
        ["Point recall on injected conditions", "False alerts / 1,000 total observations"],
        strict=True,
    ):
        positions = np.arange(len(names))
        for i, (method, color, label) in enumerate(zip(DETECTORS, colors, labels, strict=True)):
            values = [summary[s][method][key]["mean"] for s in names]
            ax.barh(positions + (i - 1) * 0.24, values, height=0.22, label=label, color=color)
        ax.set_yticks(positions, [s.replace("_", " ") for s in names])
        ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        if key == "recall":
            ax.set_xlim(0, 1.03)
    fig.suptitle(
        "Synthetic stress v1: three fixed seeds; clean paired training/calibration\n"
        "Executed mean values, no parameter tuning, no real-world validity claim",
        fontsize=13,
    )
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=3, frameon=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
