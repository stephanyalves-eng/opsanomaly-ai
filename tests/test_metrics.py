import pandas as pd
import pytest
from opsanomaly.inference import metrics


def fixture():
    return pd.DataFrame(
        {
            "label": [0, 0, 1, 1],
            "mad_score": [0.1, 0.8, 0.9, 0.2],
            "mad_alert": [False, True, True, False],
            "rolling_mad_score": [0.1, 0.8, 0.9, 0.2],
            "rolling_mad_alert": [False, True, True, False],
            "isolation_forest_score": [0.1, 0.8, 0.9, 0.2],
            "isolation_forest_alert": [False, True, True, False],
        }
    )


def test_known_metrics():
    result = metrics(fixture())["mad"]
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5
    assert result["false_alarms_per_1000_observations"] == 250
    assert result["pr_auc_average_precision"] == pytest.approx(5 / 6)


def test_unlabeled_no_quality_claims():
    result = metrics(fixture().drop(columns="label"))["mad"]
    assert "precision" not in result
    assert "false_alarms_per_1000_observations" not in result
    assert result["alert_rate"] == 0.5


def test_zero_positives():
    frame = fixture()
    frame["label"] = 0
    assert metrics(frame)["mad"]["pr_auc_average_precision"] is None


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_pr_metrics_are_explicitly_undefined(label):
    frame = fixture()
    frame["label"] = label
    result = metrics(frame)["mad"]
    assert result["pr_auc_average_precision"] is None
    assert result["pr_auc_trapezoid"] is None
    assert "single-class" in result["pr_auc_status"]


def test_metrics_counts_and_rank_vs_threshold_distinction():
    frame = fixture()
    entry = metrics(frame)["mad"]
    assert entry["confusion_matrix"] == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
    frame["mad_alert"] = False
    altered = metrics(frame)["mad"]
    assert altered["pr_auc_average_precision"] == entry["pr_auc_average_precision"]
    assert altered["recall"] == 0
    assert "precision" in altered["undefined_metrics"]


@pytest.mark.parametrize(
    "kind", ["empty", "partial_labels", "invalid_labels", "nonfinite_scores", "string_alerts"]
)
def test_invalid_metric_inputs_rejected(kind):
    frame = fixture()
    if kind == "empty":
        frame = frame.iloc[:0]
    elif kind == "partial_labels":
        frame.loc[0, "label"] = float("nan")
    elif kind == "invalid_labels":
        frame.loc[0, "label"] = 2
    elif kind == "nonfinite_scores":
        frame.loc[0, "mad_score"] = float("inf")
    else:
        frame["mad_alert"] = "False"
    with pytest.raises(ValueError):
        metrics(frame)


def test_perfect_ranking_metrics_remain_bounded_despite_roundoff():
    import numpy as np

    frame = pd.DataFrame({"label": [0] * 191 + [1] * 9})
    for name in ("mad", "rolling_mad", "isolation_forest"):
        frame[f"{name}_score"] = np.arange(200, dtype=float)
        frame[f"{name}_alert"] = frame.label.eq(1)
    for item in metrics(frame).values():
        assert 0 <= item["pr_auc_average_precision"] <= 1
        assert 0 <= item["pr_auc_trapezoid"] <= 1
