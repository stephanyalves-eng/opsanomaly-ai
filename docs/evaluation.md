# Evaluation — audited protocol v2

All values below are executed results from the original offline synthetic generator, not targets or external benchmark claims.

Fixed protocol: seeds 42,43,44; 3 assets x 1,000 observations; shared UTC 60/20/20 periods; rolling window 31; calibration alert budget 0.02; 200 forest trees. No settings were selected from test labels during this comparison. Each run has 600 held-out point observations. Labels are independent injected spikes, not failures.

## Fixed three-seed comparison

| Method | Mean precision | Mean recall | Mean F1 | F1 population std | Mean false alarms / 1,000 |
|---|---:|---:|---:|---:|---:|
| mad | 1.0000 | 0.6422 | 0.7656 | 0.1486 | 0.0000 |
| rolling_mad | 1.0000 | 0.6364 | 0.7627 | 0.1386 | 0.0000 |
| isolation_forest | 1.0000 | 0.6305 | 0.7631 | 0.1098 | 0.0000 |

These are equal-weight arithmetic averages of per-seed metrics, not metrics on pooled scores. Three runs and this easy protocol do not establish a winning detector, significance or reliable confidence intervals. Population standard deviation describes these three runs only. Near-perfect rankings and zero false alarms reflect simple synthetic injections and small holdouts.

## Every seed (avoid best-seed selection)

| Seed | Test positives | Calibration positives / rows | Frozen MAD F1 | Rolling MAD F1 | Forest F1 |
|---|---:|---:|---:|---:|---:|
| 42 | 26 | 25 / 600 | 0.5556 | 0.7907 | 0.6667 |
| 43 | 26 | 21 / 600 | 0.8696 | 0.9167 | 0.9167 |
| 44 | 22 | 17 / 600 | 0.8718 | 0.5806 | 0.7059 |

## Interpretation and definitions

The seed-42 rolling baseline catches 17/26 injected points (F1 0.7907), while the forest catches 13/26 (F1 0.6667). This advantage does not persist across all seeds. The approximately 4% contamination of calibration and fixed 2% score-tail budget can suppress recall; no labels were removed or used to optimize thresholds. Repeatedly selecting settings after viewing these results would invalidate independent evaluation.

Scores are evaluated at original point granularity, with no event-window adjustment. Each method has a distinct calibration threshold and strict greater-than alerts. Counts, confusion matrices, support, per-asset results, actual thresholds, split-membership checksums and package versions are in benchmark.json and run_metadata.json.

PR summaries are reported separately as average precision (non-interpolated) and trapezoidal PR area. They are null for a single-class labeled test. Undefined precision/recall/F1 denominators use zero plus explicit undefined_metrics flags. Only tiny arithmetic endpoint roundoff is bounded to [0,1]. False alarms / 1,000 = FP / all scored observations x 1,000, not FP / negatives.

Unlabeled views in every benchmark run omit precision, recall, F1, PR-AUC, confusion matrices and false-alarm metrics. They expose score quantiles, alert counts/rates and threshold provenance only. The full unlabeled train/evaluate CLI path has an integration test.

Rolling scores update each asset reference with already observed test sensor values after scoring; the forest and frozen MAD remain fixed. This is prequential sensor adaptation, not label fitting or a causal explanation. Calls replay the complete future sequence from calibration history, not arbitrary stream chunks.

Timing records are single local CPU batch measurements excluding file export. Timing varies with cache/process load; it is not an SLA or statistically robust latency benchmark. No real industrial dataset or independent anomaly benchmark was used.

Reproduce: `python scripts/generate_synthetic.py`, `python scripts/train.py`, `python scripts/evaluate.py`, then `python scripts/benchmark.py`. See [audit](audit.md), [machine-readable findings](audit-findings.json), [full benchmark](benchmark.json) and [known limitations](limitations.md).
