# Data-quality rules: new-seed validation v1

Executed 2026-11-21 on local Windows/Python 3.12.10. These are descriptive raw-data checks, not a new ML fault classifier. The original numeric detector decisions are unchanged.

## Frozen protocol and evidence

[Protocol fixed before evaluation](quality_protocol.md), configs/quality_protocol.json. Seeds **1051,2083,4099**, excluded previous stress seeds 42/43/44, 11 scenarios per seed, 33 runs. Each case has 3 assets and 600 held-out observations. Same paired clean fit periods for original cases and quantized/plateau controls; constant_sensor is explicitly fitted separately. No parameter search or post-result rule change.

[Full machine-readable results](quality-results.json) preserve each seed/scenario, exact partition membership, raw quality summaries, rule-target confusion counts and event delay/coverage, unlabeled summaries, unchanged numeric thresholds, source/protocol hashes and versions. CSVs with numeric scores plus separate quality columns are under ignored artifacts/quality-validation. Earlier stress-results.json is retained unchanged as historical evidence; new code files mean historical source fingerprints are not statements about the current tree.

## Fixed behavior

- Inspect raw values before imputation. Every NaN gets an immediate per-sensor informational flag.
- Request review from the third consecutive missing observation per sensor.
- Request review from the eighth consecutive observed value within absolute tolerance 1e-8 of a fixed run anchor. Anchoring avoids mistaking accumulated slow movement for a constant run.
- Enable repetition assessment only for sensors with at least three observed training values and training range above tolerance. Training constants report not-assessable, rather than a stuck-fault alert.
- Learn cadence from training only. Missing observations reset repetition; observed values reset missing runs. Gaps above 1.5 times training median cadence reset both. Calibration replays prior context but cannot enable a training-constant gate.
- Keep histories independent per asset and replay prediction batches immutably. No future input or label sets rules, gates or thresholds. Review is not diagnosis; legitimate plateaus can trigger exactly the same repetition alerts.

## Coverage of the declared injections

| Condition | Rule | Mean point recall of declared interval | First-alert delay, all 9 asset/seed events |
|---|---|---:|---:|
| stuck_sensor | repeat | 95.00% | 6 observations |
| missing_block | missing | 98.33% | 2 observations |

These percentages describe original 120-observation synthetic injection intervals with explicit detection latency; they are not overall accuracy or real fault recall. For stuck_sensor, the final pre-injection observation already equals the frozen value, so it counts toward the eight-value run: first alert is six observations after the labeled event start. For the outage, the first two NaN rows are informational only; persistent review begins on the third row, delay two. No event point adjustment or alert backfill was applied. Event hit/delay and raw point coverage remain separate in JSON.

## Controls and the ambiguity that remains

| Scenario | Mean review-alert rate over all held-out observations |
|---|---:|
| clean | 0.00% |
| benign_drift | 0.00% |
| gradual_drift | 0.00% |
| sustained_shift | 0.00% |
| stuck_sensor | 57.00% |
| missing_random | 0.11% |
| missing_block | 59.00% |
| spikes_with_missing | 0.11% |
| legitimate_plateau | 57.00% |
| quantized_clean | 0.00% |
| constant_sensor | 0.00% |

**legitimate_plateau has exactly the same sensor measurements as stuck_sensor, but normal operational labels. It receives the same repeated-value flags: 114 of 200 rows per asset (57% of the full test, 95% of the plateau interval).** No method observing only those identical readings can infer which assigned physical interpretation is correct. The output therefore says repeated values / review, never sensor broken. Calling these alerts confirmed faults would create false positives. Zero review alerts on the clean/quantized/constant controls in these small simulations is not proof of a zero future false-alert rate.

Random missingness and spikes_with_missing have immediate missing-value information flags, but persistent review is rare (about 0.11% of rows here). Such review alerts are compared against the explicitly declared block-outage target, not presented as universal fault precision. Direct agreement of missing flags with raw NaN masks is reported only as a deterministic plumbing check, not learned predictive accuracy.

## Numeric models remain separate

| Original scenario | Frozen MAD point recall | Rolling MAD point recall | Isolation Forest point recall |
|---|---:|---:|---:|
| stuck_sensor | 2.59% | 0.74% | 4.63% |
| missing_block | 0.00% | 1.94% | 0.00% |
| gradual_drift | 74.63% | 3.24% | 79.26% |
| sustained_shift | 100.00% | 13.89% | 100.00% |

No numeric alert was OR-combined with a quality flag. Numeric models still have their prior blind spots; the separate quality channel addresses observable raw conditions with its own semantics and limitations. Do not claim an overall ML accuracy improvement from direct missingness checks. The UI now exports/displays quality_* columns and a separate data_quality_rules JSON summary alongside numeric scores; the quality CLI does not use operational labels as quality ground truth.

## API and reproduction

```python
from opsanomaly.quality import QualityMonitor
monitor = QualityMonitor().fit(train, calibration, sensors)
quality_rows = monitor.predict(future_batch)  # raw, un-imputed sensor columns
```

```powershell
python scripts/quality_benchmark.py --export-csv
python scripts/quality_score.py --data data/synthetic.csv --output artifacts/quality
python app/gradio_app.py
ruff check .
ruff format --check .
mypy src scripts app
pytest -q --junitxml=docs/quality-tests.xml
```

quality_score.py takes a complete historical CSV, uses the shared 60/20/20 split to fit context and export held-out quality.csv/quality.json; it is not a serialized live-monitor service. Its labeled and unlabeled versions produce identical quality summaries. The library accepts future rows strictly after all calibration, only known assets and exact raw schemas; quality_ is a reserved output prefix. Calls replay the entire future batch from immutable calibration history, rather than preserving state across independent chunks.

## Verification and remaining limits

25 new focused tests passed. Full suite: **113 passed**, zero failures/errors/skips, 14 pre-existing Matplotlib/pyparsing deprecation warnings, 32.10 seconds. Ruff lint/format pass (32 Python files); strict Mypy passes (23 source files); pip check reports no broken requirements. Full JUnit: quality-tests.xml. The updated UI assertions are additionally checked by quality-ui-test.xml. Core tests cover raw missingness before imputation, exact run onset, interruptions/gap reset, fixed tolerance anchor, training-only variability gates, constant sensors, asset isolation, label/future invariance, immutable replay, identical plateau/stuck measurements, unchanged numeric decisions, labeled/unlabeled CLI exports and exclusion of old seeds.

No new package, dependency version, API, external dataset, model-parameter tuning, purchase or publication. The full UI test requires local sockets outside the sandbox, as diagnosed previously. No new visual/browser screenshot is claimed; interface integration is verified by its construction/callback tests. Existing dependency vulnerability evidence remains point-in-time and was not rerun.

Limits: only three new seeds in the same synthetic family, no statistical significance or industrial generalization; subjective injection labels; finite 120-row disturbances; homogeneous assets and clocks; absolute tolerance depends on sensor units/resolution; eight-value and three-missing thresholds are engineering defaults, not validated fault thresholds. Legitimate standby or stable processes can trigger repetition alerts. Training-constant or insufficiently observed channels cannot reliably assess repetition and may miss later faults. Irregular sampling changes elapsed detection latency; gaps reset context. Missing timestamps/absent records are not NaN cells and are not detected as continuous outages: no resampling, heartbeat or external clock timer exists. Predictions need the complete future sequence to preserve intervening context. Model scoring still imputes missing values, but quality flags must always receive the original raw frame. No causal diagnosis, authentication, remote serving, persistent stream state or real failure validation is supplied.

These seed results are now inspected development evidence. Future changes need another frozen validation plan and untouched evaluation; do not tune these same seeds and reuse them as a final blind test.
