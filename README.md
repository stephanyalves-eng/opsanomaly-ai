# OpsAnomaly AI

[![CI](https://github.com/stephanyalves-eng/opsanomaly-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/stephanyalves-eng/opsanomaly-ai/actions/workflows/ci.yml)

Local-first operational anomaly scoring: frozen MAD, strictly past-only rolling median/MAD, and Isolation Forest. Synthetic evaluation tests engineering behavior; it does not establish failure detection or diagnosis.

## Reproduce (PowerShell, Python 3.11 or 3.12)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.lock
python -m pip install --no-deps --no-build-isolation -e .
python scripts/generate_synthetic.py
python scripts/train.py
python scripts/evaluate.py
python scripts/benchmark.py
python app/gradio_app.py
```

The app binds 127.0.0.1 with sharing/telemetry disabled. After initial package installation, runtime is offline CPU-only. No API key or paid API. Models are trusted local joblib files; do not load downloaded/untrusted artifacts. Old v0.1 artifacts must be retrained for schema v2.

CSV requires offset-aware `timestamp`, nonempty `asset_id` and 1..64 numeric sensors. Optional labels must be complete 0/1. Times increase within each asset; duplicate asset/time pairs and infinities/out-of-range values are rejected. Missing values and sampling irregularities are reported. Training-only medians impute missing sensor values.

Shared UTC cutoffs separate train/calibration/test globally, including staggered assets. Ratios are 60/20/20 of distinct times, not always each asset's row count. Assets must occur in all periods. Test scores use fixed calibration thresholds; rolling scores use prior per-asset observations only, excluding the current point before scoring. Earlier test values may update subsequent windows, never labels. Repeated calls replay the full future batch from immutable calibration-end history; they are not streaming chunks. New assets or rows at/before calibration end are rejected.

## Audit and evidence

[Audit findings](docs/audit.md), [machine-readable comparison](docs/benchmark.json), [evaluation protocol/results](docs/evaluation.md), [architecture](docs/architecture.md), [dataset card](docs/dataset_card.md), [model card](model_card.md), [known limitations](docs/limitations.md), [checklist](docs/implementation_checklist.md) and [cost log](COST_LOG.md).

```powershell
ruff check .
ruff format --check .
mypy src scripts app
pytest -q --junitxml=docs/audit-tests.xml
python scripts/benchmark.py --seeds 42,43,44 --window 31 --alert-rate 0.02
```

Evaluation exports `artifacts/scored.csv`, `evaluation.json`, paired metadata and charts; it verifies dataset/model/partition hashes. Benchmark JSON includes labeled metrics, independent unlabeled views, per-asset support, confusion counts, actual thresholds, fixed protocol and measured timing. AP and trapezoidal PR area are separate; single-class PR summaries are null. Unlabeled input returns distributions and alert rates only. No test-label threshold optimization or point adjustment.

Calibrate once before test inspection: the UI's slider is exploratory. Changing window/budget after seeing test results contaminates confirmatory evaluation. Calibration can contain anomalies and the 2% budget need not achieve a 2% future alert rate. Feature deviations are descriptive; the forest reference explicitly says it is not model attribution or causality. Manual `predict(..., threshold=...)` overrides only frozen MAD, never incompatible method score scales.

For a genuinely future CSV, use `python scripts/score.py --data path/to/future.csv`. Include all observations since calibration to preserve rolling history; model/schema/time checks prevent retrospective scoring as a future holdout. Do not confuse arbitrary labeled scoring with the immutable benchmark.

## Local examples and publication status

The [static showcase](app/static_showcase/index.html) displays real precomputed results and executes no Python. The [two-minute offline terminal replay](docs/demo/index.html), also [asciinema v2](docs/demo/demo.cast), captures actual CLI output with edited playback pauses; regenerate with `python scripts/record_demo.py`. It is not a browser screen recording.

[Foundation UI screenshot](docs/screenshots/local-demo.png) predates the rolling audit and is historical; the current app is exercised by integration tests. No new browser screenshot is claimed for v0.2.

Dependency versions remain locked (102 packages, including app/dev/build transitive dependencies). Installing `-e .[dev,app]` alone does not lock transitive versions. The [prepublication dependency audit](docs/publication-dependency-audit.json) reported zero known advisories; CI audits the same lock. Historical reports remain available. Original generated data/code uses MIT; no external dataset license is asserted. Publication status and hosted CI evidence are tracked in [publication.md](docs/publication.md). No production deployment, purchase or public model/dataset weights are included. Local checks are evidence for an experimental MVP, not certification of production infrastructure.

## Expanded synthetic stress suite

Eight predeclared scenarios now test drift, prolonged changes, plausible stuck sensors, random missingness, block outages and spikes with missing values, plus clean/nuisance controls. [Protocol](docs/stress_protocol.md), [measured results and failures](docs/stress-evaluation.md), [machine-readable evidence](docs/stress-results.json).

```powershell
python scripts/stress_benchmark.py --export-csv
python scripts/plot_stress.py
```

Fit once per seed on paired clean train/calibration; reuse thresholds across all scenarios. No model retuning. This clean-calibration suite is separate from the earlier contaminated point-spike benchmark. Rolling MAD adapts away prolonged disturbances; all methods have major blind spots for plausible stuck sensors and outages. At the stress-v1 stage the suite had 88 passing tests; see docs/stress-tests.xml.

## Separate data-quality checks

Raw missing-value flags and persistent-missing/repeated-value review rules are now available through `opsanomaly.quality.QualityMonitor`, the local Gradio analysis and `python scripts/quality_score.py --data data/synthetic.csv`. They inspect values before imputation and do not change numeric model alerts or diagnose faults. Constant training sensors are explicitly not assessable for repetition.

[Fixed protocol](docs/quality_protocol.md), [new-seed results, controls and limitations](docs/quality-evaluation.md), [machine-readable evidence](docs/quality-results.json). Reproduce the 33-case experiment with `python scripts/quality_benchmark.py --export-csv`. The three new seeds are separate from the previous stress seeds. Identical legitimate plateaus and injected stuck values receive identical review flags, so human/process context remains necessary. Current full suite: 113 passed; docs/quality-tests.xml.

## First external real-data evaluation: NAB subset

The project now has a reproducible offline adapter/evaluation for three preselected publisher-released NAB realKnownCause series: two pass validation and are scored; machine_temperature_system_failure is retained as rejected due to 12 conflicting timestamps. No source rows are repaired or performance-based replacements selected.

[Data/license/provenance card](docs/nab_dataset_card.md), [fixed protocol](docs/nab_protocol.md), [measured results and failures](docs/nab-evaluation.md), [JSON evidence](docs/nab-results.json), [original attribution](third_party/NAB-NOTICE.md). Publisher repository-declared MIT is preserved; raw inputs stay local/ignored. This is point-level anomaly-window membership under our own chronological split, **not official NAB scoring or real failure diagnosis**. Results are poor and were not retuned.

```powershell
python scripts/download_nab.py  # explicit pinned public download, approximately 1.53 MB
python scripts/evaluate_nab.py  # offline, frozen raw hashes and fixed settings
python scripts/plot_nab.py
```

No dependency changes or automatic network calls during evaluation/tests. Full suite at this stage: 131 passed. Source timezone is unspecified; UTC is used solely as an ordering coordinate. The inspected final periods cannot be reused as untouched validation after tuning.

## Public repository and hosted CI

Source is published at https://github.com/stephanyalves-eng/opsanomaly-ai . Standard Linux CI passed all 131 tests on Python 3.11 and 3.12, plus Ruff, strict Mypy, synthetic reproduction and dependency audit. [Verified run and publication evidence](docs/publication.md). No hosted app, public model weights or raw dataset redistribution. Earlier platform-status snapshots in historical reports are superseded by this hosted evidence.
