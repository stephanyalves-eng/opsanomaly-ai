# Implementation checklist

- [x] Package, exact dependency pins, CI and pre-commit
- [x] Strict ingestion and quality reporting
- [x] Deterministic synthetic injection protocol and provenance
- [x] Immutable shared UTC temporal partitions across assets and training-only preprocessing
- [x] Frozen MAD, past-only rolling MAD, Isolation Forest and calibration-only thresholds
- [x] Evidence, labeled/unlabeled metrics and exports
- [x] Local interactive demo and static showcase
- [x] Edge-case and integration tests; lint and typing
- [x] Measured evaluation, benchmark, model card and limitations
- [x] Foundation fresh lock-file install: 26 passed historically; audit v0.2 full rerun: 63 passed
- [x] Two-minute edited CLI recording, offline player and actual browser screenshot
- [x] Publication gates documented; no publication or spending

## Acceptance evidence map

| Requirement | Implementation / evidence |
|---|---|
| CSV schema and quality report | ingest.py; test_ingest.py |
| Deterministic labeled synthetic data | synthetic.py, generate_synthetic.py; manifest and determinism test |
| Honest baseline and ML comparator | baseline.py, model.py; measured evaluation.md |
| Chronological split and train-only preprocessing | features.py; test_splits.py and leakage tests |
| Scores, thresholds, chart and alert evidence | inference.py, explain.py, gradio_app.py; current callback integration test and historical foundation screenshot |
| CSV/JSON exports and edge cases | evaluate.py, score.py; tests and end-to-end CLI test |
| Labeled and unlabeled metrics | metrics function; hand-calculated test and unlabeled CLI integration |
| Synthetic protocol rather than mislabeled RUL dataset | dataset_card.md; no external dataset used |

Publication gates remain deliberately separate: no hosted CI result, external benchmark, public demo URL or remote model revision is claimed.

## ML audit v0.2

- [x] Shared UTC boundaries and staggered-asset leakage regressions
- [x] Past-only rolling median/MAD with exact prior-window and no future/current self-inclusion tests
- [x] Label/feature/preprocessing/threshold invariance and immutable history tests
- [x] Model and partition membership pairing guards
- [x] Invalid/single-class metric guards and explicit labeled/unlabeled CLI coverage
- [x] Independent seed-by-seed comparison, machine-readable findings and JUnit evidence
- [x] Known scientific, temporal, explanation and operational limitations documented
- [x] Legacy v1 artifacts retrained; current static results and edited CLI replay regenerated

## Synthetic stress step 1 (2026-11-21)

- [x] Predeclared eight-case, three-seed protocol before execution
- [x] Paired clean fit periods and fixed thresholds, no parameter search
- [x] Drift, prolonged shift, stuck sensors, random/block missingness and combined spikes
- [x] Raw point metrics, event delay/coverage, missingness diagnostics, unlabeled views and CSV exports
- [x] 25 added stress tests; complete suite 88 passed
- [x] Measured failures, machine-readable evidence, inspected chart and limitations

## Raw-quality channel and new-seed validation

- [x] Rules/protocol fixed before measuring new seeds 1051/2083/4099
- [x] Raw missing information, persistent-missing and anchored repetition review flags
- [x] Training-only cadence/assessability; constant sensors and gap reset handled
- [x] Separate API, standalone CSV/JSON CLI and local app integration; numeric decisions unchanged
- [x] Legitimate plateau, quantization and constant controls included; all 33 cases retained
- [x] 25 focused tests and complete 113-test suite passed; labeled/unlabeled summaries separate
- [x] Latency, false-review ambiguity, limits and source/protocol fingerprints documented

## Step 2: first external real-data cycle

- [x] Official publisher license/metadata inspected; immutable revision fixed
- [x] Source/label/metadata hashes, local notice and original license retained
- [x] Explicit bounded downloader; raw inputs ignored; evaluation/tests offline
- [x] Independent per-series chronological splits and unchanged parameters
- [x] Two real series evaluated; third retained as source-validation rejection, without repair
- [x] Window-label/timezone assumptions, weak metrics and quality limits documented
- [x] 18 new offline tests; complete 131-test suite passed
- [ ] Machine-temperature series requires justified source/duplicate handling before evaluation
- [ ] Broader untouched external validation and deployment evidence remain future work

## Step 4: public source and hosted CI

- [x] Public visibility explicitly selected; correct account and new-repository name verified
- [x] Reviewed source-only commit; secrets/raw data/models/environments excluded
- [x] Original license/attribution and honest measured limitations retained
- [x] Public repository created and source pushed
- [x] Workflow context error corrected; SHA-pinned actions and cost controls retained
- [x] Hosted Linux/Python 3.11 and 3.12: 131 tests each plus lint, typing, reproduction and audit passed
- [x] Public URL, successful run, first failure/correction and resource controls documented
