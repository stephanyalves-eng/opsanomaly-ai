# ML validity audit — OpsAnomaly v0.2.0

Audit date: 2026-11-15, America/Sao_Paulo. Local Windows / Python 3.12.10. Scope: leakage, chronology, preprocessing, threshold calibration, score/metric integrity, explanation wording and labeled/unlabeled evaluation. No publication, paid service or external dataset. Nine findings fixed; retained design limitations are explicit below and in limitations.md.

## Findings and fixes

| ID / priority | Finding | Correction |
|---|---|---|
| A01 P1 | Per-asset row cutoffs could leak future information across staggered assets in a pooled model | Shared UTC cutoffs; global train < calibration < inference; reject incompatible coverage |
| A02 P1 | Predict accepted historical train/calibration rows as future observations | Strict future/context/schema/known-asset validation; necessary for safe rolling history |
| A03 P2 | Existing MAD was frozen, not a rolling comparator | Add exact past-only per-asset rolling median/MAD; retain frozen baseline for an honest comparison |
| A04 P2 | Dataset hash alone did not pair model/split identities | Artifact schema v2, model hash and exact split membership verification before local deserialization |
| A05 P2 | Single-class AP and unchecked metric inputs could mislead | Null single-class PR summaries, input guards, denominator flags, support and confusion counts |
| A06 P2 | A scalar manual cutoff affected incompatible method score scales; UI enabled post-test selection | MAD-only manual override, actual thresholds/source fields, explicit exploratory warning |
| A07 P2 | Shared MAD evidence could be mistaken for forest attribution | Separate rolling evidence and explicitly marked contextual MAD reference for forest |
| A08 P2 | Generic evaluation called arbitrary CSVs synthetic | Generic held-out scope; synthetic claim restricted to actual generator benchmark |
| A09 P3 | Floating arithmetic could report AP infinitesimally above one | Bound only endpoint roundoff within 1e-12; fail larger invalid outputs |

No evidence was found of automatic test-label tuning, test-fitted medians/scales or label-driven model fitting. These previously correct properties now have adversarial invariance regression tests. Labels are used for diagnostics/evaluation only. Isolation Forest uses raw training-imputed features: there is no StandardScaler to fit. Shared preprocessing is frozen after training.

## Executed verification

```powershell
python -m ruff check .
python -m ruff format --check .
python -m mypy src scripts app
python -m pytest -q -o faulthandler_timeout=60 --junitxml=docs/audit-tests.xml
python scripts/generate_synthetic.py
python scripts/train.py
python scripts/evaluate.py
python scripts/benchmark.py
python scripts/record_demo.py
python -m pip install --no-deps --no-build-isolation -e .
```

Final full suite: **63 passed, zero failures/errors/skips**, 14 upstream Matplotlib/pyparsing deprecation warnings, 30.18 seconds. JUnit: audit-tests.xml. Ruff lint/format and strict Mypy pass (16 source files; external untyped dependencies ignored). Machine-readable findings with named regression tests: audit-findings.json. Fixed three-seed benchmark: benchmark.json. Updated default scored CSV/JSON, model and chart: artifacts (ignored); precomputed JSON/chart copies: app/static_showcase. The offline terminal replay captures the new real CLI output with edited pacing. Foundation browser screenshot is historical, not a new audited UI capture; current app construction and callback pass integration tests.

Tests cover shared cutoffs on staggered timelines, same-UTC timestamp boundary grouping, invalid/nonoverlapping assets, duplicate/unsorted rows, frozen medians/scales, label permutation invariance, calibration-only quantile thresholds, no future/current-row self-inclusion in rolling scores, history isolation by asset, missing-value train imputation, model roundtrip, immutable predictions, retrospective/new-asset rejection, manual threshold provenance, single-class/invalid metrics, known confusion counts, forest attribution wording, dataset/model/partition mismatch rejection and full labeled/unlabeled command paths. No result was silently skipped.

## Comparison and retained concerns

Fixed seeds 42/43/44, rolling window 31, 2% calibration budget, identical shared holdouts. Mean F1: frozen MAD 0.7656, rolling MAD 0.7627, Isolation Forest 0.7631. Rolling F1 population std 0.1386; forest 0.1098. These three easy simulation runs do not establish a winner or real-world validity. See evaluation.md for every seed and benchmark.json for per-asset/label support and independent unlabeled views.

Positive calibration contamination is deliberately retained (counts exported), not cleaned by labels or hidden. It can raise thresholds and hurt recall; this is an unsupervised calibration design limitation, not test leakage. High AP does not imply high recall at a fixed threshold. The UI remains exploratory: no software can establish that a person did not previously tune settings against the holdout. Confirmatory use needs frozen settings and new untouched data.

Rolling references consume prior test features, never test labels; this is declared online adaptation. All calls reset to calibration-end history, so replay the complete future sequence rather than treating separate chunks as persisted state. Fixed-count windows have variable elapsed duration with irregular sampling. Sensor clock/arrival errors, model drift, sustained anomalies, missingness effects, unit-dependent scale fallback, small synthetic sample size, unreliable uncertainty intervals and explanation mismatch remain known limitations. Full scientific, temporal, security and operational list: limitations.md.

## Failures encountered and resolved

- First edits had an extra comma in the train metadata dict; Ruff/Mypy caught the syntax error before execution and it was corrected.
- Two legacy tests initially failed because they passed a subset feature list with an extra sensor still in their input. Their fixtures were corrected to exercise intended overlap/infinity guards under the new exact-schema contract; subsequent full runs passed.
- Full app tests require local Python socket-pair access blocked by this sandbox (previously diagnosed); ran the full suite with approved escalation, not skips or a networking workaround.
- Refreshing editable metadata under the sandbox failed because the sandbox's brace-containing temp path was interpreted by setuptools as a substitution variable. The same offline editable install succeeded with approved escalation and installed v0.2.0.
- Existing upstream Matplotlib deprecation warnings remain. No dependency changed and no new vulnerability audit was claimed. Previous point-in-time advisory report remains separate from this ML audit.

## Residual release status

Trusted local pickle loading only; hashes are integrity/pairing checks, not authentication. No model trained on private data may be published without a separate data/rights review. No public hosting/CI execution, dependency authenticity guarantee, security certification or Python 3.11/Linux verification is claimed. All previously generated v1 model artifacts require retraining. See architecture.md and model_card.md for the versioned protocol; validation.md is retained as superseded historical foundation evidence.
