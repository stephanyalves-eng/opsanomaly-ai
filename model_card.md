# OpsAnomaly AI model card — v0.2.0

## Intended use and boundaries
Exploratory local anomaly review of numeric operational time series for known assets. Human review required. Not for diagnosis, safety decisions, autonomous actions, failure prediction, causality or RUL.

## Models and provenance
Original frozen training median/MAD and past-only per-asset rolling median/MAD (31 prior observations), plus scikit-learn Isolation Forest with 200 trees, one worker, seed 42 by default. No pretrained model or external dataset. Original code and generated synthetic outputs use this repository's MIT license; third-party dependencies retain their own terms.

## Preprocessing, splits and calibration
Training-only median imputation and scaled MAD; labels/identifiers excluded. Forest uses imputed raw numeric features, with no StandardScaler. Shared UTC cutoffs across assets at 60% and 80% of distinct times. All assets must occur in every period; fitting and inference reject temporal overlap. Rolling calibration history starts at the training tail; rolling test history includes prior train/calibration observations and earlier test features, never future values or labels. Calls are immutable batch replays, not persistent streaming.

Higher 98th-percentile calibration thresholds, one per method, strict greater-than alerts. No label-based threshold fitting. Positive calibration contamination remains; count diagnostics are exported. Manual/UI tuning after inspecting test outcomes invalidates independent validation. Settings were fixed before the three-seed synthetic audit comparison.

## Evidence and evaluation
Measured per-seed, per-asset and equal-weight seed summaries are in docs/benchmark.json and docs/evaluation.md. Single-class AP/trapezoidal PR metrics are null; labeled metrics include support and confusion counts. Unlabeled reports contain only score/alert summaries and provenance. No event adjustment or real industrial benchmark. No claim that three seeds establish a superior model.

MAD evidence identifies descriptive feature deviations. Forest output labels the MAD reference as contextual only, NOT forest attribution. No causal explanation or failure probability is returned.

## Artifacts and reproducibility
Run README commands. Artifact schema v2 requires retraining v1 artifacts. Dataset/model SHA-256, partition-membership hashes, boundaries, thresholds, window, versions and calibration counts are recorded and checked. Local joblib/pickle is trusted-input-only and may retain sensor history. No remote revision or published Git SHA is invented. See docs/architecture.md, docs/audit.md and docs/limitations.md for protocol, findings and all currently identified limitations.

## Expanded validation: synthetic stress v1

No detector parameters were changed. Eight predeclared cases and three seeds reuse paired clean fitting periods; results are in docs/stress-results.json and docs/stress-evaluation.md. Sustained changes expose rolling adaptation, plausible stuck values evade amplitude detectors, and imputation hides block outages. These tests add failure evidence, not a real-world robustness claim. Future quality-signal changes require a new untouched evaluation.

## Separate quality-rule channel

The numeric models and parameters are unchanged. A raw-value QualityMonitor now flags missing/repeated observations for review, with explicit training-constant assessability. New seeds 1051/2083/4099 and 33 cases are documented in docs/quality-evaluation.md and docs/quality-results.json. Direct missingness agreement is a plumbing check, not ML accuracy. Legitimate plateaus cannot be distinguished from sensor freezes with identical readings; no physical diagnosis is returned. The channel is not OR-combined with model alerts.

## Initial external real-data evidence

Fixed existing models were evaluated on two accepted preselected NAB series; a third was rejected for conflicting timestamps. See docs/nab-evaluation.md and docs/nab-results.json. Performance is poor and labels denote broad anomaly-window membership, not independent per-point fault truth. No official NAB score, deployment suitability or fault diagnosis is claimed. This adds limited real-data evidence without changing model parameters; inspected holdouts are now development data for any future changes. Publisher license/source evidence and attribution are retained separately.
