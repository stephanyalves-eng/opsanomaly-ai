# Synthetic dataset card

Original, offline NumPy generator, protocol v1. Three fictional assets, 1,000 one-minute observations each by default; seed 42. Temperature combines a sinusoid and Gaussian noise; vibration uses Gaussian noise. Each row independently has 4% probability of an injected temperature spike (+/-6â€“12) and vibration spike (+2â€“4). Binary labels indicate injection, never physical failures. No real entities or personal information. No external dataset license is asserted. Original generated outputs are offered under the repository MIT license.

The process is deliberately easy and stationary, has no validated physics, and does not cover contextual anomalies, drifts, real missingness or genuine breakdowns. Contamination occurs in training and calibration as well as test. Test labels remain hidden during fitting and threshold selection. Protocol v2 uses shared UTC 60/20/20 cutoffs over distinct timestamps; aligned synthetic assets retain 600/200/200 rows each. All pooled training precedes all calibration/test observations. The audit benchmark uses fixed seeds 42,43,44, 31-row rolling windows and 2% calibration budgets without test-label selection. Positive calibration contamination is retained and reported.

Do not call this a verified public benchmark. C-MAPSS remains a separate optional RUL experiment; it is not used here.

## Additional stress dataset v1 (2026-11-21)

Original code in stress.py generates paired nominal datasets, with interventions restricted to test periods. Same clean training/calibration per seed; eight cases and seeds 42/43/44, no post-result tuning. Label policies explicitly distinguish nuisance missingness/small drift from injected stuck/outage conditions. They do not establish physical fault truth. See stress_protocol.md and stress-evaluation.md for exact amplitudes, boundaries, limitations and executed failures. Original outputs use the repository MIT terms; no third-party data license is asserted.

## Quality validation v1

New predeclared seeds 1051/2083/4099 extend the same synthetic family with legitimate plateaus, held-out quantization and training-constant sensor controls. Plateau and stuck cases intentionally have identical sensor readings but different operational labels, making diagnostic ambiguity explicit. Constant-sensor cases are fitted separately; other cases retain paired clean train/calibration. Raw quality flags and model scores use separate targets. See quality_protocol.md and quality-evaluation.md; the data remains original synthetic code/output under repository MIT terms, not an external dataset or new industrial evidence.

## First real-data subset: NAB local v1

The synthetic-only descriptions above concern earlier experiments. A separate pinned NAB subset now provides initial external real-data evidence. Publisher-declared repository MIT, source/label hashes, copyright notice, timezone/label assumptions and the source rejection are documented in nab_dataset_card.md and nab-evaluation.md. Three preselected series, two evaluated and one rejected without repair. No claim of official NAB score, complete industrial validation, independently verified per-point fault labels or redistribution approval.
