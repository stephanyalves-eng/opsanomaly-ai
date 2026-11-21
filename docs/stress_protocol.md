# Synthetic stress protocol v1 — predeclared

Plan fixed before first stress execution, 2026-11-21. Configuration: configs/stress_protocol.json. No parameter search; preserve every scenario/seed result, including failures. Keep the earlier spike benchmark separate.

Three seeds (42/43/44), three fictional assets, 1,000 one-minute rows each. Shared UTC 60/20/20 split. Training and calibration are clean and identical across scenarios within a seed. Fit once per seed; reuse the exact detector/thresholds across cases. Default 31-row rolling window, 2% calibration budget and 200 forest trees. No test-label fitting or threshold tuning. Perturb only held-out observations.

All scenarios use the same seed-specific nominal temperature sinusoid plus Gaussian noise and vibration Gaussian noise. Event rows [floor(.82*n), floor(.94*n)) within each asset are fixed before scoring. At default size this is a 120-row event, longer than the 31-observation rolling reference, with 20-row pre-event and 60-row recovery intervals. No rows outside this range receive event perturbations. Control differences express explicit benchmark label policy, not an inferred physical truth.

| Scenario | Perturbation / label policy |
|---|---|
| clean | No perturbation, all labels zero; false-alert control |
| benign_drift | Temperature ramps 0 to +1.5 during event, all labels zero; assumed acceptable nuisance drift |
| gradual_drift | Temperature ramps 0 to +12 and vibration 0 to +3; all event rows labeled one, including barely observable onset |
| sustained_shift | Temperature +8 and vibration +2.5 throughout event; event labeled one |
| stuck_sensor | Temperature held at its immediately prior observed value; vibration unchanged; event labeled one despite plausible numeric range |
| missing_random | Each held-out sensor independently missing with probability .10; labels zero; nuisance data-quality control |
| missing_block | Both sensors missing throughout event; event labeled one as synthetic sensor outage |
| spikes_with_missing | Independent 4% held-out point-spike injections (+/-6..12 temperature and +2..4 vibration), plus .10 random sensor missingness; labels record original injected points even if missingness hides them |

Report original point precision/recall/F1, AP, false alerts per 1,000, per-asset support/confusion counts and alert rates. Single-class control PR summaries are null; do not average control F1 into anomaly comparisons. No event point adjustment. Separately report event hit/miss, first alert delay in observations, and raw coverage/early-vs-late recall within sustained events; event hit must never replace point recall. Missingness coverage is descriptive and is not a separate trained missingness detector. Include unlabeled views containing no supervised quality metrics. Timings are local measurements, not SLAs.

Known expected challenges (hypotheses, not measured outcomes): rolling adaptation may erase long disturbances; a plausible stuck value may evade amplitude detectors; median imputation may hide outages or spikes. No model changes are part of this comparison. Three seeds, manually specified synthetic severities and subjective label policies cannot validate real industrial accuracy or establish a winner.
