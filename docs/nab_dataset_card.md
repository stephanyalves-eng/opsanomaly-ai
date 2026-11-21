# NAB subset data card

## Provenance and permissions

External publisher repository: [numenta/NAB](https://github.com/numenta/NAB/tree/ea702d75cc2258d9d7dd35ca8e5e2539d71f3140), revision `ea702d75cc2258d9d7dd35ca8e5e2539d71f3140`. Verified through read-only git ls-remote and pinned raw HTTPS downloads. Publisher [LICENSE.txt](https://github.com/numenta/NAB/blob/ea702d75cc2258d9d7dd35ca8e5e2539d71f3140/LICENSE.txt) declares MIT, corroborated by [.zenodo.json](https://github.com/numenta/NAB/blob/ea702d75cc2258d9d7dd35ca8e5e2539d71f3140/.zenodo.json). Preserve the complete original notice in third_party/NAB-LICENSE.txt and attribution in third_party/NAB-NOTICE.md. This is the publisher's repository-level declaration, not an invented separate dataset license or a guarantee of the full upstream rights chain. No raw data, labels or models were published.

The [publisher data README](https://github.com/numenta/NAB/blob/ea702d75cc2258d9d7dd35ca8e5e2539d71f3140/data/README.md) categorizes these inputs as realKnownCause data. We use the three preselected CSV names for machine temperature, ambient temperature and CPU utilization/misconfiguration. Units/physical clocks and source-event causal narratives were not independently verified. This corpus is public and widely reused, not a private blind deployment trial. No new data/model service or paid API.

Download corpus subset: seven files, 1,525,913 bytes in total (three CSVs, combined-window JSON, original license and two metadata files). Raw inputs remain under ignored data/external/nab. The exact upstream paths, URLs, byte counts and SHA-256 digests are frozen in configs/nab_source_lock.json. Downloader refuses mutable revisions, unexpected paths, changed bytes against an existing lock and publisher license/metadata mismatches. Evaluation never downloads implicitly and verifies every frozen raw input before scoring.

## Labels and preprocessing

Labels come from [combined_windows.json](https://github.com/numenta/NAB/blob/ea702d75cc2258d9d7dd35ca8e5e2539d71f3140/labels/combined_windows.json), not locally invented failures. Points inside supplied intervals, including endpoints, receive one; points outside receive zero. These are broad anomaly-window membership labels, not proof that every point inside a window is physically abnormal or a breakdown. No point-adjusted alert filling. This evaluation is not official NAB scoring and must not be compared to its leaderboard.

Source clock strings lack timezone offsets. The adapter localizes them to UTC solely as a reproducible sorting/index coordinate; actual publisher timezone remains unverified. Original CSV bytes are preserved. No resampling or source-row repairs. Numeric missing values would be imputed using training medians only, while quality flags inspect raw values. Each series uses an independent model and its own 60/20/20 distinct-timestamp split; unrelated series are not pooled.

## Source quality and scope

machine_temperature_system_failure has 22,695 rows and 12 repeated timestamps with different values, no exact duplicate rows. It is retained in results as source_validation_rejected with diagnostics and no metrics. Inferring corrected times, dropping records or averaging values would require a separate sourced decision. Two other selected series pass strict input validation and are evaluated. This exclusion is structural and was decided before inspecting model metrics; no better-performing replacement was selected.

Temperature ambient has 7,267 rows; CPU has 18,050 rows. Exact partitions, positive-window counts, source missingness/irregularities and test-window boundary clipping are in docs/nab-results.json. Training may contain labeled windows; labels are diagnostic only, not a cleaning criterion. Quality review outputs have no independently annotated quality target here and therefore receive no quality precision/recall claim. The final test periods have now been inspected; future changes need new untouched data/protocol.
