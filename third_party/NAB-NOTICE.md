# NAB source attribution

Publisher: Numenta Inc. Project: Numenta Anomaly Benchmark (NAB). Source: https://github.com/numenta/NAB/tree/ea702d75cc2258d9d7dd35ca8e5e2539d71f3140

The publisher repository LICENSE.txt declares MIT, corroborated by .zenodo.json at the pinned revision. Copyright notice: Copyright 2014-2024 Numenta Inc. The unmodified downloaded license is retained in NAB-LICENSE.txt. No separate per-file data license or independent third-party provenance guarantee is asserted. The bundled data README identifies the chosen inputs as realKnownCause series. Raw CSVs and labels remain local/ignored; no redistribution/publication performed.

Selected series and SHA-256/byte counts are in configs/nab_source_lock.json, together with the original source URLs. Adaptations: source-clock timestamps localized to UTC only as an ordering coordinate; anomaly-window membership mapped inclusively to a binary label; strictly chronological 60/20/20 split per separate series. No source data repairs, imputation in saved raw files, relabeling, window expansion or official NAB scoring. Machine-temperature series rejected without scoring due to conflicting repeated timestamps.

Publisher-requested reference: Ahmad, S., Lavin, A., Purdy, S., & Agha, Z. (2017). Unsupervised real-time anomaly detection for streaming data. Neurocomputing DOI: https://doi.org/10.1016/j.neucom.2017.04.070 . See upstream README for exact bibliographic details.
