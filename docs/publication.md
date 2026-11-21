# Public repository and hosted CI evidence

Published on 2026-11-21 (America/Sao_Paulo): **[https://github.com/stephanyalves-eng/opsanomaly-ai](https://github.com/stephanyalves-eng/opsanomaly-ai)**, owner stephanyalves-eng, visibility public as explicitly selected by the user. Source code, reproducible local demo, tests and honest evaluation reports are public. No live inference hosting, release tag, dataset/model Hub upload, domain or purchase was performed.

## Verified hosted checks

Validated code/workflow commit: [`22bd11f`](https://github.com/stephanyalves-eng/opsanomaly-ai/commit/22bd11f25fc6de3864211329535c85989bdc7d37). [Successful GitHub Actions run](https://github.com/stephanyalves-eng/opsanomaly-ai/actions/runs/44943208717) (44943208717), both jobs completed successfully:

| Standard Linux job | Tests | Existing upstream warnings | Test duration | Other checks |
|---|---:|---:|---:|---|
| Python 3.11 | 131 passed | 14 | 26.90 s | dependency install/check, Ruff, strict Mypy, synthetic reproduction and vulnerability audit passed |
| Python 3.12 | 131 passed | 14 | 33.54 s | dependency install/check, Ruff, strict Mypy, synthetic reproduction and vulnerability audit passed |

Mypy checked 27 source files. Both audit steps reported no known vulnerabilities. The fresh local prepublication audit covered 102 pinned packages with zero known advisories; publication-dependency-audit.json contains that result. This is point-in-time evidence, not a security guarantee. [Machine-readable publication evidence](publication-evidence.json). This document update changes no scientific source, model parameters or dependency pins; later docs-only commits do not alter the referenced validated experiment. Current branch workflow status is available on the repository Actions tab.

## Reviewed publication scope

Initial reviewed index: 96 files, 3,082,282 bytes. No token/private-key patterns found. Raw NAB CSVs/downloaded labels, all serialized models, .venv/.verification environments, .env files and runtime artifacts were excluded. The frozen public source URLs/hashes, original license/attribution, reported results and derived plots remain included; the poor real-data metrics and rejected source are not concealed. Historical JUnit machine hostnames were removed while preserving test counts, outcomes and durations. Public advisory references were the only email-like scanner matches.

.gitattributes disables automatic EOL conversion so reviewed code bytes, original license bytes and recorded scientific source fingerprints remain reproducible across Windows/Linux checkouts. This preserves bytes without disabling text diffs. Staged bytes were checked against reviewed working files before publication. No sensitive raw data or model binary is needed to run the offline synthetic demo/tests.

## CI resource controls

SHA-pinned official checkout/setup-python actions; standard ubuntu-latest runners; read-only contents token; checkout credentials not persisted; 15-minute job timeout; concurrency cancellation. No larger runners, cache uploads, artifact storage uploads, scheduled jobs, real-dataset downloads or paid inference. The job is gated to public repositories: changing visibility to private does not silently start potentially billed private CI.

GitHub documents standard public GitHub-hosted runners as free: https://docs.github.com/en/billing/concepts/product-billing/github-actions . No paid resource was requested or purchased; this does not approve future paid compute/storage changes. Billing dashboard/invoice was not used as evidence. Usage here consists of standard public Linux jobs, with no cache/artifact storage requested.

## Failure and correction retained

Initial code commit d20bdf85bf422ced8d24ba982c623641730bdf46 was published successfully. [First workflow attempt](https://github.com/stephanyalves-eng/opsanomaly-ai/actions/runs/44942634771) failed validation without starting any jobs because runner.temp was incorrectly referenced in job-level env. Commit 22bd11f moved it to step-level env for tests/synthetic evaluation, where that context is available. Reference: https://docs.github.com/en/actions/reference/workflows-and-actions/contexts . The corrected run above is green; no scientific data, metric or dependency was changed to hide the failure.

## Limits

Public source and passing CI do not establish fault diagnosis, unseen-asset generalization, real-time SLAs or production serving security. NAB uses broad anomaly windows; two series perform poorly, one source is rejected for conflicting timestamps. Synthetic tests retain their stated scope, and quality-review flags are separate from model decisions. Those limitations and original historical evidence remain in their respective reports. Hosted Linux/Python 3.11/3.12 results supersede earlier snapshots that said hosted CI had not yet run.
