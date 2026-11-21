# Cost log

| Provider/resource | Expected monetary charge | Approval and actual usage |
|---|---|---|
| Local CPU, Python and disk | No API charge | Authorized by build request; measured runtime in evaluation JSON |
| Public Python package downloads | No paid service | Local environment only; network installation required |
| Hosted GitHub Actions | Not started | Workflow authored only; account allowance/billing not verified |
| Hugging Face / hosted inference | Not used | No deployment, token, GPU or paid API |

No purchase or publication is authorized by this implementation. Existing Codex plan usage is outside repository measurement.

## ML validity audit v0.2

Local CPU only: 63-test suite, three fixed synthetic seeds, offline editable metadata refresh and deterministic demo regeneration. No dependencies added or changed, external dataset/model downloaded, new hosted job, paid API, purchase or publication. Existing point-in-time vulnerability report retained; no new advisory lookup was performed.

## Synthetic stress suite (2026-11-21)

Local CPU only: 24 paired comparisons, CSV/JSON exports, rendered chart and 88-test full suite. No dependency installation/change, external data/model download, paid API, hosted job, purchase or publication.

## Quality validation v1 (2026-11-21)

Local CPU only: 33 new-seed comparisons, raw quality CSV/JSON exports, 25 focused tests and full 113-test suite. No package/dependency change, paid API, external dataset/model download, hosted job, purchase or publication.

## First external dataset cycle (2026-11-21)

Read-only Git revision lookup and seven pinned official NAB HTTPS files (1,525,913 bytes, approximately 1.53 MB). Local CPU fitting/scoring only; evaluation and tests offline. No package/dependency change, paid API, cloud/GPU compute, hosted job, purchase, publication, remote repository or outbound message. Raw data remains local and ignored by Git.
