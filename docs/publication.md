# GitHub publication preparation â€” 2026-11-21

User selected public repository `stephanyalves-eng/opsanomaly-ai`. Intended scope: source, original synthetic generators, tests, documented results/plots, third-party notice and frozen source metadata. No raw NAB data, downloaded labels, model binaries, virtual environments, secrets or runtime artifacts are included. Existing local test reports have hostname attributes removed; test counts, outcomes, durations and source/metric evidence are unchanged.

Workflow: SHA-pinned official checkout/setup-python actions, standard ubuntu-latest runners, Python 3.11/3.12, read-only token, checkout credentials not persisted, 15-minute job timeout, concurrency cancellation. No cache, artifact upload, larger runner, scheduled job, dataset download, hosted inference or deployment. A public-repository-only job condition prevents runs if visibility later becomes private. GitHub documents standard public runners as free: https://docs.github.com/en/billing/concepts/product-billing/github-actions . This does not approve any paid resource or future storage/compute change.

Local preparation and dependency audit are recorded before first commit/push. Remote repository and hosted CI status must be verified after upload; a workflow file alone is not successful hosted execution. Raw dataset/model artifacts remain local, and data/license/poor-performance limitations must remain visible. No deployment, release tag, public model/dataset Hub upload, domain or purchase is part of this step.

Pending evidence: local candidate/secret review, initial commit SHA, public repository URL and actual workflow result/URL. This preparation record will be followed by measured publication evidence, without fabricating hosted success.

## Local review evidence

94 initial candidate files, approximately 3.07 MB before this publication record and attributes. No raw external CSV/label input, serialized model, environment, runtime artifact, key or token was selected. Public advisory references were the only email-like matches. Historical JUnit machine hostnames were removed without changing results. Fresh publication audit returned zero known advisories for 102 pinned packages; see publication-dependency-audit.json.

.gitattributes disables automatic text/EOL conversion so scientific code-byte fingerprints, original license bytes and reviewed file content remain reproducible across Windows/Linux checkouts. This does not disable text diffs or change Python behavior. The CI uses no package/data/model publishing actions.
