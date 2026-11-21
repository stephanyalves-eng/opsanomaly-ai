# AI Portfolio Engineering Blueprint
Version 1.0 | 2026-11-15

## Executive decision
Build **OpsAnomaly AI** first to demonstrate actual ML training, evaluation, temporal validation, explainability and inference; build **DocuMind AI** second to demonstrate document intelligence, NLP, traceable extraction and human review. Keep each product independent. Target 9 weeks at 10–12 h/week; this is a planning assumption, not a delivery commitment.

## Architecture rules
- Python 3.11+, `uv` (or pip with locked versions), Ruff, Pytest, Mypy, GitHub Actions, pre-commit.
- No paid APIs; local inference by default. Use Hugging Face Hub for models, datasets (only permitted data) and model cards. Static Spaces for static demos; verify current hardware eligibility before deploying any Python/Gradio Space. A static UI cannot execute server-side Python; either use browser-side inference where feasible or publish a reproducible local Gradio demo and a static showcase.
- No proprietary supplier documents. Synthetic documents must be conspicuously labeled; respect dataset and pretrained model licenses. Do not commit real PDFs, secrets, tokens, or large model binaries to Git.
- Reproducibility: fixed seeds, versioned dataset manifest, immutable train/test splits, evaluation scripts, pinned dependencies, and model metadata.
- Each repo: README, architecture diagram, reproducible demo, tests, benchmark, evaluation report, model card, limitations, license, CI and a 2-minute demo recording.

## 1. OpsAnomaly AI
**Problem:** identify anomalous behavior in industrial/operational time-series without claiming diagnosis or predicting failures unsupported by labels.

### MVP acceptance criteria
1. Import and validate CSV with timestamp, asset_id and numeric sensors; report missing values and sampling irregularities.
2. Provide synthetic sample data with explicit labels and reproducible generation seed.
3. Implement baseline rolling z-score or robust median absolute deviation; Isolation Forest as ML comparator.
4. Split chronologically by asset/sequence to prevent leakage; preprocessing fit on training only.
5. Expose anomaly scores, configurable threshold, charts and per-alert evidence. Explainability describes influential features but never claims causality.
6. Export scored CSV and evaluation JSON. Unit tests cover missing data, constant sensors, duplicates, timezone issues, and out-of-range values.
7. For labeled test data, report precision, recall, F1, PR-AUC and false alarms per 1,000 observations; for unlabeled data, report score distributions and alert rates only.
8. Keep C-MAPSS as an optional follow-up for remaining useful life (RUL) experiments, not as the default labeled anomaly dataset. Use a verified anomaly benchmark with known labels for supervised evaluation, or a synthetic injection protocol clearly described as synthetic.

### Repo layout
```text
opsanomaly-ai/
  README.md
  pyproject.toml
  .github/workflows/ci.yml
  configs/default.yaml
  data/README.md
  docs/{architecture.md,dataset_card.md,evaluation.md,limitations.md}
  src/opsanomaly/{ingest.py,features.py,baseline.py,model.py,explain.py,inference.py}
  scripts/{generate_synthetic.py,train.py,evaluate.py}
  app/{gradio_app.py,static_showcase/}
  tests/{test_ingest.py,test_splits.py,test_inference.py,test_metrics.py}
  model_card.md
```

### Phases
- Week 1: skeleton, CI, threat and cost checks, dataset research and data card.
- Week 2: ingestion, validation, synthetic data, baseline, temporal split.
- Week 3: Isolation Forest, threshold calibration, evaluation and feature-level explanations.
- Week 4: interactive app, model card, QA, local demo and publication.

### Codex prompt A (repository foundation)
```text
Act as a senior ML engineer and Python maintainer. Create a production-quality, local-first repository named opsanomaly-ai. Implement the structure and acceptance criteria in AI_PORTFOLIO_BLUEPRINT.md. Use Python 3.11+, pinned dependencies, Ruff, Mypy, Pytest and GitHub Actions. First write docs/architecture.md and an implementation checklist. Do not use paid APIs, fake metrics, or invent dataset licenses. Include an offline deterministic synthetic dataset generator and an honest baseline. Run tests and summarize evidence, commands, failures and remaining risks. Do not publish or spend money without approval.
```

### Codex prompt B (ML validation)
```text
Audit OpsAnomaly for data leakage, incorrect chronological splits, contamination of threshold tuning, feature scaling fit on test data, inflated metrics and non-causal explanations. Add unit and integration tests, compare rolling robust statistics with Isolation Forest, and produce machine-readable metrics. Distinguish labeled from unlabeled evaluation. Fix defects, rerun tests and document every limitation.
```

## 2. DocuMind AI
**Problem:** extract and compare structured fields from proposals, contracts and technical documents with verifiable page/quote evidence.

### MVP acceptance criteria
1. Support digitally generated PDFs first, with file size/page limits and PDF validation; scanned PDFs/OCR are a separate feature flag.
2. Extract field schemas for vendor, currency, amount, delivery term, warranty and payment terms, with `value`, `source_page`, `evidence_text`, `extraction_method`, `confidence_or_status`.
3. Validate arithmetic, currencies, dates and required fields with Pydantic and deterministic rules.
4. Compare two proposals using normalization and sentence embeddings for semantic matching; retain numeric comparisons as deterministic logic.
5. Return `matched`, `different`, `missing`, `needs_review` statuses; never invent missing fields.
6. Show provenance, allow manual corrections, and export JSON/CSV; do not train on user-uploaded documents by default.
7. Evaluate against manually annotated synthetic/public test documents: field precision/recall/F1, numeric exact match, evidence citation accuracy, divergence detection precision, latency and abstention rate.
8. Treat model confidence as uncalibrated unless independently calibrated; prefer explicit review flags.

### Repo layout
```text
documind-ai/
  README.md
  pyproject.toml
  .github/workflows/ci.yml
  configs/fields.yaml
  docs/{architecture.md,data_policy.md,evaluation.md,limitations.md}
  src/documind/{pdf_ingest.py,schemas.py,extract.py,normalize.py,compare.py,evidence.py,export.py}
  scripts/{generate_test_docs.py,evaluate.py}
  app/{gradio_app.py,static_showcase/}
  tests/{test_pdf.py,test_extraction.py,test_compare.py,test_evidence.py}
  model_card.md
```

### Phases
- Weeks 5–6: safe PDF ingestion, extraction schema, provenance and annotated fixture set.
- Week 7: semantic comparisons, numeric rules, review queue, baseline comparison.
- Week 8: interface, export, benchmarking, failure analysis, tests and documentation.
- Week 9: independent QA and publication of both projects.

### Codex prompt C (DocuMind foundation)
```text
Act as a senior NLP/document intelligence engineer. Build documind-ai as an independent local-first repository following AI_PORTFOLIO_BLUEPRINT.md. Start with text-based PDFs, PyMuPDF, Pydantic, deterministic extraction and field provenance. Add sentence-transformer semantic matching only where it improves a measured baseline. Use synthetic or properly licensed public test files. Do not send documents to remote APIs. Produce an evaluation harness, explicit abstention/review states, complete tests and architecture documentation. Do not fabricate metrics, model capabilities or source evidence.
```

### Codex prompt D (security and quality audit)
```text
Perform a release-blocking audit of DocuMind: malformed PDFs, file size limits, text injection in documents, privacy, path traversal, temporary-file cleanup, unsafe logs, unsupported currencies, inconsistent totals, incorrect evidence page numbers, hallucinated fields and regression tests. Treat PDF text as untrusted input. Return prioritized findings, patches, executed tests and residual risks. Do not deploy without approval.
```

## 3. Hugging Face publishing protocol
1. Check the current pricing, hardware eligibility and storage/quotas on the official Hugging Face site before creating any paid-compute Space.
2. Create separate repositories for each project's model, optionally a dataset with redistribution rights, and a Space only if the deployment target is supported without additional charges.
3. Provide model card sections: intended use, out-of-scope use, data provenance, preprocessing, splits, evaluation protocol, results, limitations, license and reproducibility.
4. Never publish supplier documents, credentials, personal data, or restricted model weights.
5. Test fresh-environment installation and demo; link GitHub commit SHA, model revision and dataset revision.
6. For a free static Space, create a standalone frontend showing real precomputed results clearly labeled as examples, or browser-executed inference; do not misrepresent precomputed outputs as live inference.

## 4. Release gates
- Gate A: tests and lint pass; CI clean; no high-severity dependency issues known.
- Gate B: deterministic evaluation reproducible from documented commands.
- Gate C: UI does not claim more than the model supports; known failures documented.
- Gate D: dataset and model licenses checked; no secrets or private data.
- Gate E: no unexpected billing, GPU allocation or paid inference dependency.
- Gate F: final README in English, screenshots, architecture diagram, case study and public demo links.

## 5. Portfolio and publication sequence
- After week 2: GitHub technical note on time-series validation and baselines; avoid unearned performance claims.
- After week 4: OpsAnomaly demo, GitHub release, Hugging Face model card, LinkedIn/Peerlist/DEV article on leakage-free anomaly detection.
- After week 7: DocuMind progress note on evidence-grounded extraction and deterministic validation.
- After week 9: DocuMind release and comparative article; Medium/LinkedIn post with evaluation tables, limitations and links. Kaggle notebook only where dataset terms allow.
- Website: separate project pages with problem, architecture, measurable evaluation, live/local demo, source code, failures and next steps.

## 6. Costs and operational controls
- Use local CPU first; optional GPU only if already available. Before any Hugging Face deployment inspect hardware tier, runtime, storage and billing settings. Do not assume Gradio/Docker runtime is free.
- GitHub Actions: avoid unnecessary large jobs; cache carefully and track usage limits.
- Hugging Face tokens: minimum permissions; local environment variables; no secrets in code, CI logs or notebooks.
- Codex: track usage within existing plan and approve repository actions, installs, deployment and purchases.
- Retain a COST_LOG.md containing provider, plan, resource, expected charge, approval and actual usage.

## 7. Initial commands (Windows PowerShell)
```powershell
python --version
git --version
mkdir ai-portfolio
cd ai-portfolio
mkdir opsanomaly-ai, documind-ai
# Install uv from its official documented installer, or use python -m venv .venv.
# Create and initialize each repository only after Codex generates its pyproject.toml.
```

## 8. Next action
Start OpsAnomaly. Copy this blueprint into its repository as `AI_PORTFOLIO_BLUEPRINT.md`, then paste Codex prompt A. Review architecture and proposed dependencies before approving implementation. Keep DocuMind scoped until the OpsAnomaly baseline and evaluation are reproducible.
