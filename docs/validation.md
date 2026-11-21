> Historical foundation validation (v0.1). Superseded for ML validity by [audit.md](audit.md), benchmark.json and audit-tests.xml.

# Validation evidence

Validation date: 2026-11-15 (America/Sao_Paulo). Local Windows CPU; Python 3.12.10 selected because the default system interpreter is Python 3.15.2, outside the supported package range. Python 3.11/3.12 CI is authored; hosted execution remains unverified.

## Executed commands

From repository root, using the virtual environment Python:

```powershell
python -m pip install -r requirements.lock
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
python -m ruff check .
python -m ruff format --check .
python -m mypy src scripts app
python -m pytest -q -o faulthandler_timeout=60
python scripts/generate_synthetic.py
python scripts/train.py
python scripts/evaluate.py
python scripts/record_demo.py
python -m pip_audit -r requirements.lock --disable-pip --no-deps --format json --output docs/dependency-audit.json
```

Ruff lint and format pass (20 Python files). Strict Mypy passes (14 source files; third-party libraries without types are ignored). The first completed full suite passed 25 tests in 19.56 seconds, including CLI generation/training/evaluation, immutable-dataset mismatch rejection, model serialization, Gradio construction and analysis callback. Final fresh-environment installation from the exact lock file and editable package install both succeeded; all 26 tests passed in 53.96 seconds with 14 upstream deprecation warnings. This run additionally checks the sensor-count bound and actual unlabeled CSV scoring exports.

Actual Gradio browser smoke check: selected the original synthetic example, invoked analysis and observed the held-out score table, populated MAD chart, quality/threshold/metric report and CSV/JSON download links. Screenshot: screenshots/local-demo.png. No public share link. The browser renders chart times in its local timezone; CSV and table timestamps retain UTC offsets.

Three repeated generation/training/evaluation executions produced identical metric dictionaries; synthetic generator and model determinism also have regression tests. See run_metadata.json for dataset checksum, exact split boundaries, versions and thresholds. Static showcase JSON and PNG are copies of measured artifacts. Benchmark is a single local warm inference timing, excluding training, loading and file export; it is not a throughput guarantee.

## Failures and corrections

1. Sandboxed package installation failed DNS resolution. Retried with approved network access; no paid package or API used.
2. A premature check while installation was still running reported missing pytest/mypy. Waited for completion and reran checks.
3. Initial Mypy found an untyped Gradio Blocks return; added an explicit cast at the library boundary.
4. Initial audit found 81 advisory entries across Gradio, Pillow, pip, pytest and Starlette. A first patched Gradio pin rejected Pillow 13; resolution failed without changing the environment. Updated to compatible Gradio 6.32.0 and patched dependencies. A follow-up found three setuptools advisories; upgraded the build pin to 84.2.1.
5. Full tests stalled twice in sandboxed Python asyncio socket-pair setup while constructing Gradio. Captured faulthandler stack, interrupted those runs, then passed the suite outside the sandbox. This was not hidden as a skip.
6. Matplotlib could not write its default user cache in the sandbox. Configured its cache under the OS temporary directory. Upstream Matplotlib/pyparsing deprecation warnings remain; they are not test failures.
7. Fresh lock preparation found SciPy/contourpy versions requiring Python 3.12. Pinned compatible versions so Python 3.11 remains a CI target. Python 3.13+ is explicitly excluded pending separate audio/runtime compatibility validation.
8. Full-page browser screenshot capture failed; ordinary viewport capture succeeded and is bundled.

## Security and release gates

Final dependency audit: 102 locked packages, zero known advisories returned; full machine-readable result is dependency-audit.json. Initial advisory evidence remains in dependency-audit-initial.json. This is a point-in-time database check, not proof of absence of vulnerabilities. The lock pins versions without wheel hashes; artifact authenticity and future advisories remain supply-chain risks.

Local Git repository initialized on main, no remote configured, no commit/push/release/hosting or purchase performed. CI files exist but have not run on GitHub. No pretrained weights, external dataset, supplier documents or credentials are bundled. Original synthetic data is expressly labeled, with MIT terms chosen for this repository's original code and outputs; no external license is invented.

Residual scope limits: easy stationary synthetic point injections, contaminated calibration, fixed alert-budget tradeoff, pooled asset distributions, no real-world benchmark, no causal explanations, trusted-only joblib loading, no hardened multi-user serving. Private uploads and generated exports can remain in local temp storage; users must manage retention on shared devices. See limitations.md and COST_LOG.md.


## Final fresh-environment result

```text
python -m pip install -r requirements.lock         PASS
python -m pip install --no-deps --no-build-isolation -e .  PASS
pytest -q                                        26 passed, 14 warnings in 53.96s
ruff check .                                     All checks passed
ruff format --check .                            20 files already formatted
mypy src scripts app                             Success: 14 source files
pip-audit (102 locked packages)                   No known vulnerabilities found
```

Verified on Windows / Python 3.12.10. Linux and Python 3.11 remain authored CI targets, not locally executed claims. Requirement-lock SHA-256: `e2f8852665843ff6022762e4bb93321d57f71a5a62d05cd599d64e3b32302eee`.
