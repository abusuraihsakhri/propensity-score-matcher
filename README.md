# Propensity Score Matcher

Python tools for propensity-score estimation and descriptive matching analyses of observational research datasets. The core analytical module uses only the Python standard library.

## Capabilities

- Propensity-score estimation using a logistic regression IRLS solver.
- Nearest-neighbor matching with optional control reuse, configurable ratio, and a caliper in **standard deviations of logit propensity scores**.
- Greedy global closest-pair matching (an approximation, **not** mathematically optimal assignment).
- Propensity-score stratum assignment (descriptive; no stratified treatment-effect estimator).
- Covariate balance using absolute standardized mean differences (SMD).
- Descriptive average treatment effect on the matched treated (ATT) when outcomes are available.
- Command-line, CSV batch, optional FastAPI, and browser interfaces.

**Statistical limitations:** Treatment assignment, overlap, model specification, and unmeasured confounding remain the investigator's responsibility. Matching does not establish causal identification. The matched-pair normal interval is approximate; for reused controls, variable matching ratios, or fewer than two independent pairs, standard errors and confidence intervals are not reported. The library does not implement valid ATC or population ATE estimation. No automatic model selection, imputation, or multiple-testing correction is provided.

## Browser application

The static browser interface in [web/](web/) executes the repository's Python matching module in Pyodide, with CSV import, matching options, covariate balance display, and CSV export. It can be served locally or deployed by the included GitHub Pages workflow after Pages has been enabled.

Browser files run locally; uploaded CSV contents are not sent to an application server. The browser downloads the Pyodide runtime from jsDelivr, so first launch requires network access. Do not upload identified clinical data. Supported environments are modern browsers with WebAssembly enabled. Large cohorts may take appreciable time.

To serve the web interface locally, install the web extra and run:

```bash
python -m pip install -e ".[web]"
uvicorn agents.api:app --reload
```

Then open http://127.0.0.1:8000/. The server is also responsible for the separate demonstration audit endpoints; the browser's propensity-score analysis does not call those endpoints.

## Installation and CLI

Python 3.10 or newer is required.

```bash
python -m pip install -e .
ps-matcher match --treatment 1 1 0 0 --covariates '[[0.5],[1.2],[0.8],[1.5]]' --json
ps-matcher balance --treatment 1 1 0 0 --covariates '[[0.5],[1.2],[0.8],[1.5]]'
ps-matcher ps --treatment 1 1 0 0 --covariates '[[0.5],[1.2],[0.8],[1.5]]'
ps-matcher batch --input sample.csv --output results.csv
```

CSV input needs a binary <code>treatment</code> column (0/1), optionally an <code>outcome</code> column, and numeric covariates. Missing, non-finite or inconsistent values are rejected rather than silently replaced. Batch output contains propensity scores, matched indices and, for stratification, stratum indices.

Use `python cli.py match --help` for matching settings. Analysis results should be independently verified before use in publications.

## Optional API and audit demo

```bash
python -m pip install -e ".[web]"
uvicorn agents.api:app --host 127.0.0.1 --port 8000
```

Endpoints: `GET /health`, `GET /metrics` (JSON counters), `POST /api/audit`, `POST /api/chat` (mock provider only), and `GET /api/audit/logs`. These are generic demonstration services, not clinical triage or medical decision support.

The audit chain uses HMAC-SHA256 signatures and verifies signed entry fields. Its entries are stored **in memory**, and their hashed input payloads are not retained for verification. Configure `AUDIT_SECRET_KEY` for reproducible signing; otherwise a new random signing key is generated per process. The limited regex-based identifier check is not a comprehensive de-identification or HIPAA-compliance guarantee.

To run Docker Compose, first set a strong `AUDIT_SECRET_KEY` environment variable. Do not expose the demonstration API publicly without access control and appropriate deployment security.

## Tests

```bash
python -m pip install -e ".[web,test]"
pytest -v
python -m pip check
node --check web/app.js
```

GitHub Actions runs the Python test suite on Python 3.10–3.12, validates installation, and checks the browser JavaScript syntax.

## License

[MIT](LICENSE).
