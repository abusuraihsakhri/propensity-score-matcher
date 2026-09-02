# Propensity Score Matcher

> **Domain:** Clinical Decision Support & Biomedical Computing  
> **Reference Guidelines & Standards:** `Standard Clinical Formulations & ISO/IEC Quality Frameworks`

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg?logo=fastapi&logoColor=white)
![Audit Trail](https://img.shields.io/badge/Audit-HMAC--SHA256_Tamper--Evident-brightgreen.svg)
![Zero-PHI Guard](https://img.shields.io/badge/Guard-Zero--PHI_Outbound-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)

</div>

---

## 📖 What It Does

Propensity Score Matcher

Real implementation of:
  - Logistic regression for propensity score estimation (IRLS/Newton-Raphson)
  - Matching algorithms:
    - Nearest neighbor (with caliper)
    - Optimal matching (greedy)
    - Stratification
  - Balance assessment: standardized mean difference (SMD)
    - SMD < 0.1: good balance
  - Covariate balance checking
  - ATT, ATC, ATE estimation
  - Matching with/without replacement

Pure Python stdlib — no external dependencies.

---

## ⚙️ Key Capabilities & Algorithmic Modules

### 🔬 Analytical Functions

- **`logistic_regression()`**: Fit logistic regression using Iteratively Reweighted Least Squares (IRLS).

Parameters
----------
X : list of list of float
    Design matrix (each row is a subject, columns are covariates).
    An intercept column is added automatically.
y : list of int
    Binary outcome (0 or 1).
max_iter : int
    Maximum iterations.
tol : float
    Convergence tolerance.

Returns
-------
dict with keys:
    coefficients : estimated β (including intercept)
    probabilities : predicted probabilities (propensity scores)
    log_likelihood : final log-likelihood
    iterations : number of iterations
- **`estimate_propensity_scores()`**: Estimate propensity scores using logistic regression.

Parameters
----------
treatment : list of int
    Treatment indicator (1=treated, 0=control).
covariates : list of list of float
    Covariate values for each subject.

Returns
-------
list of float : propensity scores (probability of treatment)
- **`nearest_neighbor_match()`**: Nearest neighbor matching with optional caliper.

Parameters
----------
treatment : list of int
    Treatment indicator.
propensity_scores : list of float
    Propensity scores.
caliper : float
    Maximum allowed distance for matching (in SD of logit PS).
replace : bool
    Whether to match with replacement.
ratio : int
    Number of control matches per treated subject.

Returns
-------
dict with keys:
    matched_pairs : list of (treated_idx, control_idx) tuples
    matched_treated : indices of treated subjects that were matched
    matched_control : indices of control subjects that were matched
    n_matched : number of matched pairs
- **`optimal_match()`**: Optimal (greedy) matching: match each treated to the closest available control.

This is a greedy approximation — true optimal matching requires the
Hungarian algorithm, but this gives good results for most cases.

Parameters
----------
treatment : list of int
    Treatment indicator.
propensity_scores : list of float
    Propensity scores.
caliper : float
    Maximum allowed distance.

Returns
-------
dict with matched pairs.
- **`stratification_match()`**: Stratification on propensity scores.

Divides subjects into strata based on propensity score quantiles,
then computes treatment effect within each stratum.

Parameters
----------
treatment : list of int
    Treatment indicator.
propensity_scores : list of float
    Propensity scores.
n_strata : int
    Number of strata (default 5).

Returns
-------
dict with keys:
    strata : list of stratum assignments
    stratum_effects : treatment effect estimate per stratum
    overall_effect : pooled treatment effect (ATE)

---

## 💻 CLI Quickstart & Usage

### 1. Guided Interactive Mode
```bash
python cli.py
```

### 2. Direct Parameterized Evaluation
```bash
python cli.py --treatment <value> --covariates <value> --input <value> --output <value>
```

### Parameter Reference
- `--treatment`: Specifies input measurement or parameter value.
- `--covariates`: Specifies input measurement or parameter value.
- `--input`: Specifies input measurement or parameter value.
- `--output`: Specifies input measurement or parameter value.
- `---`: Specifies input measurement or parameter value.
- `--outcome`: Specifies input measurement or parameter value.
- `--method`: Specifies input measurement or parameter value.
- `--caliper`: Specifies input measurement or parameter value.
- `--replace`: Specifies input measurement or parameter value.
- `--labels`: Specifies input measurement or parameter value.

### Input Data Schema

| Field | Description | Requirement |
|:------|:------------|:------------|
| `treatment` | Parameter / observation metric | Required |
| `age` | Parameter / observation metric | Required |
| `income` | Parameter / observation metric | Required |
| `education` | Parameter / observation metric | Required |
| `outcome` | Parameter / observation metric | Required |

---

## 🛡️ Security & Enterprise Architecture

* **Zero-PHI Outbound Interceptor:** Active AST and regex inspection blocking SSNs, MRNs, phone numbers, and patient identifiers.
* **Tamper-Evident HMAC-SHA256 Audit Trail:** Chained, cryptographically signed logs for every evaluation and state transition.
* **Air-Gapped LLM Reasoning Adapter:** Agnostic integration for local Ollama instances (`llama3`, `mistral`), Claude 3.5 Sonnet, GPT-4o, and deterministic test mocks.
* **Active Learning Bayesian Calibration:** Dynamic tracker updating worker reliability weights and monitoring Brier calibration drift.
* **FastAPI & Prometheus Telemetry:** Exposes OpenAPI 3.1 REST endpoints and operational Prometheus metrics (`/metrics`).

---

## 🧪 Testing & Verification

Run the automated test suite:

```bash
pytest -v
```

Execute high-throughput batch simulation benchmarks:

```bash
python simulator.py --tasks 1000 --concurrency 8
```

---

## 🐳 Container Deployment

```bash
docker build -t propensity-score-matcher .
docker run -p 8000:8000 propensity-score-matcher
```
