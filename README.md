# Propensity Score Matcher

> **Domain:** Clinical Decision Support & Biomedical Computing

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg?logo=fastapi&logoColor=white)
![Audit Trail](https://img.shields.io/badge/Audit-HMAC--SHA256_Tamper--Evident-brightgreen.svg)
![Zero-PHI Guard](https://img.shields.io/badge/Guard-Zero--PHI_Outbound-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)

</div>

---

## Overview

Propensity Score Matcher is a Python library for estimating treatment effects from observational data using propensity score methods. It provides:

- **Logistic regression** for propensity score estimation (IRLS/Newton-Raphson)
- **Matching algorithms:**
  - Nearest neighbor (with caliper)
  - Optimal matching (greedy)
  - Stratification
- **Balance assessment** using standardized mean difference (SMD)
  - SMD < 0.1 indicates good balance
- **Covariate balance checking** before and after matching
- **Treatment effect estimation** (ATT, ATC, ATE)
- **Matching with/without replacement**

The core algorithm (`ps_matcher.py`) uses only Python stdlib — no external dependencies required.

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

### Installation
```bash
pip install -e .
```

### Commands

#### Match - Run propensity score matching
```bash
python cli.py match --treatment 1 1 0 0 1 0 --covariates '[[0.5,65],[1.2,70],[0.8,55],[1.5,60],[0.3,72],[1.1,68]]'
```

With outcome for treatment effect estimation:
```bash
python cli.py match --treatment 1 1 0 0 --covariates '[[0.5],[1.2],[0.8],[1.5]]' --outcome 12.5 10.2 8.1 9.5
```

#### Balance - Check covariate balance
```bash
python cli.py balance --treatment 1 1 0 0 1 0 --covariates '[[0.5],[1.2],[0.8],[1.5],[0.3],[1.1]]'
```

#### PS - Estimate propensity scores only
```bash
python cli.py ps --treatment 1 1 0 0 --covariates '[[0.5],[1.2],[0.8],[1.5]]'
```

#### Batch - Process CSV file
```bash
python cli.py batch --input sample.csv --output results.csv
```

#### Audit - Run audit task through supervisor
```bash
python cli.py audit --task-id TASK-001
```

#### Chat - Supervisory LLM chat
```bash
python cli.py chat "Explain the methodology"
```

#### Verify-audit - Verify HMAC audit trail integrity
```bash
python cli.py verify-audit
```

### Input Data Schema (CSV)

| Field | Description | Requirement |
|:------|:------------|:------------|
| `treatment` | Treatment indicator (1=treated, 0=control) | Required |
| `age` | Covariate | Optional |
| `income` | Covariate | Optional |
| `education` | Covariate | Optional |
| `outcome` | Outcome variable for effect estimation | Optional |

---

## 🛡️ Security & Enterprise Architecture

* **Zero-PHI Outbound Interceptor:** Active regex inspection blocking SSNs, MRNs, phone numbers, and patient identifiers.
* **Tamper-Evident HMAC-SHA256 Audit Trail:** Chained, cryptographically signed logs for every evaluation and state transition. Configure via `AUDIT_SECRET_KEY` environment variable.
* **LLM Integration:** Agnostic integration for local Ollama instances, Claude, OpenAI, and deterministic test mocks.
* **FastAPI REST API:** Exposes endpoints at `/health`, `/metrics`, `/api/audit`, `/api/chat`, and `/api/audit/logs`.
* **Prometheus Metrics:** Operational metrics available via the `/metrics` endpoint.

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
