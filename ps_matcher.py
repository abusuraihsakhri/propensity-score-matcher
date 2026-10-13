#!/usr/bin/env python3
"""
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
"""

import math
import csv
import json
import sys
import random
from typing import List, Tuple, Dict, Any, Optional


def _validate_data(treatment, covariates, outcome=None, labels=None):
    """Fail explicitly on malformed observational data; never coerce missing values to zero."""
    n = len(treatment)
    if n == 0 or len(covariates) != n:
        raise ValueError("Treatment and covariates must have equal, nonzero lengths")
    if any(t not in (0, 1) for t in treatment):
        raise ValueError("Treatment must contain only 0 and 1")
    if not any(t == 1 for t in treatment) or not any(t == 0 for t in treatment):
        raise ValueError("Both treated and control subjects are required")
    p = len(covariates[0])
    if any(len(row) != p for row in covariates):
        raise ValueError("Every subject must have the same number of covariates")
    if any(not isinstance(v, (int, float)) or not math.isfinite(v)
           for row in covariates for v in row):
        raise ValueError("Covariates must be finite numbers without missing values")
    if outcome is not None:
        if len(outcome) != n or any(
            not isinstance(v, (int, float)) or not math.isfinite(v) for v in outcome
        ):
            raise ValueError("Outcomes must contain one finite number per subject")
    if labels is not None and len(labels) != p:
        raise ValueError("Covariate labels must match the number of covariates")


def _validate_scores(treatment, propensity_scores):
    if len(treatment) != len(propensity_scores):
        raise ValueError("Treatment and propensity scores must have equal lengths")
    if any(t not in (0, 1) for t in treatment):
        raise ValueError("Treatment must contain only 0 and 1")
    if any(not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1
           for p in propensity_scores):
        raise ValueError("Propensity scores must be finite probabilities in [0, 1]")


# ---------------------------------------------------------------------------
# Logistic regression for propensity scores
# ---------------------------------------------------------------------------

def logistic_regression(
    X: List[List[float]],
    y: List[int],
    max_iter: int = 50,
    tol: float = 1e-8,
) -> Dict[str, Any]:
    """
    Fit logistic regression using Iteratively Reweighted Least Squares (IRLS).

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
    """
    n = len(y)
    _validate_data(y, X)
    if not isinstance(max_iter, int) or max_iter < 1 or not math.isfinite(tol) or tol <= 0:
        raise ValueError("max_iter must be positive and tol must be finite and positive")

    # Add intercept column
    X_aug = [[1.0] + row for row in X]
    p = len(X_aug[0])

    # Initialize beta to zeros
    beta = [0.0] * p
    converged = False

    for iteration in range(max_iter):
        # Compute predicted probabilities
        probs = []
        for i in range(n):
            xb = sum(X_aug[i][j] * beta[j] for j in range(p))
            probs.append(_sigmoid(xb))

        # Compute gradient and Hessian
        grad = [0.0] * p
        hess = [[0.0] * p for _ in range(p)]

        for i in range(n):
            w = probs[i] * (1.0 - probs[i])
            r = y[i] - probs[i]
            for j in range(p):
                grad[j] += X_aug[i][j] * r
                for k in range(p):
                    hess[j][k] -= X_aug[i][j] * X_aug[i][k] * w

        # Newton-Raphson update: beta_new = beta - H^{-1} * g
        neg_grad = [-g for g in grad]
        try:
            delta = _solve_linear_system(hess, neg_grad)
        except ValueError as exc:
            if iteration == 0:
                raise ValueError(
                    "Propensity score regression has a singular design matrix; "
                    "remove redundant covariates or reduce model complexity"
                ) from exc
            # Perfect/quasi separation can render the Hessian singular
            # after several iterations. Retain the last finite estimate
            # but make non-convergence explicit in the result.
            break

        beta_new = [beta[j] + delta[j] for j in range(p)]

        # Check convergence
        if all(abs(delta[j]) < tol for j in range(p)):
            beta = beta_new
            converged = True
            break
        beta = beta_new

    # Final probabilities
    probs = []
    for i in range(n):
        xb = sum(X_aug[i][j] * beta[j] for j in range(p))
        probs.append(_sigmoid(xb))

    # Log-likelihood
    ll = 0.0
    for i in range(n):
        pi = max(1e-15, min(1 - 1e-15, probs[i]))
        ll += y[i] * math.log(pi) + (1 - y[i]) * math.log(1 - pi)

    return {
        "coefficients": beta,
        "probabilities": probs,
        "log_likelihood": ll,
        "iterations": iteration + 1,
        "converged": converged,
        "warning": None if converged else (
            "Logistic IRLS did not converge; possible separation or unstable model"
        ),
    }


def estimate_propensity_scores(
    treatment: List[int],
    covariates: List[List[float]],
) -> List[float]:
    """
    Estimate propensity scores using logistic regression.

    Parameters
    ----------
    treatment : list of int
        Treatment indicator (1=treated, 0=control).
    covariates : list of list of float
        Covariate values for each subject.

    Returns
    -------
    list of float : propensity scores (probability of treatment)
    """
    result = logistic_regression(covariates, treatment)
    return result["probabilities"]


# ---------------------------------------------------------------------------
# Matching algorithms
# ---------------------------------------------------------------------------

def nearest_neighbor_match(
    treatment: List[int],
    propensity_scores: List[float],
    caliper: float = 0.2,
    replace: bool = False,
    ratio: int = 1,
) -> Dict[str, Any]:
    """
    Nearest neighbor matching with optional caliper.

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
    """
    _validate_scores(treatment, propensity_scores)
    if not isinstance(ratio, int) or ratio < 1:
        raise ValueError("ratio must be a positive integer")
    if not math.isfinite(caliper) or caliper < 0:
        raise ValueError("caliper must be finite and non-negative")
    n = len(treatment)
    treated_idx = [i for i in range(n) if treatment[i] == 1]
    control_idx = [i for i in range(n) if treatment[i] == 0]

    if not treated_idx or not control_idx:
        return {"matched_pairs": [], "matched_treated": [], "matched_control": [], "n_matched": 0}

    # Caliper and distance must both use logit(PS) units.
    # Caliper = supplied multiple of the sample SD of logit(PS).
    logit_ps = [_logit(max(1e-6, min(1 - 1e-6, ps))) for ps in propensity_scores]
    sd_logit = _std(logit_ps)
    caliper_val = caliper * sd_logit

    matched_pairs = []
    matched_treated = set()
    matched_control = set()
    available_control = set(control_idx)

    # Sort treated by propensity score (optional, helps with greedy matching)
    for ti in treated_idx:
        # Find closest control(s)
        candidates = []
        for ci in available_control:
            dist = abs(logit_ps[ti] - logit_ps[ci])
            if dist <= caliper_val:
                candidates.append((dist, ci))

        candidates.sort()

        for dist, ci in candidates[:ratio]:
            matched_pairs.append((ti, ci))
            matched_treated.add(ti)
            matched_control.add(ci)
            if not replace:
                available_control.discard(ci)

    return {
        "matched_pairs": matched_pairs,
        "matched_treated": sorted(matched_treated),
        "matched_control": sorted(matched_control),
        "n_matched": len(matched_pairs),
    }


def optimal_match(
    treatment: List[int],
    propensity_scores: List[float],
    caliper: float = float('inf'),
) -> Dict[str, Any]:
    """
    Optimal (greedy) matching: match each treated to the closest available control.

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
    """
    _validate_scores(treatment, propensity_scores)
    if math.isnan(caliper) or caliper < 0:
        raise ValueError("caliper must be non-negative")
    n = len(treatment)
    treated_idx = [i for i in range(n) if treatment[i] == 1]
    control_idx = [i for i in range(n) if treatment[i] == 0]

    if not treated_idx or not control_idx:
        return {"matched_pairs": [], "matched_treated": [], "matched_control": [], "n_matched": 0}

    # Match using the same standardized logit distance as nearest neighbor.
    logits = [_logit(max(1e-6, min(1 - 1e-6, ps))) for ps in propensity_scores]
    limit = caliper * _std(logits) if math.isfinite(caliper) else float("inf")
    distances = []
    for ti in treated_idx:
        for ci in control_idx:
            dist = abs(logits[ti] - logits[ci])
            if dist <= limit:
                distances.append((dist, ti, ci))

    # Sort by distance (greedy)
    distances.sort()

    matched_pairs = []
    matched_treated = set()
    matched_control = set()

    for dist, ti, ci in distances:
        if ti not in matched_treated and ci not in matched_control:
            matched_pairs.append((ti, ci))
            matched_treated.add(ti)
            matched_control.add(ci)

    return {
        "matched_pairs": matched_pairs,
        "matched_treated": sorted(matched_treated),
        "matched_control": sorted(matched_control),
        "n_matched": len(matched_pairs),
    }


def stratification_match(
    treatment: List[int],
    propensity_scores: List[float],
    n_strata: int = 5,
) -> Dict[str, Any]:
    """
    Stratification on propensity scores.

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
    """
    _validate_scores(treatment, propensity_scores)
    if not isinstance(n_strata, int) or n_strata < 1 or not treatment:
        raise ValueError("n_strata must be positive and data cannot be empty")
    n = len(treatment)

    # Sort by propensity score and assign strata
    sorted_indices = sorted(range(n), key=lambda i: propensity_scores[i])
    stratum = [0] * n
    for rank, idx in enumerate(sorted_indices):
        s = min(rank * n_strata // n, n_strata - 1)
        stratum[idx] = s

    # Compute within-stratum effects
    stratum_effects = []
    for s in range(n_strata):
        s_treated = [i for i in range(n) if stratum[i] == s and treatment[i] == 1]
        s_control = [i for i in range(n) if stratum[i] == s and treatment[i] == 0]

        if s_treated and s_control:
            mean_t = sum(propensity_scores[i] for i in s_treated) / len(s_treated)
            mean_c = sum(propensity_scores[i] for i in s_control) / len(s_control)
            stratum_effects.append({
                "stratum": s,
                "n_treated": len(s_treated),
                "n_control": len(s_control),
                "mean_ps_treated": mean_t,
                "mean_ps_control": mean_c,
            })
        else:
            stratum_effects.append({
                "stratum": s,
                "n_treated": len(s_treated),
                "n_control": len(s_control),
                "mean_ps_treated": 0.0,
                "mean_ps_control": 0.0,
            })

    return {
        "strata": stratum,
        "stratum_effects": stratum_effects,
        "n_strata": n_strata,
    }


# ---------------------------------------------------------------------------
# Balance assessment
# ---------------------------------------------------------------------------

def standardized_mean_difference(
    treated: List[float],
    control: List[float],
) -> float:
    """
    Compute the standardized mean difference (SMD).

    SMD = |mean_treated - mean_control| / sqrt((var_treated + var_control) / 2)

    SMD < 0.1 indicates good balance.
    SMD > 0.1 indicates imbalance.
    """
    n_t = len(treated)
    n_c = len(control)
    if n_t == 0 or n_c == 0:
        return float("nan")

    mean_t = sum(treated) / n_t
    mean_c = sum(control) / n_c
    var_t = sum((x - mean_t) ** 2 for x in treated) / max(1, n_t - 1)
    var_c = sum((x - mean_c) ** 2 for x in control) / max(1, n_c - 1)

    pooled_sd = math.sqrt((var_t + var_c) / 2.0)
    if pooled_sd == 0:
        return 0.0 if mean_t == mean_c else float("inf")

    return abs(mean_t - mean_c) / pooled_sd


def covariate_balance(
    treatment: List[int],
    covariates: List[List[float]],
    matched_pairs: Optional[List[Tuple[int, int]]] = None,
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Assess covariate balance before and after matching.

    Parameters
    ----------
    treatment : list of int
        Treatment indicator.
    covariates : list of list of float
        Covariate values.
    matched_pairs : list of (treated_idx, control_idx), optional
        If provided, compute balance on matched sample.
    labels : list of str, optional
        Covariate names.

    Returns
    -------
    dict with balance statistics for each covariate.
    """
    _validate_data(treatment, covariates, labels=labels)
    n = len(treatment)
    p = len(covariates[0])

    if labels is None:
        labels = [f"X{i+1}" for i in range(p)]

    treated_idx = [i for i in range(n) if treatment[i] == 1]
    control_idx = [i for i in range(n) if treatment[i] == 0]

    balance = []
    for k in range(p):
        treated_vals = [covariates[i][k] for i in treated_idx]
        control_vals = [covariates[i][k] for i in control_idx]

        smd_before = standardized_mean_difference(treated_vals, control_vals)

        if matched_pairs is not None and len(matched_pairs) > 0:
            # Use the same pre-matching denominator before/after.
            # Each treated subject contributes once, even for variable-ratio matching.
            def sample_variance(values):
                mu = sum(values) / len(values)
                return sum((v - mu) ** 2 for v in values) / max(1, len(values) - 1)

            standardizer = math.sqrt(
                (sample_variance(treated_vals) + sample_variance(control_vals)) / 2
            )
            groups = {}
            for ti, ci in matched_pairs:
                if treatment[ti] != 1 or treatment[ci] != 0:
                    raise ValueError("Matched pairs must contain treated then control indices")
                groups.setdefault(ti, []).append(ci)
            matched_t_mean = sum(covariates[ti][k] for ti in groups) / len(groups)
            matched_c_mean = sum(
                sum(covariates[ci][k] for ci in controls) / len(controls)
                for controls in groups.values()
            ) / len(groups)
            difference = abs(matched_t_mean - matched_c_mean)
            smd_after = (difference / standardizer if standardizer > 0
                         else (0.0 if difference == 0 else float("inf")))
        elif matched_pairs is None:
            smd_after = smd_before
        else:
            smd_after = None  # No matched cohort: post-match balance is undefined.

        balance.append({
            "covariate": labels[k],
            "smd_before": round(smd_before, 4),
            "smd_after": None if smd_after is None else round(smd_after, 4),
            "balanced_before": smd_before < 0.1,
            "balanced_after": smd_after is not None and smd_after < 0.1,
        })

    return {"balance": balance}


# ---------------------------------------------------------------------------
# Treatment effect estimation
# ---------------------------------------------------------------------------

def estimate_treatment_effect(
    outcome: List[float],
    treatment: List[int],
    matched_pairs: List[Tuple[int, int]],
    effect_type: str = "ATT",
) -> Dict[str, Any]:
    """
    Estimate treatment effect from matched pairs.

    Parameters
    ----------
    outcome : list of float
        Outcome values.
    treatment : list of int
        Treatment indicator.
    matched_pairs : list of (treated_idx, control_idx)
        Matched pairs.
    effect_type : str
        "ATT", "ATC", or "ATE".

    Returns
    -------
    dict with effect estimate, SE, CI.
    """
    if effect_type != "ATT":
        raise ValueError("Matched treated-control differences estimate ATT, not ATC or ATE")
    if not matched_pairs:
        return {"effect": 0.0, "se": None, "ci_lower": None,
                "ci_upper": None, "n_pairs": 0, "effect_type": effect_type}
    if len(outcome) != len(treatment) or any(not math.isfinite(x) for x in outcome):
        raise ValueError("Outcome must have one finite number per subject")
    if any(treatment[ti] != 1 or treatment[ci] != 0 for ti, ci in matched_pairs):
        raise ValueError("Each matched pair must contain treated then control indices")

    # A treated subject with multiple controls contributes once to ATT.
    groups = {}
    for ti, ci in matched_pairs:
        groups.setdefault(ti, []).append(outcome[ci])
    diffs = [
        outcome[ti] - sum(controls) / len(controls)
        for ti, controls in groups.items()
    ]
    mean_diff = sum(diffs) / len(diffs)
    n_pairs = len(matched_pairs)
    # Pair-level independent normal CI is inappropriate for reused controls
    # and variable-ratio matching. Return descriptive ATT without a false SE.
    independent = n_pairs == len(groups) == len({ci for _, ci in matched_pairs})
    if independent and len(diffs) > 1:
        var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (len(diffs) - 1)
        se = math.sqrt(var_diff / len(diffs))
        ci_lo, ci_hi = mean_diff - 1.96 * se, mean_diff + 1.96 * se
        t_stat = mean_diff / se if se > 0 else None
        p_value = 2 * (1 - _norm_cdf(abs(t_stat))) if t_stat is not None else None
    else:
        se = ci_lo = ci_hi = t_stat = p_value = None
    return {
        "effect": round(mean_diff, 6),
        "se": None if se is None else round(se, 6),
        "ci_lower": None if ci_lo is None else round(ci_lo, 6),
        "ci_upper": None if ci_hi is None else round(ci_hi, 6),
        "t_statistic": None if t_stat is None else round(t_stat, 4),
        "p_value": None if p_value is None else round(p_value, 6),
        "n_pairs": n_pairs,
        "n_matched_treated": len(groups),
        "effect_type": effect_type,
    }


# ---------------------------------------------------------------------------
# Full matching pipeline
# ---------------------------------------------------------------------------

def propensity_score_match(
    treatment: List[int],
    covariates: List[List[float]],
    outcome: Optional[List[float]] = None,
    method: str = "nearest",
    caliper: float = 0.2,
    replace: bool = False,
    ratio: int = 1,
    n_strata: int = 5,
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Full propensity score matching pipeline.

    Parameters
    ----------
    treatment : list of int
        Treatment indicator (1=treated, 0=control).
    covariates : list of list of float
        Covariate values.
    outcome : list of float, optional
        Outcome values for treatment effect estimation.
    method : str
        "nearest", "optimal", or "stratification".
    caliper : float
        Caliper for nearest neighbor matching.
    replace : bool
        Match with replacement.
    ratio : int
        Matching ratio.
    n_strata : int
        Number of strata for stratification.
    labels : list of str, optional
        Covariate labels.

    Returns
    -------
    dict with propensity scores, matching results, balance, and effect estimates.
    """
    _validate_data(treatment, covariates, outcome, labels)
    if not isinstance(ratio, int) or ratio < 1:
        raise ValueError("ratio must be a positive integer")
    # Step 1: Estimate propensity scores
    ps_fit = logistic_regression(covariates, treatment)
    ps = ps_fit["probabilities"]

    # Step 2: Match
    if method == "nearest":
        match_result = nearest_neighbor_match(treatment, ps, caliper=caliper, replace=replace, ratio=ratio)
    elif method == "optimal":
        match_result = optimal_match(treatment, ps, caliper=caliper)
    elif method == "stratification":
        strat_result = stratification_match(treatment, ps, n_strata=n_strata)
        match_result = {"matched_pairs": [], "strata": strat_result}
    else:
        raise ValueError(f"Unknown method: {method}")

    # Step 3: Balance assessment
    matched_pairs = match_result.get("matched_pairs", [])
    balance = covariate_balance(treatment, covariates, matched_pairs, labels)

    # Step 4: Treatment effect (if outcome provided)
    effect = None
    if outcome is not None and matched_pairs:
        effect = estimate_treatment_effect(outcome, treatment, matched_pairs)

    return {
        "propensity_scores": ps,
        "ps_model_converged": ps_fit["converged"],
        "ps_model_warning": ps_fit["warning"],
        "matching": match_result,
        "balance": balance,
        "effect": effect,
        "method": method,
        "n_treated": sum(treatment),
        "n_control": len(treatment) - sum(treatment),
    }


# ---------------------------------------------------------------------------
# CSV batch processing
# ---------------------------------------------------------------------------

def process_csv(input_path: str, output_path: str, method: str = "nearest", caliper: float = 0.2) -> Dict[str, Any]:
    """Process a CSV file for propensity score matching."""
    with open(input_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not fieldnames:
        raise ValueError("CSV file has no headers")

    treatment_col = _find_col(fieldnames, ["treatment", "treated", "group", "arm", "exposure"])
    if treatment_col is None:
        raise ValueError(f"No treatment column found. Expected one of: treatment, treated, group, arm, exposure. Got: {fieldnames}")

    outcome_col = _find_col(fieldnames, ["outcome", "y", "result", "response"])
    covariate_cols = [c for c in fieldnames if c != treatment_col and c != outcome_col]

    def numeric(row, column, row_number):
        raw = row.get(column)
        try:
            value = float(raw)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid or missing numeric value in column '{column}', row {row_number}") from exc
        if not math.isfinite(value):
            raise ValueError(f"Non-finite numeric value in column '{column}', row {row_number}")
        return value

    treatment = [numeric(r, treatment_col, i + 2) for i, r in enumerate(rows)]
    outcome = ([numeric(r, outcome_col, i + 2) for i, r in enumerate(rows)]
               if outcome_col is not None else None)
    covariates = [[numeric(r, c, i + 2) for c in covariate_cols]
                  for i, r in enumerate(rows)]

    result = propensity_score_match(
        treatment, covariates, outcome,
        method=method, caliper=caliper, labels=covariate_cols
    )

    # Write output
    out_fields = ["subject", "treatment", "propensity_score", "matched_to", "stratum"]
    out_rows = []
    for i, r in enumerate(rows):
        row = {
            "subject": str(i),
            "treatment": str(treatment[i]),
            "propensity_score": str(round(result["propensity_scores"][i], 6)),
            "matched_to": "",
            "stratum": "",
        }
        # Find match
        for ti, ci in result["matching"].get("matched_pairs", []):
            if i == ti:
                row["matched_to"] = str(ci)
            elif i == ci:
                row["matched_to"] = str(ti)
        if "strata" in result["matching"]:
            row["stratum"] = str(result["matching"]["strata"].get("strata", [None])[i] if isinstance(result["matching"]["strata"], dict) else "")
        out_rows.append(row)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(out_rows)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    """Sigmoid function with overflow protection."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def _logit(p: float) -> float:
    """Logit function: log(p / (1-p))."""
    return math.log(p / (1.0 - p))


def _std(x: List[float]) -> float:
    """Standard deviation."""
    n = len(x)
    if n < 2:
        return 0.0
    mean = sum(x) / n
    var = sum((xi - mean) ** 2 for xi in x) / (n - 1)
    return math.sqrt(var)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _find_col(fieldnames: List[str], candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in fieldnames}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _solve_linear_system(A: List[List[float]], b: List[float]) -> List[float]:
    """Solve Ax = b using Gaussian elimination with partial pivoting."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]

    for col in range(n):
        max_row = col
        for row in range(col + 1, n):
            if abs(M[row][col]) > abs(M[max_row][col]):
                max_row = row
        M[col], M[max_row] = M[max_row], M[col]

        if abs(M[col][col]) < 1e-15:
            raise ValueError("Singular matrix")

        for row in range(col + 1, n):
            factor = M[row][col] / M[col][col]
            for j in range(col, n + 1):
                M[row][j] -= factor * M[col][j]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = M[i][n]
        for j in range(i + 1, n):
            x[i] -= M[i][j] * x[j]
        x[i] /= M[i][i]

    return x


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summary(
    treatment: List[int],
    covariates: List[List[float]],
    outcome: Optional[List[float]] = None,
    labels: Optional[List[str]] = None,
    method: str = "nearest",
    caliper: float = 0.2,
    replace: bool = False,
) -> str:
    """Return a formatted summary using the requested matching configuration."""
    result = propensity_score_match(treatment, covariates, outcome,
                                    labels=labels, method=method,
                                    caliper=caliper, replace=replace)

    lines = []
    lines.append("Propensity Score Matching Analysis")
    lines.append("=" * 60)
    lines.append(f"  Treated: {result['n_treated']}   Control: {result['n_control']}")
    lines.append(f"  Method: {result['method']}")
    lines.append(f"  Matched pairs: {result['matching'].get('n_matched', 0)}")
    if result["ps_model_warning"]:
        lines.append(f"  Model warning: {result['ps_model_warning']}")
    lines.append("")
    lines.append("  Covariate Balance:")
    lines.append(f"  {'Covariate':<15} {'SMD Before':>12} {'SMD After':>12} {'Balanced':>10}")
    lines.append("  " + "-" * 52)
    for b in result["balance"]["balance"]:
        bal = "Yes" if b["balanced_after"] else "No"
        after = f"{b['smd_after']:.4f}" if b["smd_after"] is not None else "N/A"
        lines.append(f"  {b['covariate']:<15} {b['smd_before']:>12.4f} {after:>12} {bal:>10}")

    if result["effect"]:
        lines.append("")
        lines.append(f"  Treatment Effect ({result['effect']['effect_type']}):")
        lines.append(f"    Effect: {result['effect']['effect']:.4f}")
        if result["effect"]["se"] is None:
            lines.append("    SE / CI: unavailable for non-independent or insufficient pairs")
        else:
            lines.append(f"    SE:     {result['effect']['se']:.4f}")
            lines.append(f"    Approx. 95% CI: [{result['effect']['ci_lower']:.4f}, {result['effect']['ci_upper']:.4f}]")
            if result["effect"]["p_value"] is not None:
                lines.append(f"    Approx. normal p-value: {result['effect']['p_value']:.6f}")

    return "\n".join(lines)
