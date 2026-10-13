#!/usr/bin/env python3
"""
Tests for Propensity Score Matcher.
"""
import math
import os
import sys
import tempfile

import pytest

from ps_matcher import (
    logistic_regression,
    estimate_propensity_scores,
    nearest_neighbor_match,
    optimal_match,
    stratification_match,
    standardized_mean_difference,
    covariate_balance,
    estimate_treatment_effect,
    propensity_score_match,
    summary,
    process_csv,
    _sigmoid,
    _logit,
)


# ---------------------------------------------------------------------------
# Logistic regression
# ---------------------------------------------------------------------------

class TestLogisticRegression:
    def test_simple_fit(self):
        """Basic logistic regression."""
        X = [[0.5], [1.0], [1.5], [2.0], [2.5]]
        y = [0, 0, 1, 1, 1]
        result = logistic_regression(X, y)
        assert "coefficients" in result
        assert "probabilities" in result
        assert len(result["probabilities"]) == 5

    def test_probabilities_bounded(self):
        """Probabilities should be in [0, 1]."""
        X = [[0.5], [1.0], [1.5], [2.0], [2.5], [3.0]]
        y = [0, 0, 0, 1, 1, 1]
        result = logistic_regression(X, y)
        for p in result["probabilities"]:
            assert 0.0 <= p <= 1.0

    def test_separation(self):
        """Clear separation should give extreme probabilities."""
        X = [[-10], [-9], [-8], [8], [9], [10]]
        y = [0, 0, 0, 1, 1, 1]
        result = logistic_regression(X, y)
        # Low X -> low probability, high X -> high probability
        assert result["probabilities"][0] < 0.1
        assert result["probabilities"][-1] > 0.9

    def test_convergence(self):
        """Should converge in reasonable iterations."""
        X = [[0.5], [1.0], [1.5], [2.0], [2.5]]
        y = [0, 0, 1, 1, 1]
        result = logistic_regression(X, y)
        assert result["iterations"] <= 50

    def test_multivariate(self):
        """Logistic regression with multiple covariates."""
        X = [[0.5, 65], [1.2, 70], [0.8, 55], [1.5, 60], [0.3, 72], [1.1, 68]]
        y = [1, 0, 1, 1, 0, 1]
        result = logistic_regression(X, y)
        # 3 coefficients: intercept + 2 covariates
        assert len(result["coefficients"]) == 3

    def test_mismatched_lengths_raises(self):
        """Mismatched X and y should raise ValueError."""
        with pytest.raises(ValueError):
            logistic_regression([[1], [2]], [1])


# ---------------------------------------------------------------------------
# Propensity score estimation
# ---------------------------------------------------------------------------

class TestPropensityScores:
    def test_estimate_ps(self):
        """Propensity scores should be probabilities."""
        treatment = [1, 1, 0, 0, 1, 0]
        covariates = [[0.5], [1.2], [0.8], [1.5], [0.3], [1.1]]
        ps = estimate_propensity_scores(treatment, covariates)
        assert len(ps) == 6
        for p in ps:
            assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# Matching algorithms
# ---------------------------------------------------------------------------

class TestNearestNeighbor:
    def test_basic_matching(self):
        """Basic nearest neighbor matching."""
        treatment = [1, 1, 0, 0, 0]
        ps = [0.8, 0.6, 0.7, 0.3, 0.5]
        result = nearest_neighbor_match(treatment, ps, caliper=0.5)
        assert result["n_matched"] > 0
        assert len(result["matched_pairs"]) > 0

    def test_caliper_restriction(self):
        """Caliper should restrict matches."""
        treatment = [1, 0, 0]
        ps = [0.9, 0.1, 0.2]
        # Very tight caliper
        result = nearest_neighbor_match(treatment, ps, caliper=0.01)
        # Might not match due to large distance
        assert result["n_matched"] <= 1

    def test_no_replacement(self):
        """Without replacement, each control used at most once."""
        treatment = [1, 1, 0]
        ps = [0.7, 0.8, 0.75]
        result = nearest_neighbor_match(treatment, ps, caliper=1.0, replace=False)
        # Only 1 control available, so at most 1 match
        assert result["n_matched"] <= 1

    def test_with_replacement(self):
        """With replacement, controls can be reused."""
        treatment = [1, 1, 0]
        ps = [0.7, 0.8, 0.75]
        result = nearest_neighbor_match(treatment, ps, caliper=1.1, replace=True)
        # Both treated can match to same control
        assert result["n_matched"] == 2


class TestOptimalMatch:
    def test_optimal_matching(self):
        """Optimal matching should find good matches."""
        treatment = [1, 1, 0, 0, 0]
        ps = [0.8, 0.6, 0.7, 0.3, 0.5]
        result = optimal_match(treatment, ps)
        assert result["n_matched"] > 0

    def test_optimal_no_treated(self):
        """No treated subjects should return empty."""
        result = optimal_match([0, 0, 0], [0.1, 0.2, 0.3])
        assert result["n_matched"] == 0


class TestStratification:
    def test_stratification(self):
        """Stratification should assign strata."""
        treatment = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        ps = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.15, 0.25]
        result = stratification_match(treatment, ps, n_strata=3)
        assert "strata" in result
        assert len(result["strata"]) == 10
        assert result["n_strata"] == 3


# ---------------------------------------------------------------------------
# Balance assessment
# ---------------------------------------------------------------------------

class TestBalance:
    def test_smd_identical(self):
        """Identical distributions should have SMD ≈ 0."""
        treated = [1.0, 2.0, 3.0, 4.0, 5.0]
        control = [1.0, 2.0, 3.0, 4.0, 5.0]
        smd = standardized_mean_difference(treated, control)
        assert smd < 1e-10

    def test_smd_different(self):
        """Very different distributions should have large SMD."""
        treated = [10.0, 11.0, 12.0, 13.0, 14.0]
        control = [1.0, 2.0, 3.0, 4.0, 5.0]
        smd = standardized_mean_difference(treated, control)
        assert smd > 1.0

    def test_smd_threshold(self):
        """SMD < 0.1 indicates good balance."""
        treated = [1.0, 2.0, 3.0]
        control = [1.01, 2.01, 3.01]
        smd = standardized_mean_difference(treated, control)
        assert smd < 0.1

    def test_covariate_balance_structure(self):
        """Balance check should return proper structure."""
        treatment = [1, 1, 0, 0]
        covariates = [[0.5, 65], [1.2, 70], [0.8, 55], [1.5, 60]]
        result = covariate_balance(treatment, covariates, labels=["age", "income"])
        assert "balance" in result
        assert len(result["balance"]) == 2
        assert "smd_before" in result["balance"][0]
        assert "balanced_before" in result["balance"][0]


# ---------------------------------------------------------------------------
# Treatment effect
# ---------------------------------------------------------------------------

class TestTreatmentEffect:
    def test_effect_estimation(self):
        """Treatment effect from matched pairs."""
        outcome = [10.0, 12.0, 8.0, 9.0, 15.0, 7.0]
        treatment = [1, 1, 0, 0, 1, 0]
        matched_pairs = [(0, 2), (1, 3), (4, 5)]
        result = estimate_treatment_effect(outcome, treatment, matched_pairs)
        # Effect = mean(10-8, 12-9, 15-7) = mean(2, 3, 8) = 13/3 ≈ 4.333
        assert abs(result["effect"] - 13.0 / 3.0) < 0.01
        assert result["n_pairs"] == 3

    def test_effect_ci(self):
        """CI should contain the effect estimate."""
        outcome = [10.0, 12.0, 8.0, 9.0, 15.0, 7.0]
        treatment = [1, 1, 0, 0, 1, 0]
        matched_pairs = [(0, 2), (1, 3), (4, 5)]
        result = estimate_treatment_effect(outcome, treatment, matched_pairs)
        assert result["ci_lower"] <= result["effect"]
        assert result["ci_upper"] >= result["effect"]

    def test_no_pairs(self):
        """No pairs should return zero effect."""
        result = estimate_treatment_effect([1, 2], [1, 0], [])
        assert result["effect"] == 0.0
        assert result["n_pairs"] == 0


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_nearest_pipeline(self):
        """Full nearest neighbor matching pipeline."""
        treatment = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        covariates = [[0.5, 65], [1.2, 70], [0.8, 55], [1.5, 60], [0.3, 72],
                      [1.1, 68], [0.9, 75], [1.4, 58], [0.6, 80], [1.0, 62]]
        outcome = [12.5, 10.2, 8.1, 9.5, 14.0, 7.8, 11.3, 6.5, 13.1, 5.2]
        result = propensity_score_match(treatment, covariates, outcome, method="nearest")
        assert "propensity_scores" in result
        assert "matching" in result
        assert "balance" in result

    def test_stratification_pipeline(self):
        """Full stratification pipeline."""
        treatment = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        covariates = [[0.5], [1.2], [0.8], [1.5], [0.3], [1.1], [0.9], [1.4], [0.6], [1.0]]
        result = propensity_score_match(treatment, covariates, method="stratification")
        assert "strata" in result["matching"]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_output(self):
        """Summary should return formatted string."""
        treatment = [1, 1, 0, 0, 1, 0]
        covariates = [[0.5], [1.2], [0.8], [1.5], [0.3], [1.1]]
        outcome = [12.5, 10.2, 8.1, 9.5, 14.0, 7.8]
        s = summary(treatment, covariates, outcome)
        assert isinstance(s, str)
        assert "Propensity" in s


# ---------------------------------------------------------------------------
# CSV processing
# ---------------------------------------------------------------------------

class TestCSVProcessing:
    def test_batch_process(self, tmp_path):
        """Batch CSV processing should produce output."""
        csv_in = tmp_path / "input.csv"
        csv_out = tmp_path / "output.csv"
        csv_in.write_text(
            "treatment,age,income,outcome\n1,65,50000,12.5\n1,70,60000,10.2\n0,55,45000,8.1\n0,60,55000,9.5\n1,72,70000,14.0\n0,68,48000,7.8\n",
            encoding="utf-8"
        )
        result = process_csv(str(csv_in), str(csv_out))
        assert csv_out.exists()
        assert result["n_treated"] == 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_sigmoid(self):
        """Sigmoid function at known values."""
        assert abs(_sigmoid(0) - 0.5) < 1e-10
        assert _sigmoid(100) > 0.99
        assert _sigmoid(-100) < 0.01

    def test_logit(self):
        """Logit function at known values."""
        assert abs(_logit(0.5)) < 1e-10
        assert _logit(0.9) > 0
        assert _logit(0.1) < 0

    def test_sigmoid_logit_inverse(self):
        """Sigmoid and logit should be inverse functions."""
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert abs(_sigmoid(_logit(p)) - p) < 1e-10


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_all_treated(self):
        """All treated should return no matches."""
        result = nearest_neighbor_match([1, 1, 1], [0.5, 0.6, 0.7])
        assert result["n_matched"] == 0

    def test_all_control(self):
        """All control should return no matches."""
        result = nearest_neighbor_match([0, 0, 0], [0.5, 0.6, 0.7])
        assert result["n_matched"] == 0

    def test_large_dataset(self):
        """Performance test with larger dataset."""
        import random
        random.seed(42)
        n = 100
        treatment = [random.choice([0, 1]) for _ in range(n)]
        covariates = [[random.uniform(0, 1)] for _ in range(n)]
        result = propensity_score_match(treatment, covariates)
        assert len(result["propensity_scores"]) == n
