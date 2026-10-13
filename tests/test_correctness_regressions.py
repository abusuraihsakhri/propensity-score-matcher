"""Regression tests for statistical correctness, input validation and audit integrity."""
import csv
import math
import pytest

from ps_matcher import (
    nearest_neighbor_match, standardized_mean_difference, covariate_balance,
    estimate_treatment_effect, propensity_score_match, process_csv,
)
from agents.base import AuditTrail
from cli import main


def test_caliper_compares_logit_distance_not_probability_distance():
    result = nearest_neighbor_match([1, 0, 0], [0.9, 0.8, 0.1], caliper=0.2)
    assert result["n_matched"] == 0


@pytest.mark.parametrize("treatment,covariates", [
    ([1, 0, 0.5], [[1], [2], [3]]),
    ([1, 0], [[1], [2, 3]]),
    ([1, 0], [[1], [float("nan")]]),
    ([1, 1], [[1], [2]]),
])
def test_pipeline_rejects_invalid_study_data(treatment, covariates):
    with pytest.raises(ValueError):
        propensity_score_match(treatment, covariates)


def test_pipeline_rejects_mismatched_outcomes_or_labels():
    with pytest.raises(ValueError):
        propensity_score_match([1, 0], [[1], [2]], outcome=[1])
    with pytest.raises(ValueError):
        propensity_score_match([1, 0], [[1], [2]], labels=["x", "y"])


def test_zero_variance_with_different_means_is_not_balanced():
    assert math.isinf(standardized_mean_difference([2, 2], [3, 3]))


def test_unmatched_balance_is_unavailable_not_copied_from_baseline():
    balance = covariate_balance([1, 0], [[1], [2]], matched_pairs=[])
    assert balance["balance"][0]["smd_after"] is None
    assert not balance["balance"][0]["balanced_after"]


def test_invalid_caliper_and_ratio_fail():
    with pytest.raises(ValueError):
        nearest_neighbor_match([1, 0], [0.5, 0.5], caliper=-1)
    with pytest.raises(ValueError):
        nearest_neighbor_match([1, 0], [0.5, 0.5], ratio=0)
    with pytest.raises(ValueError):
        nearest_neighbor_match([1, 0], [float("nan"), 0.5])


def test_variable_ratio_att_weights_each_treated_subject_once():
    outcome = [10, 20, 0, 10, 4]
    result = estimate_treatment_effect(
        outcome, [1, 1, 0, 0, 0], [(0, 2), (0, 4), (1, 3)]
    )
    assert result["effect"] == 9
    assert result["n_matched_treated"] == 2
    assert result["se"] is None


def test_reused_controls_disable_naive_interval():
    result = estimate_treatment_effect([10, 20, 8], [1, 1, 0], [(0, 2), (1, 2)])
    assert result["effect"] == 7
    assert result["ci_lower"] is None


def test_unsupported_ate_and_atc_are_not_mislabelled():
    for kind in ("ATE", "ATC"):
        with pytest.raises(ValueError):
            estimate_treatment_effect([10, 8], [1, 0], [(0, 1)], effect_type=kind)


def test_csv_rejects_missing_covariates(tmp_path):
    source, target = tmp_path / "input.csv", tmp_path / "output.csv"
    source.write_text("treatment,age\n1,\n0,45\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid or missing"):
        process_csv(str(source), str(target))
    assert not target.exists()


def test_audit_entry_signature_is_recomputed():
    trail = AuditTrail(secret_key="fixed-test-key")
    trail.log("worker", "test", "check", {"status": "ok"})
    assert trail.verify_integrity()
    snapshot = trail.get_trail()
    snapshot[0]["event_type"] = "edited"
    assert trail.verify_integrity()  # snapshot cannot mutate the internal ledger
    trail.logs[0]["event_type"] = "edited"
    assert not trail.verify_integrity()


def test_cli_honors_selected_matching_method(capsys):
    args = [
        "match", "--treatment", "1", "0", "1", "0", "1", "0",
        "--covariates", "[[0.5],[0.6],[0.7],[0.8],[0.9],[1.0]]",
        "--method", "stratification",
    ]
    assert main(args) == 0
    assert "Method: stratification" in capsys.readouterr().out


def test_postmatch_smd_uses_baseline_scale_even_for_one_pair():
    balance = covariate_balance(
        [1, 1, 0, 0], [[1.0], [3.0], [2.0], [4.0]], matched_pairs=[(0, 2)]
    )["balance"][0]
    assert abs(balance["smd_after"] - (1 / math.sqrt(2))) < 0.0001


def test_postmatch_smd_uses_one_weight_per_treated():
    balance = covariate_balance(
        [1, 1, 0, 0, 0], [[2.0], [10.0], [2.0], [8.0], [4.0]],
        matched_pairs=[(0, 2), (0, 4), (1, 3)],
    )["balance"][0]
    # Mean treated (2 + 10)/2 = 6, mean matched controls ((2+4)/2 + 8)/2 = 5.5.
    # Variances: treated 32, controls 28/3; pooled SD = sqrt(62/3 / 2)? computed dynamically.
    t, c = [2.0, 10.0], [2.0, 8.0, 4.0]
    sd_t = sum((v - sum(t)/len(t))**2 for v in t) / (len(t)-1)
    sd_c = sum((v - sum(c)/len(c))**2 for v in c) / (len(c)-1)
    expected = 0.5 / math.sqrt((sd_t + sd_c)/2)
    assert abs(balance["smd_after"] - expected) < 0.0001
