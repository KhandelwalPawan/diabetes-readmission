import numpy as np
import pandas as pd
from src.evaluate import (
    compute_calibration_error,
    compute_cost_sensitive_thresholds,
    compute_subgroup_fairness,
)


def test_compute_calibration_error():
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1] * 10)
    y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.2, 0.8, 0.3, 0.85] * 10)

    res = compute_calibration_error(y_true, y_prob, n_bins=5)
    assert "expected_calibration_error" in res
    assert "brier_score" in res
    assert 0.0 <= res["expected_calibration_error"] <= 1.0
    assert 0.0 <= res["brier_score"] <= 1.0


def test_compute_cost_sensitive_thresholds():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1] * 10)
    y_prob = np.linspace(0.05, 0.95, 100)

    res = compute_cost_sensitive_thresholds(y_true, y_prob, cost_ratios=[1.0, 3.0, 5.0])
    assert "fn_fp_ratio_1x" in res
    assert "fn_fp_ratio_3x" in res
    assert "fn_fp_ratio_5x" in res
    # Higher FN penalty should result in lower or equal optimal decision threshold
    thresh_1x = res["fn_fp_ratio_1x"]["optimal_threshold"]
    thresh_5x = res["fn_fp_ratio_5x"]["optimal_threshold"]
    assert thresh_5x <= thresh_1x


def test_compute_subgroup_fairness():
    df = pd.DataFrame({
        "race": ["Caucasian"] * 60 + ["AfricanAmerican"] * 60,
        "gender": ["Female"] * 60 + ["Male"] * 60,
        "age": ["[60-70)"] * 120,
    })
    y_true = np.array([0, 1] * 60)
    y_prob = np.array([0.3, 0.8] * 60)

    res = compute_subgroup_fairness(df, y_true, y_prob, threshold=0.50)
    assert "race" in res
    assert "Caucasian" in res["race"]
    assert "AfricanAmerican" in res["race"]
    assert res["race"]["Caucasian"]["recall_sensitivity"] == 1.0
    assert res["race"]["Caucasian"]["specificity"] == 1.0
