import numpy as np
import pandas as pd
from tests.test_pipeline import make_sample_dataframe
from src.monitor import (
    compute_psi,
    evaluate_numerical_drift,
    evaluate_missingness_drift,
    run_drift_analysis,
)


def test_compute_psi_identical():
    s1 = pd.Series(["A", "B", "C"] * 100)
    s2 = pd.Series(["A", "B", "C"] * 100)
    psi = compute_psi(s1, s2)
    assert psi < 0.05


def test_compute_psi_shifted():
    s1 = pd.Series(["A"] * 90 + ["B"] * 10)
    s2 = pd.Series(["A"] * 10 + ["B"] * 90)
    psi = compute_psi(s1, s2)
    assert psi >= 0.25


def test_evaluate_numerical_drift():
    df_base = make_sample_dataframe(50)
    df_curr = make_sample_dataframe(50)
    res = evaluate_numerical_drift(df_base, df_curr, ["num_lab_procedures"])
    assert "num_lab_procedures" in res
    assert res["num_lab_procedures"]["drift_detected"] is False


def test_evaluate_missingness_drift():
    df_base = make_sample_dataframe(50)
    df_curr = make_sample_dataframe(50)
    df_curr.loc[:20, "payer_code"] = np.nan
    res = evaluate_missingness_drift(df_base, df_curr, ["payer_code"])
    assert "payer_code" in res
    assert res["payer_code"]["spike_detected"] is True


def test_run_drift_analysis():
    df_base = make_sample_dataframe(50)
    df_curr = make_sample_dataframe(50)
    report = run_drift_analysis(df_base, df_curr)
    assert "summary" in report
    assert "numerical_drift" in report
    assert "categorical_drift" in report
    assert "missingness_drift" in report
    assert report["summary"]["requires_retraining_review"] is False
