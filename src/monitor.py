import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src import config
from src.preprocess import preprocess


def compute_psi(baseline: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """
    Computes Population Stability Index (PSI) between baseline and current series.
    PSI < 0.1: No change
    0.1 <= PSI < 0.25: Moderate shift
    PSI >= 0.25: Significant distribution shift
    """
    base_counts = baseline.value_counts(normalize=True, dropna=False)
    curr_counts = current.value_counts(normalize=True, dropna=False)

    all_cats = list(set(base_counts.index).union(set(curr_counts.index)))
    psi = 0.0
    epsilon = 1e-4

    for cat in all_cats:
        b_pct = base_counts.get(cat, epsilon)
        c_pct = curr_counts.get(cat, epsilon)
        if b_pct == 0:
            b_pct = epsilon
        if c_pct == 0:
            c_pct = epsilon
        psi += (c_pct - b_pct) * np.log(c_pct / b_pct)

    return float(round(psi, 4))


def evaluate_numerical_drift(baseline: pd.DataFrame, current: pd.DataFrame, num_cols: List[str]) -> Dict:
    """Performs two-sample Kolmogorov-Smirnov test on numerical features."""
    results = {}
    for col in num_cols:
        if col in baseline.columns and col in current.columns:
            b_vals = baseline[col].dropna()
            c_vals = current[col].dropna()
            if len(b_vals) < 10 or len(c_vals) < 10:
                continue

            ks_stat, p_value = ks_2samp(b_vals, c_vals)
            drift_detected = bool(p_value < 0.05 and ks_stat > 0.05)

            results[col] = {
                "ks_statistic": float(round(ks_stat, 4)),
                "p_value": float(round(p_value, 5)),
                "baseline_mean": float(round(b_vals.mean(), 2)),
                "current_mean": float(round(c_vals.mean(), 2)),
                "drift_detected": drift_detected,
            }
    return results


def evaluate_categorical_drift(baseline: pd.DataFrame, current: pd.DataFrame, cat_cols: List[str]) -> Dict:
    """Evaluates Population Stability Index across categorical features."""
    results = {}
    for col in cat_cols:
        if col in baseline.columns and col in current.columns:
            b_series = baseline[col].astype(str)
            c_series = current[col].astype(str)

            psi = compute_psi(b_series, c_series)
            status = "Significant Drift" if psi >= 0.25 else ("Moderate Shift" if psi >= 0.10 else "Stable")

            results[col] = {
                "psi": psi,
                "status": status,
                "drift_detected": psi >= 0.25,
            }
    return results


def evaluate_missingness_drift(baseline: pd.DataFrame, current: pd.DataFrame, feature_cols: List[str]) -> Dict:
    """Checks for unexpected shifts in missing value rates."""
    results = {}
    for col in feature_cols:
        if col in baseline.columns and col in current.columns:
            b_null_rate = float(baseline[col].isna().mean())
            c_null_rate = float(current[col].isna().mean())
            diff = abs(c_null_rate - b_null_rate)

            results[col] = {
                "baseline_null_rate": float(round(b_null_rate, 4)),
                "current_null_rate": float(round(c_null_rate, 4)),
                "absolute_diff": float(round(diff, 4)),
                "spike_detected": bool(diff > 0.10),
            }
    return results


def run_drift_analysis(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> Dict:
    """Executes feature drift, missingness drift, and distribution checks."""
    num_results = evaluate_numerical_drift(baseline_df, current_df, config.NUMERICAL_COLS)
    cat_cols = config.ORDERED_CATS + config.BINARY_CATS + config.UNORDERED_CATS
    cat_results = evaluate_categorical_drift(baseline_df, current_df, cat_cols)
    missing_results = evaluate_missingness_drift(baseline_df, current_df, config.FEATURE_COLS)

    drifted_numeric = [k for k, v in num_results.items() if v["drift_detected"]]
    drifted_categorical = [k for k, v in cat_results.items() if v["drift_detected"]]
    spiked_missing = [k for k, v in missing_results.items() if v["spike_detected"]]

    summary = {
        "baseline_records": len(baseline_df),
        "current_records": len(current_df),
        "numerical_drift_count": len(drifted_numeric),
        "categorical_drift_count": len(drifted_categorical),
        "missingness_spike_count": len(spiked_missing),
        "drifted_features": drifted_numeric + drifted_categorical,
        "requires_retraining_review": len(drifted_numeric) + len(drifted_categorical) > 3,
    }

    report = {
        "summary": summary,
        "numerical_drift": num_results,
        "categorical_drift": cat_results,
        "missingness_drift": missing_results,
    }

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Drift report written to: {out_p}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Clinical Data & Feature Drift Monitor")
    parser.add_argument("--baseline", type=str, default=None, help="Path to baseline data CSV")
    parser.add_argument("--current", type=str, default=None, help="Path to current/batch data CSV")
    parser.add_argument("--output", type=str, default=None, help="Path to output drift JSON")
    parser.add_argument("--dry-run", action="store_true", help="Split baseline into two halves for dry run verification")
    args = parser.parse_args()

    base_path = args.baseline or str(config.DEFAULT_DATA_PATH)
    print(f"Loading baseline data from: {base_path}")
    base_df = preprocess(base_path)

    if args.dry_run or not args.current:
        print("Executing drift monitor in dry-run mode (comparing two halves of baseline data)...")
        mid = len(base_df) // 2
        current_df = base_df.iloc[mid: mid + 2000].copy()
        base_df = base_df.iloc[:2000].copy()
    else:
        print(f"Loading current batch data from: {args.current}")
        current_df = preprocess(args.current)

    out_file = args.output or str(config.PROJECT_ROOT / "reports" / "drift_report.json")
    report = run_drift_analysis(base_df, current_df, output_path=out_file)
    print("Drift Summary:")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
