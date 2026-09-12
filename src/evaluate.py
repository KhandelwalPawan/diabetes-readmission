import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

from src import config
from src.preprocess import preprocess


def compute_calibration_error(y_true, y_prob, n_bins: int = 10) -> Dict:
    """Computes Expected Calibration Error (ECE) and bin statistics."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    binids = np.clip(binids, 0, n_bins - 1)

    ece = 0.0
    total_samples = len(y_true)
    for i in range(n_bins):
        mask = binids == i
        if np.any(mask):
            bin_size = np.sum(mask)
            bin_actual = np.mean(y_true[mask])
            bin_predicted = np.mean(y_prob[mask])
            ece += (bin_size / total_samples) * np.abs(bin_actual - bin_predicted)

    return {
        "expected_calibration_error": float(round(ece, 4)),
        "brier_score": float(round(brier_score_loss(y_true, y_prob), 4)),
        "prob_true": [float(round(p, 4)) for p in prob_true],
        "prob_pred": [float(round(p, 4)) for p in prob_pred],
    }


def compute_cost_sensitive_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_ratios: List[float] = [1.0, 2.0, 3.0, 5.0, 10.0],
) -> Dict:
    """
    Evaluates optimal decision thresholds under varying clinical cost ratios.
    Cost Ratio = Cost(False Negative) / Cost(False Positive)
    """
    thresholds = np.linspace(0.10, 0.90, 81)
    results = {}

    for ratio in cost_ratios:
        best_cost = float("inf")
        best_thresh = 0.50
        cost_curve = []

        cost_fp = 1.0
        cost_fn = ratio

        for thresh in thresholds:
            pred = (y_prob >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
            total_cost = (fp * cost_fp) + (fn * cost_fn)
            cost_curve.append({"threshold": float(round(thresh, 2)), "cost": float(total_cost)})

            if total_cost < best_cost:
                best_cost = total_cost
                best_thresh = float(round(thresh, 2))

        # Metrics at optimal threshold
        opt_pred = (y_prob >= best_thresh).astype(int)
        results[f"fn_fp_ratio_{int(ratio)}x"] = {
            "optimal_threshold": best_thresh,
            "min_relative_cost": float(round(best_cost, 1)),
            "recall": float(round(recall_score(y_true, opt_pred, zero_division=0), 4)),
            "precision": float(round(precision_score(y_true, opt_pred, zero_division=0), 4)),
            "f1": float(round(f1_score(y_true, opt_pred, zero_division=0), 4)),
        }

    return results


def compute_subgroup_fairness(
    df_eval: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
) -> Dict:
    """
    Audits model performance and demographic parity across race, gender, and age slices.
    """
    y_pred = (y_prob >= threshold).astype(int)
    slices = {}

    subgroup_cols = ["race", "gender", "age"]
    for col in subgroup_cols:
        if col not in df_eval.columns:
            continue

        slices[col] = {}
        for group in df_eval[col].dropna().unique():
            idx = (df_eval[col] == group).values
            n_sub = int(idx.sum())
            if n_sub < 50:
                continue

            sub_true = y_true[idx]
            sub_pred = y_pred[idx]
            sub_prob = y_prob[idx]

            tn, fp, fn, tp = confusion_matrix(sub_true, sub_pred, labels=[0, 1]).ravel()
            slices[col][str(group)] = {
                "sample_count": n_sub,
                "base_readmission_rate": float(round(sub_true.mean(), 4)),
                "predicted_positive_rate": float(round(sub_pred.mean(), 4)),
                "recall_sensitivity": float(round(tp / (tp + fn) if (tp + fn) > 0 else 0.0, 4)),
                "specificity": float(round(tn / (tn + fp) if (tn + fp) > 0 else 0.0, 4)),
                "precision": float(round(tp / (tp + fp) if (tp + fp) > 0 else 0.0, 4)),
                "roc_auc": float(round(roc_auc_score(sub_true, sub_prob), 4)) if len(np.unique(sub_true)) > 1 else None,
            }

    return slices


def run_evaluation(
    model_path: str = None,
    data_path: str = None,
    output_path: str = None,
    sample_size: int = None,
) -> Dict:
    """Runs complete clinical risk, calibration, and subgroup fairness evaluation."""
    m_path = Path(model_path or config.DEFAULT_LOCAL_MODEL_PATH)
    d_path = Path(data_path or config.DEFAULT_DATA_PATH)
    out_path = Path(output_path or config.PROJECT_ROOT / "reports" / "clinical_evaluation_report.json")

    print(f"Loading pipeline from: {m_path}")
    pipeline = joblib.load(m_path)

    print(f"Loading and preprocessing evaluation data from: {d_path}")
    df = preprocess(str(d_path))

    if sample_size and len(df) > sample_size:
        df = df.head(sample_size).copy()

    # Replicate grouped test holdout split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(df, df[config.TARGET_VARIABLE], df['patient_nbr']))
    test_df = df.iloc[test_idx].copy()

    X_test = test_df.drop(columns=[config.TARGET_VARIABLE, 'patient_nbr'])
    y_test = test_df[config.TARGET_VARIABLE].to_numpy()

    print(f"Scoring {len(X_test)} holdout patient encounters...")
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= config.DECISION_THRESHOLD).astype(int)

    # Standard metrics
    standard_metrics = {
        "roc_auc": float(round(roc_auc_score(y_test, y_prob), 4)),
        "pr_auc": float(round(average_precision_score(y_test, y_prob), 4)),
        "precision": float(round(precision_score(y_test, y_pred, zero_division=0), 4)),
        "recall": float(round(recall_score(y_test, y_pred, zero_division=0), 4)),
        "f1": float(round(f1_score(y_test, y_pred, zero_division=0), 4)),
        "brier_score": float(round(brier_score_loss(y_test, y_prob), 4)),
        "test_sample_count": len(y_test),
        "actual_positives": int(y_test.sum()),
    }

    print("Computing calibration curve & ECE...")
    calibration_report = compute_calibration_error(y_test, y_prob)

    print("Computing cost-sensitive threshold analysis...")
    cost_report = compute_cost_sensitive_thresholds(y_test, y_prob)

    print("Computing demographic fairness audit across slices...")
    fairness_report = compute_subgroup_fairness(test_df, y_test, y_prob, threshold=config.DECISION_THRESHOLD)

    report = {
        "model_path": str(m_path),
        "dataset_path": str(d_path),
        "standard_metrics": standard_metrics,
        "calibration": calibration_report,
        "cost_sensitive_thresholds": cost_report,
        "subgroup_fairness": fairness_report,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Successfully saved clinical evaluation report to: {out_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Clinical Evaluation & Fairness Audit")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()

    run_evaluation(
        model_path=args.model_path,
        data_path=args.data_path,
        output_path=args.output,
        sample_size=args.sample,
    )


if __name__ == "__main__":
    main()
