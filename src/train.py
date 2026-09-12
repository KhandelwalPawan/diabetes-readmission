import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
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
from src.preprocess import build_pipeline, preprocess


def load_config(config_path: str = None) -> dict:
    """Loads configuration from YAML file or defaults."""
    default_config_path = config.PROJECT_ROOT / "configs" / "train_config.yaml"
    path_to_use = Path(config_path) if config_path else default_config_path

    if path_to_use.exists():
        with open(path_to_use, "r") as f:
            return yaml.safe_load(f)

    # Built-in fallback config
    return {
        "data": {
            "raw_data_path": str(config.DEFAULT_DATA_PATH),
            "target_variable": config.TARGET_VARIABLE,
            "group_col": "patient_nbr",
        },
        "split": {"test_size": 0.3, "random_state": 42},
        "model": {
            "n_estimators": 100,
            "max_depth": 10,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        },
        "evaluation": {
            "decision_threshold": config.DECISION_THRESHOLD,
            "high_risk_threshold": config.HIGH_RISK_THRESHOLD,
        },
        "mlflow": {
            "experiment_name": "Diabetic Readmission Production",
            "tracking_uri": None,
            "artifact_path": "model_pipeline",
            "registered_model_name": "diabetes-readmission",
        },
        "export": {
            "local_pipeline_path": str(config.DEFAULT_LOCAL_MODEL_PATH),
            "metadata_path": str(config.DEFAULT_METADATA_PATH),
        },
    }


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Calculates comprehensive clinical and statistical classification metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "predicted_positives": int(y_pred.sum()),
        "actual_positives": int(y_true.sum()),
    }
    return metrics


def train(
    config_path: str = None,
    data_path_override: str = None,
    dry_run: bool = False,
    n_estimators_override: int = None,
    max_depth_override: int = None,
) -> dict:
    """
    Executes the reproducible training pipeline:
    1. Loads and preprocesses clinical dataset
    2. Splits with GroupShuffleSplit on patient_nbr to avoid patient leakage
    3. Builds unified Pipeline (preprocessor + classifier)
    4. Evaluates on holdout test set
    5. Logs run, metrics, and unified artifact to MLflow
    6. Saves local fallback pipeline bundle and metadata
    """
    cfg = load_config(config_path)

    raw_data_path = data_path_override or cfg["data"].get("raw_data_path", str(config.DEFAULT_DATA_PATH))
    if not Path(raw_data_path).is_absolute():
        raw_data_path = str(config.PROJECT_ROOT / raw_data_path)

    print(f"Loading and preprocessing data from: {raw_data_path}...")
    df = preprocess(raw_data_path)

    if dry_run:
        print("Dry run mode: using first 1000 records for fast verification.")
        df = df.head(1000).copy()

    group_col = cfg["data"].get("group_col", "patient_nbr")
    target_col = cfg["data"].get("target_variable", config.TARGET_VARIABLE)

    groups = df[group_col]
    X = df.drop(columns=[target_col, group_col])
    y = df[target_col]

    test_size = cfg["split"].get("test_size", 0.3)
    split_seed = cfg["split"].get("random_state", 42)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=split_seed)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(f"Train split: {len(X_train)} samples ({y_train.sum()} positives)")
    print(f"Test split:  {len(X_test)} samples ({y_test.sum()} positives)")

    # Model Hyperparameters
    model_cfg = cfg.get("model", {})
    n_estimators = n_estimators_override or model_cfg.get("n_estimators", 100)
    max_depth = max_depth_override or model_cfg.get("max_depth", 10)
    class_weight = model_cfg.get("class_weight", "balanced")
    model_seed = model_cfg.get("random_state", 42)
    n_jobs = model_cfg.get("n_jobs", -1)

    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=model_seed,
        n_jobs=n_jobs,
    )

    pipeline = build_pipeline(classifier)

    print("Fitting unified scikit-learn pipeline (preprocessor + classifier)...")
    pipeline.fit(X_train, y_train)

    print("Evaluating pipeline on holdout test set...")
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    decision_threshold = cfg.get("evaluation", {}).get("decision_threshold", config.DECISION_THRESHOLD)
    y_pred = (y_prob >= decision_threshold).astype(int)

    metrics = compute_metrics(y_test, y_pred, y_prob)
    print("Holdout Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # MLflow tracking
    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri") or config.MLFLOW_TRACKING_URI
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    experiment_name = mlflow_cfg.get("experiment_name", "Diabetic Readmission Production")
    mlflow.set_experiment(experiment_name)

    artifact_path = mlflow_cfg.get("artifact_path", "model_pipeline")
    registered_model_name = mlflow_cfg.get("registered_model_name", "diabetes-readmission")

    run_id = None
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("class_weight", class_weight)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("decision_threshold", decision_threshold)
        mlflow.log_param("dry_run", dry_run)

        # Log metrics
        for metric_name, val in metrics.items():
            mlflow.log_metric(metric_name, val)

        # Log single unified pipeline artifact
        print("Logging unified pipeline artifact to MLflow...")
        try:
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name if not dry_run else None,
            )
        except Exception as e:
            print(f"MLflow model logging note: {e}")

    # Export local pipeline artifact bundle for resilient offline serving
    export_cfg = cfg.get("export", {})
    local_path = Path(export_cfg.get("local_pipeline_path", config.DEFAULT_LOCAL_MODEL_PATH))
    metadata_path = Path(export_cfg.get("metadata_path", config.DEFAULT_METADATA_PATH))

    local_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving local fallback pipeline artifact to: {local_path}")
    joblib.dump(pipeline, local_path)

    metadata = {
        "run_id": run_id,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "features": list(X_train.columns),
        "metrics": metrics,
        "hyperparameters": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "class_weight": str(class_weight),
            "decision_threshold": decision_threshold,
        },
        "sample_counts": {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        },
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved model metadata to: {metadata_path}")
    return {"metrics": metrics, "run_id": run_id, "local_path": str(local_path)}


def main():
    parser = argparse.ArgumentParser(description="Train Diabetes Readmission Pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML training config")
    parser.add_argument("--data-path", type=str, default=None, help="Path to diabetic_data.csv")
    parser.add_argument("--n-estimators", type=int, default=None, help="Number of trees")
    parser.add_argument("--max-depth", type=int, default=None, help="Maximum tree depth")
    parser.add_argument("--dry-run", action="store_true", help="Run quick verification on a data sample")
    args = parser.parse_args()

    train(
        config_path=args.config,
        data_path_override=args.data_path,
        dry_run=args.dry_run,
        n_estimators_override=args.n_estimators,
        max_depth_override=args.max_depth,
    )


if __name__ == "__main__":
    main()