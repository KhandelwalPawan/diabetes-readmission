import argparse
from pathlib import Path

import joblib
import pandas as pd

from src import config
from src.preprocess import preprocess
from src.validation import validate_tabular_data


def predict_batch(
    input_path: str,
    output_path: str,
    model_path: str = None,
    decision_threshold: float = None,
    high_risk_threshold: float = None,
) -> pd.DataFrame:
    """
    Scores a batch of patient encounter records:
    1. Validates input schema against clinical data contracts
    2. Runs batch scoring using the unified pipeline
    3. Stratifies risk tiers and saves enriched results
    """
    thresh = decision_threshold if decision_threshold is not None else config.DECISION_THRESHOLD
    high_thresh = high_risk_threshold if high_risk_threshold is not None else config.HIGH_RISK_THRESHOLD

    m_path = Path(model_path or config.DEFAULT_LOCAL_MODEL_PATH)
    if not m_path.exists():
        raise FileNotFoundError(f"Trained model pipeline not found at: {m_path}")

    print(f"Loading pipeline from: {m_path}")
    pipeline = joblib.load(m_path)

    print(f"Reading input batch file: {input_path}")
    df_raw = pd.read_csv(input_path, na_values=['?'], low_memory=False)

    # Validate incoming batch against clinical data contract
    is_valid, errors = validate_tabular_data(df_raw, is_training=False, raise_on_error=False)
    if not is_valid:
        print(f"Warning: Batch data had contract warnings: {errors[:3]}")

    # Extract feature columns
    feature_cols = [c for c in config.FEATURE_COLS if c in df_raw.columns]
    df_features = df_raw[feature_cols].copy()

    print(f"Running inference for {len(df_features)} records...")
    probs = pipeline.predict_proba(df_features)[:, 1]
    preds = (probs >= thresh).astype(int)

    risk_tiers = []
    for p in probs:
        if p >= high_thresh:
            risk_tiers.append("High")
        elif p >= thresh:
            risk_tiers.append("Moderate")
        else:
            risk_tiers.append("Low")

    df_out = df_raw.copy()
    df_out["readmission_probability"] = probs.round(4)
    df_out["readmission_prediction"] = preds
    df_out["risk_tier"] = risk_tiers

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_p, index=False)
    print(f"Successfully saved batch predictions to: {out_p}")
    print(f"Predicted positive rate: {preds.mean():.2%} ({preds.sum()} patients)")
    return df_out


def main():
    parser = argparse.ArgumentParser(description="Batch Inference for Diabetes Readmissions")
    parser.add_argument("--input", type=str, default=None, help="Input CSV path")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--model-path", type=str, default=None, help="Model pipeline path")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold")
    parser.add_argument("--dry-run", action="store_true", help="Score first 100 records of dataset")
    args = parser.parse_args()

    if args.dry_run or not args.input:
        print("Executing predict_batch in dry-run mode using sample from default dataset...")
        df_sample = pd.read_csv(config.DEFAULT_DATA_PATH, nrows=100)
        sample_in = config.PROJECT_ROOT / "data" / "sample_batch_input.csv"
        df_sample.to_csv(sample_in, index=False)
        in_path = str(sample_in)
        out_path = str(config.PROJECT_ROOT / "data" / "sample_batch_output.csv")
    else:
        in_path = args.input
        out_path = args.output or str(config.PROJECT_ROOT / "data" / "predictions.csv")

    predict_batch(
        input_path=in_path,
        output_path=out_path,
        model_path=args.model_path,
        decision_threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
