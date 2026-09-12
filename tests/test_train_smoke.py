import pytest
from pathlib import Path
from src import config
from src.train import train


def test_train_dry_run(tmp_path):
    if not Path(config.DEFAULT_DATA_PATH).exists():
        pytest.skip("Dataset diabetic_data.csv not present locally.")

    test_config_file = tmp_path / "test_train_config.yaml"
    test_pipeline_path = tmp_path / "test_pipeline.joblib"
    test_metadata_path = tmp_path / "test_metadata.json"

    import yaml
    cfg = {
        "data": {
            "raw_data_path": str(config.DEFAULT_DATA_PATH),
            "target_variable": config.TARGET_VARIABLE,
            "group_col": "patient_nbr",
        },
        "split": {"test_size": 0.3, "random_state": 42},
        "model": {"n_estimators": 5, "max_depth": 2, "class_weight": "balanced", "random_state": 42, "n_jobs": 1},
        "evaluation": {"decision_threshold": 0.5, "high_risk_threshold": 0.7},
        "mlflow": {"experiment_name": "Test Experiment", "tracking_uri": None, "artifact_path": "model_pipeline", "registered_model_name": None},
        "export": {
            "local_pipeline_path": str(test_pipeline_path),
            "metadata_path": str(test_metadata_path),
        },
    }
    with open(test_config_file, "w") as f:
        yaml.dump(cfg, f)

    result = train(
        config_path=str(test_config_file),
        dry_run=True,
    )
    assert "metrics" in result
    assert "roc_auc" in result["metrics"]
    assert "local_path" in result
    assert Path(result["local_path"]).exists()
    assert Path(test_pipeline_path).exists()
    assert Path(test_metadata_path).exists()
