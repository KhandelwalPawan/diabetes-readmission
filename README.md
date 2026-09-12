# Diabetes 30-Day Readmission Risk ML Pipeline

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange.svg)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/MLflow-3.12.0-blueviolet.svg)](https://mlflow.org/)
[![Tests](https://img.shields.io/badge/tests-20%2F20%20passing-brightgreen.svg)]()

A production-grade machine learning pipeline for predicting 30-day early hospital readmissions for diabetic patients using the UCI Diabetes 130-US Hospitals dataset (1999–2008).

---

## Architecture & Production Upgrades

This project implements an end-to-end reproducible machine learning architecture designed to eliminate training-serving skew and support robust clinical risk stratification:

1. **Unified Pipeline Artifact**:
   - Preprocessing (missing value imputation, binary mapping, ordinal encoding, and one-hot encoding) is bundled directly with the classifier into a single `sklearn.pipeline.Pipeline`.
   - The serving service receives raw data and executes the exact same transformation pipeline as training.
2. **Environment & Config-Driven Serving**:
   - Zero hard-coded local absolute paths.
   - Dynamic model resolution via `DIABETES_MODEL_URI` (MLflow Model Registry or run artifact) with graceful local fallback to `models/pipeline.joblib`.
3. **Reproducible & Configurable Training**:
   - Fully decoupled CLI training interface (`python -m src.train --config configs/train_config.yaml`).
   - Grouped train/test splitting (`GroupShuffleSplit` on `patient_nbr`) preventing patient leakage across splits.
   - Automatic registration and version tracking in the MLflow Model Registry (`diabetes-readmission`).
4. **Clinical Data Validation & API Observability**:
   - Strict Pydantic schemas validating clinical categories, age bands, medication states, and numeric physiological ranges.
   - Standard `/health` liveness/readiness endpoint reporting model load state and active model version.
   - Risk probability scoring, clinical risk tier stratification (`Low`, `Moderate`, `High`), request ID tracing (`X-Request-ID`), and latency metrics (`X-Process-Time`).
5. **Standardized Packaging & Test Suite**:
   - Standard `pyproject.toml` configuration enabling editable installs (`pip install -e .`).
   - 20 unit, integration, pipeline, and training smoke tests passing from repo root via `pytest`.

---

## Project Structure

```text
diabetes-readmission/
├── api/
│   ├── main.py                  # FastAPI service with health, predict & middleware
│   └── schemas.py               # Pydantic clinical input/output contracts & Enums
├── configs/
│   └── train_config.yaml        # Declarative training and evaluation configuration
├── data/
│   ├── diabetic_data.csv        # UCI Diabetes raw dataset
│   └── description.pdf          # Dataset attribute definitions
├── models/
│   ├── pipeline.joblib          # Standalone serialized scikit-learn pipeline
│   └── model_metadata.json      # Run metrics, hyperparameters, and feature metadata
├── notebooks/
│   └── eda.ipynb                # Exploratory Data Analysis
├── src/
│   ├── __init__.py              # Package init
│   ├── config.py                # Global constants, feature sets, and runtime settings
│   ├── preprocess.py            # Custom transformers & unified pipeline constructor
│   └── train.py                 # Configurable CLI training pipeline & MLflow logger
├── tests/
│   ├── test_api.py              # FastAPI endpoint tests
│   ├── test_pipeline.py         # Unified pipeline fit/transform tests
│   ├── test_preprocess.py       # Data cleaning and custom transformer tests
│   └── test_train_smoke.py      # End-to-end training smoke test
├── pyproject.toml               # PEP 518/621 build and package definition
├── pytest.ini                   # Pytest discovery and runner settings
├── requirements.txt             # Pinned production and development dependencies
└── README.md
```

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/KhandelwalPawan/diabetes-readmission.git
   cd diabetes-readmission
   ```

2. **Create and activate virtual environment**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies and package**:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

---

## Training the Pipeline

Train the pipeline with default configuration:
```bash
python -m src.train --config configs/train_config.yaml
```

Run a fast dry-run verification:
```bash
python -m src.train --dry-run
```

View MLflow experiment dashboard:
```bash
mlflow ui
```

### Evaluation Metrics (Holdout Test Split)
- **ROC-AUC**: `0.652`
- **PR-AUC**: `0.201`
- **Recall**: `0.532`
- **Specificity**: `0.678`
- **F1-Score**: `0.267`
- **Brier Score**: `0.221`

---

## Running the API Service

Start the FastAPI application using Uvicorn from the repository root:
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Health Check Endpoint
```bash
curl -X GET "http://localhost:8000/health"
```
**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_source": ".../models/pipeline.joblib",
  "model_version": "v1",
  "timestamp": "2026-09-04T18:43:11.033Z"
}
```

### Risk Prediction Endpoint
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: req-001" \
  -d '{
    "race": "Caucasian",
    "gender": "Female",
    "age": "[60-70)",
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
    "time_in_hospital": 4,
    "payer_code": "MC",
    "medical_specialty": "InternalMedicine",
    "num_lab_procedures": 45,
    "num_procedures": 1,
    "num_medications": 14,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 1,
    "number_diagnoses": 9,
    "max_glu_serum": "None",
    "A1Cresult": ">8",
    "metformin": "Steady",
    "glimepiride": "No",
    "glipizide": "No",
    "glyburide": "No",
    "pioglitazone": "No",
    "rosiglitazone": "No",
    "insulin": "Steady",
    "change": "Ch",
    "diabetesMed": "Yes"
  }'
```
**Response**:
```json
{
  "prediction": 1,
  "probability": 0.6555,
  "risk_tier": "Moderate",
  "decision_threshold": 0.5,
  "model_version": "v1",
  "request_id": "req-001"
}
```

---

## Running Tests

Run the complete 20-test test suite from the repository root:
```bash
pytest -v
```

Tests verify:
- Preprocessing and custom transformer lifecycle logic (`tests/test_preprocess.py`)
- Unified Pipeline end-to-end fit, transform, and edge cases (`tests/test_pipeline.py`)
- FastAPI endpoints, validation rules, and error codes (`tests/test_api.py`)
- Training pipeline smoke execution (`tests/test_train_smoke.py`)

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `DIABETES_MODEL_URI` | MLflow model URI or model registry reference | `None` (uses local bundle) |
| `MLFLOW_TRACKING_URI` | MLflow tracking server location | `None` (uses local `mlruns`) |
| `DECISION_THRESHOLD` | Probability cutoff for positive classification | `0.50` |
| `HIGH_RISK_THRESHOLD`| Probability cutoff for high-risk stratification | `0.70` |
| `APP_ENV` | Environment identifier (`development`, `staging`, `production`) | `development` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |
