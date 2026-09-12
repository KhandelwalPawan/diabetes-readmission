import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

import api.main as api_module
from api.main import app
from src.preprocess import build_pipeline
from tests.test_pipeline import make_sample_dataframe


@pytest.fixture(autouse=True)
def ensure_model_loaded():
    """Ensures a valid lightweight pipeline is set on api_module._model during testing."""
    if api_module._model is None:
        df = make_sample_dataframe(10)
        y = [0, 1] * 5
        rf = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=42)
        pipe = build_pipeline(rf)
        pipe.fit(df, y)
        api_module._model = pipe
        api_module._model_source = "test_fixture"
        api_module._model_version = "test-v1"


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "timestamp" in data
    assert "X-Request-ID" in response.headers


def test_predict_endpoint_valid_payload():
    client = TestClient(app)
    payload = {
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
        "diabetesMed": "Yes",
    }
    response = client.post("/predict", json=payload, headers={"X-Request-ID": "test-req-123"})
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["probability"] <= 1.0
    assert data["risk_tier"] in ["Low", "Moderate", "High"]
    assert data["decision_threshold"] == 0.50
    assert data["request_id"] == "test-req-123"
    assert "X-Process-Time" in response.headers


def test_predict_endpoint_invalid_payload():
    client = TestClient(app)
    # Invalid time_in_hospital (> 14) and invalid age band
    payload = {
        "race": "Caucasian",
        "gender": "Female",
        "age": "invalid_age_format",
        "time_in_hospital": 99,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("age" in str(err) for err in errors)
