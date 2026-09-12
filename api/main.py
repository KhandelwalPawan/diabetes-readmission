import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import HealthResponse, PatientData, PredictionResponse
from src import config

# Setup structured logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("diabetes_api")

# Global model state
_model = None
_model_source: Optional[str] = None
_model_version: Optional[str] = None


def load_model_pipeline():
    """
    Dynamically loads the trained scikit-learn Pipeline from:
    1. MLflow Model URI (configured via DIABETES_MODEL_URI env var)
    2. Local joblib fallback file (DEFAULT_LOCAL_MODEL_PATH)
    """
    global _model, _model_source, _model_version

    # Attempt 1: Load from MLflow URI if provided
    if config.DIABETES_MODEL_URI:
        try:
            logger.info(f"Loading model from MLflow URI: {config.DIABETES_MODEL_URI}")
            if config.MLFLOW_TRACKING_URI:
                mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
            _model = mlflow.sklearn.load_model(config.DIABETES_MODEL_URI)
            _model_source = config.DIABETES_MODEL_URI
            _model_version = "mlflow"
            logger.info("Successfully loaded model from MLflow.")
            return _model
        except Exception as e:
            logger.warning(f"Failed to load from MLflow URI ({config.DIABETES_MODEL_URI}): {e}. Falling back to local artifact.")

    # Attempt 2: Load local fallback artifact
    local_path = Path(config.DEFAULT_LOCAL_MODEL_PATH)
    if local_path.exists():
        try:
            logger.info(f"Loading model from local artifact: {local_path}")
            _model = joblib.load(local_path)
            _model_source = str(local_path)
            
            # Load metadata if available
            metadata_path = Path(config.DEFAULT_METADATA_PATH)
            if metadata_path.exists():
                try:
                    with open(metadata_path, "r") as f:
                        meta = json.load(f)
                        _model_version = meta.get("run_id") or meta.get("trained_at")
                except Exception as meta_err:
                    logger.debug(f"Could not read metadata file: {meta_err}")
            if not _model_version:
                _model_version = "local-v1"

            logger.info("Successfully loaded local pipeline artifact.")
            return _model
        except Exception as e:
            logger.error(f"Failed to load local model artifact from {local_path}: {e}")

    logger.warning("No model artifact could be loaded. Service running in degraded mode.")
    return None


def get_model():
    """Returns the loaded model, attempting load if not yet initialized."""
    global _model
    if _model is None:
        load_model_pipeline()
    return _model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load ML pipeline
    load_model_pipeline()
    yield
    # Shutdown
    logger.info("Shutting down Diabetes Readmission API.")


app = FastAPI(
    title="Diabetes 30-Day Readmission Risk API",
    description="Production ML service predicting 30-day hospital readmission risk for diabetic patients.",
    version="1.0.0",
    lifespan=lifespan,
)

# Eagerly initialize model on module import
load_model_pipeline()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_observability_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = request_id
    start_time = time.time()

    response: Response = await call_next(request)

    duration = time.time() - start_time
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{duration:.4f}s"

    logger.info(
        f"Request {request.method} {request.url.path} status={response.status_code} "
        f"duration={duration:.4f}s request_id={request_id}"
    )
    return response


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Service and model health check",
)
def health_check():
    """Returns system health, model availability, and active version info."""
    model = get_model()
    model_loaded = model is not None
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_source=_model_source,
        model_version=_model_version,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    summary="Predict 30-day readmission risk for a patient encounter",
)
def predict(patient: PatientData, request: Request):
    """
    Computes readmission probability, binary classification, and risk tier
    using the unified scikit-learn pipeline.
    """
    model = get_model()
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model pipeline is not loaded or unavailable on this instance.",
        )

    request_id = getattr(request.state, "request_id", uuid.uuid4().hex)

    try:
        # Convert schema to single-row DataFrame
        patient_dict = patient.model_dump()
        df = pd.DataFrame([patient_dict])

        # Prediction probability from unified pipeline
        if hasattr(_model, "predict_proba"):
            proba = float(_model.predict_proba(df)[0, 1])
        else:
            pred_raw = int(_model.predict(df)[0])
            proba = 1.0 if pred_raw == 1 else 0.0

        decision_thresh = config.DECISION_THRESHOLD
        high_risk_thresh = config.HIGH_RISK_THRESHOLD
        prediction = 1 if proba >= decision_thresh else 0

        # Clinical risk stratification
        if proba >= high_risk_thresh:
            risk_tier = "High"
        elif proba >= decision_thresh:
            risk_tier = "Moderate"
        else:
            risk_tier = "Low"

        return PredictionResponse(
            prediction=prediction,
            probability=round(proba, 4),
            risk_tier=risk_tier,
            decision_threshold=decision_thresh,
            model_version=_model_version,
            request_id=request_id,
        )
    except Exception as e:
        logger.exception(f"Error during inference: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )
