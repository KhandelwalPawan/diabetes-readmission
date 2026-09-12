import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOCAL_MODEL_PATH = PROJECT_ROOT / "models" / "pipeline.joblib"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "diabetic_data.csv"

# Runtime & Environment Config
APP_ENV = os.getenv("APP_ENV", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", None)
DIABETES_MODEL_URI = os.getenv("DIABETES_MODEL_URI", None)
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.50"))
HIGH_RISK_THRESHOLD = float(os.getenv("HIGH_RISK_THRESHOLD", "0.70"))

# Target variable
TARGET_VARIABLE = 'readmitted'

# Columns to drop during raw data extraction
LOW_VARIANCE_MEDS = [
    'repaglinide', 'nateglinide', 'chlorpropamide', 'acetohexamide', 'tolbutamide', 
    'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 
    'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 
    'metformin-rosiglitazone', 'metformin-pioglitazone'
]
COLS_TO_DROP = ['weight', 'encounter_id', 'diag_1', 'diag_2', 'diag_3']

# Rows to drop during training data preparation
DROP_ROWS = ['race']

# Data leakage filter (expired or hospice discharges)
LEAKAGE_COL = 'discharge_disposition_id'
LEAKAGE_CODES = [11, 12, 13, 14, 19, 20, 21]

# Fill na values
FILL_WITH_UNKNOWN = ['max_glu_serum', 'A1Cresult', 'medical_specialty', 'payer_code']

# Feature Groups
ORDERED_CATS = ['age', 'max_glu_serum', 'A1Cresult']
BINARY_CATS = ['change', 'diabetesMed']

ALL_MEDS_COLS = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
    'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
    'tolazamide', 'examide', 'citoglipton', 'insulin',
    'glyburide-metformin', 'glipizide-metformin',
    'glimepiride-pioglitazone', 'metformin-rosiglitazone',
    'metformin-pioglitazone'
]
KEPT_MEDS = [col for col in ALL_MEDS_COLS if col not in LOW_VARIANCE_MEDS]
UNORDERED_CATS = ['gender', 'race', 'medical_specialty', 'payer_code'] + KEPT_MEDS

NUMERICAL_COLS = [
    'admission_type_id',
    'discharge_disposition_id',
    'admission_source_id',
    'time_in_hospital',
    'num_lab_procedures',
    'num_procedures',
    'num_medications',
    'number_outpatient',
    'number_emergency',
    'number_inpatient',
    'number_diagnoses'
]

FEATURE_COLS = ORDERED_CATS + BINARY_CATS + UNORDERED_CATS + NUMERICAL_COLS

ORDINAL_ORDER = {
    'age': [
        '[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
        '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'
    ],
    'max_glu_serum': ['Unknown', '>200', '>300', 'Norm'],
    'A1Cresult': ['Unknown', 'Norm', '>7', '>8']
}