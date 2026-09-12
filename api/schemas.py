from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class MedicationStatus(str, Enum):
    NO = "No"
    STEADY = "Steady"
    UP = "Up"
    DOWN = "Down"


class GlucoseSerumResult(str, Enum):
    NONE = "None"
    NORM = "Norm"
    GT_200 = ">200"
    GT_300 = ">300"
    UNKNOWN = "Unknown"


class A1CResult(str, Enum):
    NONE = "None"
    NORM = "Norm"
    GT_7 = ">7"
    GT_8 = ">8"
    UNKNOWN = "Unknown"


class ChangeStatus(str, Enum):
    NO = "No"
    CH = "Ch"


class DiabetesMedStatus(str, Enum):
    NO = "No"
    YES = "Yes"


class AgeBand(str, Enum):
    AGE_0_10 = "[0-10)"
    AGE_10_20 = "[10-20)"
    AGE_20_30 = "[20-30)"
    AGE_30_40 = "[30-40)"
    AGE_40_50 = "[40-50)"
    AGE_50_60 = "[50-60)"
    AGE_60_70 = "[60-70)"
    AGE_70_80 = "[70-80)"
    AGE_80_90 = "[80-90)"
    AGE_90_100 = "[90-100)"


class PatientData(BaseModel):
    """
    Validated clinical patient encounter schema for 30-day readmission prediction.
    """
    race: str = Field(default="Caucasian", description="Patient race/ethnicity")
    gender: str = Field(default="Female", description="Patient biological gender")
    age: AgeBand = Field(default=AgeBand.AGE_70_80, description="Patient age decade band")
    admission_type_id: int = Field(default=1, ge=1, le=8, description="Admission type identifier (1=Emergency, 2=Urgent, etc.)")
    discharge_disposition_id: int = Field(default=1, ge=1, le=30, description="Discharge disposition identifier")
    admission_source_id: int = Field(default=7, ge=1, le=25, description="Admission source identifier")
    time_in_hospital: int = Field(default=3, ge=1, le=14, description="Days in hospital (1-14)")
    payer_code: str = Field(default="MC", description="Payer code (e.g., MC, MD, HM, or Unknown)")
    medical_specialty: str = Field(default="InternalMedicine", description="Admitting medical specialty")
    num_lab_procedures: int = Field(default=40, ge=1, le=150, description="Number of lab procedures performed during encounter")
    num_procedures: int = Field(default=0, ge=0, le=10, description="Number of non-lab procedures performed")
    num_medications: int = Field(default=10, ge=1, le=100, description="Number of distinct medications administered")
    number_outpatient: int = Field(default=0, ge=0, le=100, description="Number of outpatient visits in the preceding year")
    number_emergency: int = Field(default=0, ge=0, le=100, description="Number of emergency visits in the preceding year")
    number_inpatient: int = Field(default=0, ge=0, le=100, description="Number of inpatient visits in the preceding year")
    number_diagnoses: int = Field(default=7, ge=1, le=16, description="Number of entered diagnoses")
    max_glu_serum: GlucoseSerumResult = Field(default=GlucoseSerumResult.NONE, description="Serum glucose test result")
    A1Cresult: A1CResult = Field(default=A1CResult.NONE, description="HbA1c test result")
    metformin: MedicationStatus = Field(default=MedicationStatus.NO)
    glimepiride: MedicationStatus = Field(default=MedicationStatus.NO)
    glipizide: MedicationStatus = Field(default=MedicationStatus.NO)
    glyburide: MedicationStatus = Field(default=MedicationStatus.NO)
    pioglitazone: MedicationStatus = Field(default=MedicationStatus.NO)
    rosiglitazone: MedicationStatus = Field(default=MedicationStatus.NO)
    insulin: MedicationStatus = Field(default=MedicationStatus.NO)
    change: ChangeStatus = Field(default=ChangeStatus.NO, description="Change in diabetic medication dosage")
    diabetesMed: DiabetesMedStatus = Field(default=DiabetesMedStatus.YES, description="Prescribed diabetes medication")

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResponse(BaseModel):
    prediction: int = Field(description="Binary readmission risk: 1 (readmission < 30 days), 0 otherwise")
    probability: float = Field(description="Estimated probability of 30-day readmission (0.0 to 1.0)")
    risk_tier: str = Field(description="Clinical risk tier: Low, Moderate, or High")
    decision_threshold: float = Field(description="Decision threshold used for binary classification")
    model_version: Optional[str] = Field(default=None, description="Active model run or version identifier")
    request_id: str = Field(description="Unique request tracing ID")


class HealthResponse(BaseModel):
    status: str = Field(description="Service status ('healthy' or 'degraded')")
    model_loaded: bool = Field(description="Whether ML model pipeline is loaded and ready")
    model_source: Optional[str] = Field(default=None, description="Artifact path or MLflow URI loaded")
    model_version: Optional[str] = Field(default=None, description="Model run ID or version")
    timestamp: str = Field(description="Current server UTC timestamp")
