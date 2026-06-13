import sys
sys.path.append('../src')
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import pandas as pd
import preprocess
import config

class PatientData(BaseModel):
    
    race: str
    gender: str
    age: str
    admission_type_id: int
    discharge_disposition_id: int
    admission_source_id: int
    time_in_hospital: int
    payer_code: str
    medical_specialty: str
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    number_diagnoses: int
    max_glu_serum: str
    A1Cresult: str
    metformin: str
    glimepiride: str
    glipizide: str
    glyburide: str
    pioglitazone: str
    rosiglitazone: str
    insulin: str
    change: str
    diabetesMed: str 

app = FastAPI()
mlflow.set_tracking_uri("file:///C:/Users/pawan/Projects/diabetes-readmission/src/mlruns")
model = mlflow.sklearn.load_model("file:///C:/Users/pawan/Projects/diabetes-readmission/src/mlruns/1/models/m-66863eca0e1a41ba9be1bca948200fb5/artifacts")
transformer = mlflow.sklearn.load_model("file:///C:/Users/pawan/Projects/diabetes-readmission/src/mlruns/1/models/m-168f0ee27f0a478b82fd858920953a69/artifacts")

@app.post("/predict")
def predict(patient: PatientData):
    data_dict = patient.model_dump()
    df = pd.DataFrame([data_dict])
    df = preprocess.binary_mapping(df, config.BINARY_CATS)
    transformed_data = transformer.transform(df)
    prediction = model.predict(transformed_data)
    return {"prediction": int(prediction[0])}
