import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src import config
from src.preprocess import build_pipeline, create_preprocessor


def make_sample_dataframe(n_rows: int = 10):
    """Creates a sample DataFrame matching the 27 feature columns."""
    data = {
        'age': ['[60-70)'] * n_rows,
        'max_glu_serum': ['None'] * n_rows,
        'A1Cresult': ['>8'] * n_rows,
        'change': ['Ch'] * n_rows,
        'diabetesMed': ['Yes'] * n_rows,
        'gender': ['Female'] * n_rows,
        'race': ['Caucasian'] * n_rows,
        'medical_specialty': ['InternalMedicine'] * n_rows,
        'payer_code': ['MC'] * n_rows,
        'metformin': ['Steady'] * n_rows,
        'glimepiride': ['No'] * n_rows,
        'glipizide': ['No'] * n_rows,
        'glyburide': ['No'] * n_rows,
        'pioglitazone': ['No'] * n_rows,
        'rosiglitazone': ['No'] * n_rows,
        'insulin': ['Steady'] * n_rows,
        'admission_type_id': [1] * n_rows,
        'discharge_disposition_id': [1] * n_rows,
        'admission_source_id': [7] * n_rows,
        'time_in_hospital': [3] * n_rows,
        'num_lab_procedures': [40] * n_rows,
        'num_procedures': [1] * n_rows,
        'num_medications': [12] * n_rows,
        'number_outpatient': [0] * n_rows,
        'number_emergency': [0] * n_rows,
        'number_inpatient': [1] * n_rows,
        'number_diagnoses': [7] * n_rows,
    }
    return pd.DataFrame(data)


def test_preprocessor_transform():
    preprocessor = create_preprocessor()
    df = make_sample_dataframe(5)
    transformed = preprocessor.fit_transform(df)
    assert transformed.shape[0] == 5
    assert transformed.shape[1] > 0
    assert not np.isnan(transformed).any()


def test_unified_pipeline_fit_predict():
    df = make_sample_dataframe(20)
    y = np.array([0, 1] * 10)

    rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    pipeline = build_pipeline(rf)

    pipeline.fit(df, y)

    preds = pipeline.predict(df)
    probs = pipeline.predict_proba(df)

    assert len(preds) == 20
    assert probs.shape == (20, 2)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_pipeline_handles_unseen_and_missing_categories():
    df_train = make_sample_dataframe(10)
    y = np.array([0, 1] * 5)

    rf = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=42)
    pipeline = build_pipeline(rf)
    pipeline.fit(df_train, y)

    # Test with unseen categorical levels and missing/None values
    df_test = make_sample_dataframe(3)
    df_test.loc[0, 'medical_specialty'] = 'BrandNewUnknownSpecialty'
    df_test.loc[1, 'race'] = None
    df_test.loc[2, 'age'] = '[999-1000)'

    probs = pipeline.predict_proba(df_test)
    assert probs.shape == (3, 2)
    assert not np.isnan(probs).any()
