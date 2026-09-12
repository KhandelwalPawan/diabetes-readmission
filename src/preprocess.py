import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

import src.config as config


class ClinicalCategoryCleaner(BaseEstimator, TransformerMixin):
    """
    Normalizes missing indicator strings ('?', 'None', None, np.nan, '')
    to 'Unknown' across categorical feature columns.
    """
    def __init__(self, fill_value: str = 'Unknown'):
        self.fill_value = fill_value

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            for col in df.columns:
                df[col] = df[col].replace(['?', 'None', None, np.nan, ''], self.fill_value)
                df[col] = df[col].fillna(self.fill_value)
            return df.to_numpy()
        else:
            arr = np.array(X, dtype=object, copy=True)
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1] if arr.ndim > 1 else 1):
                    idx = (i, j) if arr.ndim > 1 else i
                    val = arr[idx]
                    if val is None or pd.isna(val) or val in ['?', 'None', '']:
                        arr[idx] = self.fill_value
            return arr


class BinaryFeatureMapper(BaseEstimator, TransformerMixin):
    """
    Transforms binary indicator columns ('change', 'diabetesMed').
    Maps 'No', '0', 'false', None, nan to 0; any affirmative to 1.
    """
    def __init__(self, columns=None):
        self.columns = columns or config.BINARY_CATS

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            out = []
            for col in self.columns:
                if col in X.columns:
                    s = X[col].astype(str).str.strip().str.lower()
                    mapped = (~s.isin(['no', '0', 'false', 'none', 'nan', ''])).astype(int)
                    out.append(mapped.values)
                else:
                    out.append(np.zeros(len(X), dtype=int))
            return np.column_stack(out)
        else:
            arr = np.asarray(X)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            out = []
            for j in range(arr.shape[1]):
                s = pd.Series(arr[:, j]).astype(str).str.strip().str.lower()
                mapped = (~s.isin(['no', '0', 'false', 'none', 'nan', ''])).astype(int)
                out.append(mapped.values)
            return np.column_stack(out)


# --- Data Extraction & Cleaning Functions ---

def load_data(path) -> pd.DataFrame:
    """Loads a CSV file and treats '?' as null values."""
    if not str(path).endswith('.csv'):
        raise ValueError(f"Invalid file type: {path}. Only .csv files are allowed!")
    try:
        return pd.read_csv(path, na_values=['?'], low_memory=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at location: {path}. Try again.")


def drop_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Drops specified columns from dataframe."""
    existing_cols = [c for c in columns if c in df.columns]
    return df.drop(columns=existing_cols)


def drop_nulls(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Drops rows with null values in specified columns."""
    return df.dropna(subset=[c for c in columns if c in df.columns])


def fill_nulls(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Fills null values with 'Unknown' in specified columns."""
    return df.fillna({col: 'Unknown' for col in columns if col in df.columns})


def encode_target(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Encodes target variable to binary: 1 for '<30' readmission, 0 otherwise."""
    df = df.copy()
    df[column] = np.where(df[column] == '<30', 1, 0)
    return df


def filter_leakage(df: pd.DataFrame, codes: list, column: str) -> pd.DataFrame:
    """Filters out patients who expired or entered hospice care."""
    return df[~df[column].isin(codes)]


def binary_mapping(df: pd.DataFrame, binary_cols: list) -> pd.DataFrame:
    """Maps binary categorical columns: 'No' -> 0, other -> 1."""
    df = df.copy()
    for col in binary_cols:
        if col in df.columns:
            df[col] = np.where(df[col] == 'No', 0, 1)
    return df


def encode_features(ordinal_cols: list, ordinal_order: dict, nominal_cols: list) -> ColumnTransformer:
    """Constructs a legacy ColumnTransformer for backward compatibility."""
    ordinal_cats = [ordinal_order[col] for col in ordinal_cols]
    ct = ColumnTransformer(
        transformers=[
            ('ord', OrdinalEncoder(categories=ordinal_cats), ordinal_cols),
            ('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), nominal_cols)
        ],
        remainder='passthrough'
    )
    return ct


def create_preprocessor() -> ColumnTransformer:
    """
    Constructs the complete, production-grade ColumnTransformer
    handling ordinal, nominal, binary, and numerical features cleanly.
    """
    ordinal_pipeline = Pipeline(steps=[
        ('cleaner', ClinicalCategoryCleaner(fill_value='Unknown')),
        ('encoder', OrdinalEncoder(
            categories=[config.ORDINAL_ORDER[col] for col in config.ORDERED_CATS],
            handle_unknown='use_encoded_value',
            unknown_value=-1
        ))
    ])

    nominal_pipeline = Pipeline(steps=[
        ('cleaner', ClinicalCategoryCleaner(fill_value='Unknown')),
        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])

    binary_pipeline = Pipeline(steps=[
        ('mapper', BinaryFeatureMapper(columns=config.BINARY_CATS))
    ])

    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('ord', ordinal_pipeline, config.ORDERED_CATS),
            ('nom', nominal_pipeline, config.UNORDERED_CATS),
            ('bin', binary_pipeline, config.BINARY_CATS),
            ('num', numeric_pipeline, config.NUMERICAL_COLS),
        ],
        remainder='drop'
    )
    return preprocessor


def build_pipeline(classifier) -> Pipeline:
    """Constructs an end-to-end Pipeline with preprocessing and classifier."""
    return Pipeline(steps=[
        ('preprocessor', create_preprocessor()),
        ('classifier', classifier)
    ])


def preprocess(filepath: str) -> pd.DataFrame:
    """
    Offline data extraction pipeline that produces cleaned DataFrame
    for training split and model fitting.
    """
    df = load_data(filepath)
    df = drop_columns(df, config.COLS_TO_DROP + config.LOW_VARIANCE_MEDS)
    df = drop_nulls(df, config.DROP_ROWS)
    df = fill_nulls(df, config.FILL_WITH_UNKNOWN)
    df = encode_target(df, config.TARGET_VARIABLE)
    df = filter_leakage(df, config.LEAKAGE_CODES, config.LEAKAGE_COL)
    return df