import pandas as pd
import numpy as np
import config
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# load data

def load_data(path) -> pd.DataFrame:

    if not path.endswith('.csv'):
        raise ValueError(f"Invalid file type: {path}. Only .csv files are allowed!")

    try:
        return pd.read_csv(path, na_values= ['?'], low_memory= False)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at location: {path}. Try again.")
    
# drop columns
    
def drop_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    return df.drop(columns= columns)

def drop_nulls(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    return df.dropna(subset= columns)

def fill_nulls(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    return df.fillna({col: 'Unknown' for col in columns})

def encode_target(df: pd.DataFrame, column) -> pd.DataFrame:
    df[column] = np.where(df[column] == '<30', 1, 0)
    return df

def filter_leakage(df: pd.DataFrame, codes: list, column) -> pd.DataFrame:
    return df[~df[column].isin(codes)]

def binary_mapping(df: pd.DataFrame, binary_cols: list) -> pd.DataFrame:
    
    df = df.copy()

    for col in binary_cols:
        df[col] = np.where(df[col] == 'No', 0, 1)

    return df

def encode_features(ordinal_cols: list, ordinal_order: dict, nominal_cols: list) -> ColumnTransformer:

    ordinal_cats = [ordinal_order[col] for col in ordinal_cols]

    ct = ColumnTransformer(
        transformers= [
            ('ord', OrdinalEncoder(categories= ordinal_cats), ordinal_cols),
            ('ohe', OneHotEncoder(sparse_output= False, handle_unknown= 'ignore'), nominal_cols)
        ],
        remainder= 'passthrough'
    )

    # ct.set_output('pandas')

    return ct

def preprocess(filepath) -> pd.DataFrame:
    
    df = load_data(filepath)
    df = drop_columns(df, config.COLS_TO_DROP + config.LOW_VARIANCE_MEDS)
    df = drop_nulls(df, config.DROP_ROWS)
    df = fill_nulls(df, config.FILL_WITH_UNKNOWN)
    df = encode_target(df, config.TARGET_VARIABLE)
    df = filter_leakage(df, config.LEAKAGE_CODES, config.LEAKAGE_COL)
    df = binary_mapping(df, config.BINARY_CATS)

    return df