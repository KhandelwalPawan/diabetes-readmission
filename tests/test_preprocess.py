import pytest
import sys
sys.path.append('../src')
import preprocess
import pandas as pd
import numpy as np

def test_load_csv_data():

    valid_path = '../data/diabetic_data.csv'
    result = preprocess.load_data(valid_path)
    assert isinstance(result, pd.DataFrame)

def test_load_invalid_data():
    
    invalid_path = "missing.csv"
    with pytest.raises(FileNotFoundError):
        preprocess.load_data(invalid_path)

@pytest.mark.parametrize("path", ["data.txt", "data.xls", "data.xlsx", "empty/"])
def test_load_invalid_data_extension(path):
    
    with pytest.raises(ValueError):
        preprocess.load_data(path)

def test_drop_columns():
    
    #sample dataframe
    df_dict = {
        'col0': [1, 2, 3, 4, 5],
        'col1': ['cat', 'mouse', 'fish', 'horse', 'cows'],
        'col2': [1.2, 2.0, 5.4, 3.8, 9.9],
        'col3': ['ironman', 'hulk', 'thor', 'thanos', 'strange']
    }

    df = pd.DataFrame(df_dict)
    cols_to_drop = ['col0', 'col2']
    new_df = preprocess.drop_columns(df, cols_to_drop)
    expected_df = df[['col1', 'col3']]
    pd.testing.assert_frame_equal(new_df, expected_df) 

def test_drop_nulls():
    df_dict = {
        'col0': [1, 2, 3, 4, 5],
        'col1': ['cat', None, 'fish', 'horse', 'cows'],
        'col2': [1.2, 2.0, 5.4, 3.8, 9.9],
        'col3': ['ironman', 'hulk', 'thor', 'thanos', None]
    }
    
    df = pd.DataFrame(df_dict)
    rows_to_drop = ['col1', 'col3']
    new_df = preprocess.drop_nulls(df, rows_to_drop)
    expected_df = df.dropna(subset= rows_to_drop)
    pd.testing.assert_frame_equal(new_df, expected_df)

def test_fill_nulls():
    df_dict = {
        'col0': [1, 2, 3, 4, 5],
        'col1': ['cat', None, 'fish', 'horse', 'cows'],
        'col2': [1.2, 2.0, 5.4, 3.8, 9.9],
        'col3': ['ironman', 'hulk', 'thor', 'thanos', None]
    }
    
    df = pd.DataFrame(df_dict)
    cols_to_fill = ['col1', 'col3']
    new_df = preprocess.fill_nulls(df, cols_to_fill)
    expected_df = df.fillna({col: 'Unknown' for col in cols_to_fill})
    pd.testing.assert_frame_equal(new_df, expected_df)

def test_encode_target():
    df_dict = {
        'col0': [1, 2, 3, 4, 5],
        'col1': ['cat', None, 'fish', 'horse', 'cows'],
        'col2': [1.2, 2.0, 5.4, 3.8, 9.9],
        'col3': ['ironman', 'hulk', 'thor', 'thanos', None],
        'readmitted': ['>30', '<30', '<30', 'No', '>30']
    }

    df = pd.DataFrame(df_dict)
    expected_df = np.where(df['readmitted'] == '<30', 1, 0)
    new_df = preprocess.encode_target(df, 'readmitted')
    assert new_df['readmitted'].tolist() == expected_df.tolist()

def test_filter_leakage():
    df_dict = {
        'discharge_disposition_code': [11, 22, 13, 4, 15],
        'col1': ['cat', None, 'fish', 'horse', 'cows'],
        'col2': [1.2, 2.0, 5.4, 3.8, 9.9],
        'col3': ['ironman', 'hulk', 'thor', 'thanos', None],
        'readmitted': ['>30', '<30', '<30', 'No', '>30']
    }
    df = pd.DataFrame(df_dict)
    leakage_codes = [11, 12, 13, 14, 19, 20, 21]
    new_df = preprocess.filter_leakage(df, leakage_codes, 'discharge_disposition_code')
    expected_df = df[~df['discharge_disposition_code'].isin(leakage_codes)]
    pd.testing.assert_frame_equal(new_df, expected_df)