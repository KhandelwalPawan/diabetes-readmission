from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src import config, preprocess
from src.preprocess import BinaryFeatureMapper, ClinicalCategoryCleaner


def test_load_csv_data():
    valid_path = config.DEFAULT_DATA_PATH
    if Path(valid_path).exists():
        result = preprocess.load_data(str(valid_path))
        assert isinstance(result, pd.DataFrame)
        assert not result.empty


def test_load_invalid_data():
    invalid_path = "non_existent_file.csv"
    with pytest.raises(FileNotFoundError):
        preprocess.load_data(invalid_path)


@pytest.mark.parametrize("path", ["data.txt", "data.xls", "data.xlsx", "empty/"])
def test_load_invalid_data_extension(path):
    with pytest.raises(ValueError):
        preprocess.load_data(path)


def test_drop_columns():
    df = pd.DataFrame({
        'col0': [1, 2, 3, 4, 5],
        'col1': ['cat', 'mouse', 'fish', 'horse', 'cows'],
        'col2': [1.2, 2.0, 5.4, 3.8, 9.9],
        'col3': ['ironman', 'hulk', 'thor', 'thanos', 'strange'],
    })
    cols_to_drop = ['col0', 'col2']
    new_df = preprocess.drop_columns(df, cols_to_drop)
    expected_df = df[['col1', 'col3']]
    pd.testing.assert_frame_equal(new_df, expected_df)


def test_drop_nulls():
    df = pd.DataFrame({
        'col0': [1, 2, 3, 4, 5],
        'col1': ['cat', None, 'fish', 'horse', 'cows'],
        'col2': [1.2, 2.0, 5.4, 3.8, 9.9],
        'col3': ['ironman', 'hulk', 'thor', 'thanos', None],
    })
    rows_to_drop = ['col1', 'col3']
    new_df = preprocess.drop_nulls(df, rows_to_drop)
    expected_df = df.dropna(subset=rows_to_drop)
    pd.testing.assert_frame_equal(new_df, expected_df)


def test_fill_nulls():
    df = pd.DataFrame({
        'col0': [1, 2, 3, 4, 5],
        'col1': ['cat', None, 'fish', 'horse', 'cows'],
        'col2': [1.2, 2.0, 5.4, 3.8, 9.9],
        'col3': ['ironman', 'hulk', 'thor', 'thanos', None],
    })
    cols_to_fill = ['col1', 'col3']
    new_df = preprocess.fill_nulls(df, cols_to_fill)
    expected_df = df.fillna({col: 'Unknown' for col in cols_to_fill})
    pd.testing.assert_frame_equal(new_df, expected_df)


def test_encode_target():
    df = pd.DataFrame({
        'col0': [1, 2, 3, 4, 5],
        'readmitted': ['>30', '<30', '<30', 'No', '>30'],
    })
    new_df = preprocess.encode_target(df, 'readmitted')
    assert new_df['readmitted'].tolist() == [0, 1, 1, 0, 0]


def test_filter_leakage():
    df = pd.DataFrame({
        'discharge_disposition_id': [11, 22, 13, 4, 15],
        'col1': ['cat', 'dog', 'fish', 'horse', 'cow'],
    })
    leakage_codes = [11, 12, 13, 14, 19, 20, 21]
    new_df = preprocess.filter_leakage(df, leakage_codes, 'discharge_disposition_id')
    expected_df = df[~df['discharge_disposition_id'].isin(leakage_codes)]
    pd.testing.assert_frame_equal(new_df, expected_df)


def test_binary_feature_mapper():
    df = pd.DataFrame({
        'change': ['No', 'Ch', 'no', 'CH', None],
        'diabetesMed': ['No', 'Yes', 'no', 'YES', 'No'],
    })
    mapper = BinaryFeatureMapper(columns=['change', 'diabetesMed'])
    out = mapper.fit_transform(df)
    assert out.shape == (5, 2)
    # 'No' -> 0, 'Ch'/'Yes' -> 1
    assert out[0, 0] == 0
    assert out[1, 0] == 1
    assert out[0, 1] == 0
    assert out[1, 1] == 1


def test_clinical_category_cleaner():
    df = pd.DataFrame({
        'col1': ['?', 'None', 'Normal', None, 'Unknown'],
    })
    cleaner = ClinicalCategoryCleaner(fill_value='Unknown')
    out = cleaner.fit_transform(df)
    assert out[0, 0] == 'Unknown'
    assert out[1, 0] == 'Unknown'
    assert out[2, 0] == 'Normal'
    assert out[3, 0] == 'Unknown'
    assert out[4, 0] == 'Unknown'