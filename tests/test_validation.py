import pytest
import pandas as pd
from tests.test_pipeline import make_sample_dataframe
from src.validation import validate_tabular_data, DataValidationError


def test_validate_valid_dataframe():
    df = make_sample_dataframe(10)
    is_valid, errors = validate_tabular_data(df)
    assert is_valid is True
    assert len(errors) == 0


def test_validate_empty_dataframe():
    df = pd.DataFrame()
    is_valid, errors = validate_tabular_data(df)
    assert is_valid is False
    assert any("empty" in e.lower() for e in errors)


def test_validate_missing_columns():
    df = make_sample_dataframe(5).drop(columns=['age', 'time_in_hospital'])
    is_valid, errors = validate_tabular_data(df)
    assert is_valid is False
    assert any("Missing required columns" in e for e in errors)


def test_validate_out_of_bounds_values():
    df = make_sample_dataframe(5)
    # time_in_hospital range is 1-14; set 99
    df.loc[0, 'time_in_hospital'] = 99
    is_valid, errors = validate_tabular_data(df)
    assert is_valid is False
    assert any("time_in_hospital" in e for e in errors)


def test_validate_invalid_category():
    df = make_sample_dataframe(5)
    df.loc[0, 'age'] = '[999-1000)'
    is_valid, errors = validate_tabular_data(df)
    assert is_valid is False
    assert any("age" in e for e in errors)


def test_raise_on_error():
    df = pd.DataFrame()
    with pytest.raises(DataValidationError):
        validate_tabular_data(df, raise_on_error=True)
