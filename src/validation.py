from typing import List, Tuple
import numpy as np
import pandas as pd
from src import config

# Valid category sets for validation
VALID_AGE_BANDS = set(config.ORDINAL_ORDER['age'])
VALID_GENDERS = {'Female', 'Male', 'Unknown/Invalid'}
VALID_MED_STATUS = {'No', 'Steady', 'Up', 'Down'}
VALID_BINARY_CHANGE = {'No', 'Ch'}
VALID_BINARY_DIABETES_MED = {'No', 'Yes'}
VALID_GLU_SERUM = set(config.ORDINAL_ORDER['max_glu_serum']).union({'None'})
VALID_A1C = set(config.ORDINAL_ORDER['A1Cresult']).union({'None'})

# Numerical bounds
NUMERIC_BOUNDS = {
    'time_in_hospital': (1, 14),
    'num_lab_procedures': (1, 150),
    'num_procedures': (0, 15),
    'num_medications': (1, 120),
    'number_outpatient': (0, 100),
    'number_emergency': (0, 100),
    'number_inpatient': (0, 100),
    'number_diagnoses': (1, 20),
}


class DataValidationError(ValueError):
    """Raised when data fails clinical contract validation."""
    pass


def validate_tabular_data(
    df: pd.DataFrame,
    is_training: bool = False,
    raise_on_error: bool = False,
) -> Tuple[bool, List[str]]:
    """
    Validates a pandas DataFrame against the clinical data contract.
    
    Args:
        df: Input DataFrame to validate.
        is_training: If True, also validates target and group column presence.
        raise_on_error: If True, raises DataValidationError on failure.
        
    Returns:
        (is_valid, errors_list)
    """
    errors: List[str] = []

    if df.empty:
        errors.append("Validation Error: Input DataFrame is empty.")
        if raise_on_error:
            raise DataValidationError(errors[0])
        return False, errors

    # Check required feature columns
    required_cols = list(config.FEATURE_COLS)
    if is_training:
        required_cols.extend([config.TARGET_VARIABLE, 'patient_nbr'])

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # Check numeric bounds on existing columns
    for col, (min_val, max_val) in NUMERIC_BOUNDS.items():
        if col in df.columns:
            non_null = df[col].dropna()
            if not pd.api.types.is_numeric_dtype(non_null):
                errors.append(f"Column '{col}' is not numeric.")
                continue
            out_of_bounds = non_null[(non_null < min_val) | (non_null > max_val)]
            if not out_of_bounds.empty:
                errors.append(
                    f"Column '{col}' has {len(out_of_bounds)} values outside valid clinical range [{min_val}, {max_val}]."
                )

    # Check age values
    if 'age' in df.columns:
        invalid_ages = df['age'].dropna()[~df['age'].dropna().isin(VALID_AGE_BANDS)]
        if not invalid_ages.empty:
            errors.append(f"Column 'age' contains {len(invalid_ages)} invalid age bracket values.")

    # Check medication values
    for med in config.KEPT_MEDS:
        if med in df.columns:
            invalid_meds = df[med].dropna()[~df[med].dropna().isin(VALID_MED_STATUS)]
            if not invalid_meds.empty:
                errors.append(f"Medication column '{med}' contains invalid values (expected No, Steady, Up, Down).")

    is_valid = len(errors) == 0
    if not is_valid and raise_on_error:
        raise DataValidationError("; ".join(errors))

    return is_valid, errors
