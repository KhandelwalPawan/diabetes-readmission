# Target variable
TARGET_VARIABLE = 'readmitted'

# Cols to drop
LOW_VARIANCE_MEDS = ['repaglinide', 'nateglinide', 'chlorpropamide', 'acetohexamide', 'tolbutamide', 'acarbose', 'miglitol', 
                     'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'glyburide-metformin', 'glipizide-metformin', 
                     'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']
COLS_TO_DROP = ['weight', 'encounter_id', 'diag_1', 'diag_2', 'diag_3']

# Rows to drop
DROP_ROWS = ['race']

# Data leakage filter
LEAKAGE_COL = 'discharge_disposition_id'
LEAKAGE_CODES = [11, 12, 13, 14, 19, 20, 21]

# Fill na values 
FILL_WITH_UNKNOWN = ['max_glu_serum', 'A1Cresult', 'medical_specialty', 'payer_code']

# Category collapse
ORDERED_CATS = ['age', 'max_glu_serum', 'A1Cresult']
BINARY_CATS = ['change', 'diabetesMed']
ALL_MEDS_COLS = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
       'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
       'tolazamide', 'examide', 'citoglipton', 'insulin',
       'glyburide-metformin', 'glipizide-metformin',
       'glimepiride-pioglitazone', 'metformin-rosiglitazone',
       'metformin-pioglitazone']
KEPT_MEDS = [col for col in ALL_MEDS_COLS if col not in LOW_VARIANCE_MEDS]
UNORDERED_CATS = ['gender', 'race', 'medical_specialty', 'payer_code'] + KEPT_MEDS
ORDINAL_ORDER = {
    'age': ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)',
 '[70-80)', '[80-90)', '[90-100)'],
    'max_glu_serum': ['Unknown', '>200', '>300', 'Norm'],
    'A1Cresult': ['Unknown', 'Norm', '>7', '>8']
}