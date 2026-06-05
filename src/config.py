LOW_VARIANCE_MEDS = ['repaglinide', 'nateglinide', 'chlorpropamide', 'acetohexamide', 'tolbutamide', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']
COLS_TO_DROP = ['weight', 'encounter_id', 'patient_nbr']
LEAKAGE_CODES = [11, 12, 13, 14, 19, 20, 21]
TARGET_VARIABLE = 'readmitted'
FILL_WITH_UNKNOWN = ['max_glu_serum', 'A1Cresult', 'medical_specialty', 'payer_code']
DROP_ROWS = ['race', 'diag_1', 'diag_2', 'diag_3']
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
