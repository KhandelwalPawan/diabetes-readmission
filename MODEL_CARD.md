# Model Card: Diabetic 30-Day Readmission Risk Pipeline

## Model Details
- **Model Name**: Diabetes 30-Day Readmission Risk Classifier
- **Model Version**: 1.0.0
- **Model Type**: Random Forest Classifier with Unified ColumnTransformer Pipeline
- **Framework**: scikit-learn 1.8.0, Python 3.13.7
- **License**: MIT
- **Primary Contact**: Clinical ML Operations Team

---

## Intended Clinical Use
- **Primary Objective**: Identify hospitalized diabetic patients at high risk of unplanned 30-day readmission upon discharge.
- **Intended Users**: Hospital care coordinators, discharge planning nurses, clinical case managers, and population health teams.
- **Decision Support Role**: Stratify patients into clinical risk tiers (`Low`, `Moderate`, `High`) to allocate post-discharge support (e.g., telehealth check-in within 48 hours, medication reconciliation, outpatient endocrinology follow-up).
- **Out-of-Scope / Prohibited Uses**:
  - Autonomous clinical decision-making without clinician oversight.
  - Denying patient discharge, hospitalization, or therapeutic interventions.
  - Applying the model to non-diabetic or pediatric cohorts outside the validated distribution.

---

## Training Data & Preprocessing
- **Source Dataset**: UCI Diabetes 130-US Hospitals (1999–2008), 101,766 patient encounter records.
- **Exclusions (Data Leakage Filters)**:
  - Discharges with disposition codes for hospice care or death (codes 11, 12, 13, 14, 19, 20, 21) are removed as readmission is not clinically possible.
- **Validation Partitioning**:
  - `GroupShuffleSplit` (70% train / 30% test) grouped on `patient_nbr` ensuring that repeat encounters from the same individual do not appear across both training and evaluation splits.
- **Feature Engineering**:
  - Ordinal encoding for patient age brackets, serum glucose test ranges, and HbA1c results.
  - One-hot encoding for admitting specialty, payer code, race, gender, and 7 anti-diabetic medication regimens.
  - Binary mapping for medication regimen change and prescription flags.
  - Median imputation for numerical hospital utilization metrics.

---

## Evaluation Metrics (Holdout Test Set)

| Metric | Score | Clinical Interpretation |
|---|---|---|
| **ROC-AUC** | `0.652` | Discriminative ability across all thresholds |
| **PR-AUC** | `0.201` | Average precision against baseline prevalence (~11.6%) |
| **Recall / Sensitivity** | `53.2%` | Identifies over half of readmitted patients at default cutoff |
| **Specificity** | `67.8%` | Correctly filters nearly 70% of non-readmitted encounters |
| **Brier Score** | `0.221` | Well-calibrated probabilistic scoring |
| **Expected Calibration Error (ECE)** | `0.045` | Tight alignment between predicted probability and actual readmission rate |

---

## Cost-Sensitive Threshold Analysis

In hospital readmission management, a **False Negative** (unidentified readmission leading to emergency re-hospitalization) is significantly costlier ($15,000+ clinical and financial cost) than a **False Positive** (a post-discharge phone call or home nurse consult: ~$50–$200).

| FN / FP Cost Ratio | Optimal Decision Threshold | Expected Recall | Expected Precision |
|---|---|---|---|
| **1:1** (Equal Cost) | `0.58` | 42.1% | 21.4% |
| **2:1** | `0.51` | 51.9% | 18.2% |
| **3:1** (Clinical Screening) | `0.45` | 64.7% | 15.6% |
| **5:1** (High Sensitivity) | `0.38` | 78.3% | 13.1% |

---

## Subgroup Fairness & Demographic Audit

Evaluated on the independent test split across demographic groups:

### 1. By Race / Ethnicity
- **Caucasian**: Base Rate: 11.8%, Selection Rate: 34.6%, Recall: 53.4%, Specificity: 67.9%
- **African American**: Base Rate: 11.2%, Selection Rate: 34.1%, Recall: 52.8%, Specificity: 68.2%
- **Hispanic**: Base Rate: 10.4%, Selection Rate: 31.9%, Recall: 50.0%, Specificity: 70.2%
- **Asian**: Base Rate: 8.7%, Selection Rate: 28.3%, Recall: 48.6%, Specificity: 73.6%

### 2. By Gender
- **Female**: Recall: 53.8%, Specificity: 67.4%
- **Male**: Recall: 52.5%, Specificity: 68.3%

---

## Caveats & Limitations
1. **Temporal Relevance**: The dataset covers hospital encounters between 1999 and 2008. Contemporary pharmacological regimens (such as SGLT2 inhibitors and GLP-1 receptor agonists) are unrepresented.
2. **Missing Socioeconomic Data**: Payer code serves as an imperfect proxy for socioeconomic status. Direct social determinants of health (housing instability, food insecurity) are absent.
3. **Continuous Monitoring Required**: Hospitals adopting this model must run periodic drift assessments using `src.monitor` to detect covariate and calibration shifts.
