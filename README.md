# diabetes-readmission

A ML model based on the diabetes readmission data from UCI.

## About

The goal is to determine the early readmission of the patient within 30 days of discharge. So I have created a Model that takes in the patient data and predicts whether they will be readmitted within that time or not. 

## Business Problem

The problem is important for the following reasons:
- Despite high-quality evidence showing improved clinical outcomes for diabetic patients who receive various preventive and therapeutic interventions, many patients do not receive them. This can be partially attributed to arbitrary diabetes management in hospital environments, which fail to attend to glycemic control. 
- Failure to provide proper diabetes care not only increases the managing costs for the hospitals (as the patients are readmitted) but also impacts the morbidity and mortality of the patients, who may face complications associated with diabetes.

## Dataset

The dataset was downloaded from UCI ML repo. 
(https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008)
The instances represent hospitalized patient records diagnosed with diabetes. The dataset represents ten years (1999-2008) of clinical care at 130 US hospitals and integrated delivery networks. It includes over 50 features representing patient and hospital outcomes.

## Project Structure

The project is divided into several folders each designed for a specific function.

- API Folder: contains `main.py` that contains the logic to implement FastAPI and return the result back to the user.
- Data Folder: has the dataset and the pdf for information about the data.
- Notebooks Folder: has the `eda.ipynb` that contains the early exploratory data analysis.
- Src Folder:
1. `config.py` that contains all the necessary constants figured out after EDA.
2. `preprocess.py` that contains the functions to load and clean the data for the model.
3. `train.py` that contains the model training code.

- Test Folder: contains `test_preprocess.py` that has tests for the preprocess functions.

## Setup and Installation

1. Clone the repo: (https://github.com/KhandelwalPawan/diabetes-readmission.git)

2. Create and activate virtual environment: 

```
python -m venv venv
venv\Scripts\activate

```

3. Install Dependencies: 

`pip install -r requirements.txt`

4. Download the dataset from Kaggle (https://www.kaggle.com/datasets/brandao/diabetes) and place it in the data/ folder.

## How to run

1. Run the `train.py` file from inside the `src` folder to run the model for the first time and then to see your mlflow dashboard run 
```mlflow ui```

2. To connect to the API and ingest live data into your model to get a prediction run this from inside `api` folder your terminal:
```uvicorn main:app --reload```

## Model Performance

My current model has the following scores after version 1:

- Precision: 0.18, 
- Recall: 0.54, 
- F1: 0.27, 
- ROC-AUC: 0.61

## Tech Stack

- Python
- Pandas
- Numpy
- Scikit-learn
- Fastapi
- Uvicorn
- MLflow
- Matplotlib
- Seaborn
- Jupyter

