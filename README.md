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
The instances represent hospitalized patient records diagnosed with diabetes. The dataset represents ten years (1999-2008) of clinical care at 130 US hospitals and integrated delivery networks. It includes over 50 features representing patient and hospital outcomes.

## Project Structure

The project is divided into several folders each designed for a specific function.

- API Folder: contains ```main.py``` that contains the logic to implement FastAPI and return the result back to the user.
- Data Folder: has the dataset and the pdf for information about the data.
- Notebooks Folder: has the ```eda.ipynb``` that contains the early exploratory data analysis.
- Src Folder:
1. ```config.py``` that contains all the necessary constants figured out after EDA.
2. ```preprocess.py``` that contains the functions to load and clean the data for the model.
3. ```train.py``` that contains the model training code.

- Test Folder: contains ```test_preprocess.py``` that has tests for the preporcess functions.

