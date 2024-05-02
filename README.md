# KM6312-INFORMATION_MINING

## Overview
This repository contains the source code and datasets used in the development of predictive models for sleep disorders based on sleep health and lifestyle data. The project leverages various machine learning algorithms to understand and predict sleep-related health issues accurately.

## Data Description
Original Data: The primary dataset used in this project is sleep_health_and_lifestyle_dataset, which includes various metrics pertinent to sleep quality and general lifestyle habits.
Processed Data: The cleaned and processed data is available in df_clean.csv, which was used for building the machine learning models.

## Feature Engineering
Feature engineering was performed on the sleep_health_and_lifestyle_dataset to prepare the data for modeling. See Future Engineering.ipynb.

## Models
Seven different predictive models were developed and evaluated in this study:
K-Nearest Neighbors (KNN): See knn.py
Logistic Regression (LR): See LR.py
Support Vector Machine (SVM): See Svm.py
Random Forest (RF): See rf.py
XGBoost and Deep Metric Learning (DML): See XGBoost+DML.ipynb

Each script and notebook contains the model implementation, along with detailed comments explaining the process and choices made during model development.

