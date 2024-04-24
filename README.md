# README

## Overview
This repository contains code for three different tasks using XGBoost:

### Task 1: Data Preparation for Gradient Boosting
Before using XGBoost models, the data must be prepared into the expected format. This task focuses on preparing the data for XGBoost, specifically for regression predictive modeling problems.

### Task 2: Diabetes Prediction with XGBoost
In this task, XGBoost is utilized to predict the onset of diabetes based on certain diagnostic measurements included in the dataset. The dataset used is the Pima Indians Diabetes Database.

### Task 3: Time Series Forecasting with XGBoost
The second part of the code demonstrates the application of XGBoost for time series forecasting. It includes the implementation of a univariate time series forecasting model using XGBoost.



## Requirements
- NumPy
- Pandas
- XGBoost
- Matplotlib (for visualization)
- Scikit-learn

  ## Task 1: Data Preparation for Gradient Boosting
- **Loading Data**: The Iris Flower dataset is loaded using Pandas from a CSV file.
- **Data Preprocessing**: The dataset is prepared into the expected format for XGBoost, which requires numerical values as input. String class labels are encoded into integer classes using `LabelEncoder`.
- **Model Training**: An XGBoost classifier is trained on the prepared dataset.
- **Prediction and Evaluation**: Predictions are made on the test data, and accuracy is computed using the `accuracy_score` function from Scikit-learn.

## Task 2: Diabetes Prediction with XGBoost
- **Loading Data**: The Pima Indians Diabetes Database is loaded using the `loadtxt` function from NumPy.
- **Data Preprocessing**: The dataset is split into features (X) and the target variable (y), followed by splitting the data into train and test sets.
- **Model Training**: An XGBoost classifier is trained on the training dataset using the `XGBClassifier`.
- **Prediction and Evaluation**: Predictions are made on the test data, and accuracy is computed using the `accuracy_score` function from Scikit-learn.

## Task 3: Time Series Forecasting with XGBoost
- **Loading Data**: A time series dataset is loaded using Pandas from a CSV file.
- **Data Preprocessing**: The series data is converted into a supervised learning problem using the `series_to_supervised` function.
- **Model Training and Validation**: Walk-forward validation is performed to evaluate the XGBoost model's performance for time series forecasting.
- **Visualization**: The expected vs predicted values are plotted using Matplotlib to visualize the model's performance.



## Results
- For Task 1: Data preparation is crucial for using XGBoost models effectively. This task focuses on preparing the data into the expected format for regression predictive modeling with XGBoost.
- For Task 2: The XGBoost model achieves a certain level of accuracy in predicting the onset of diabetes based on diagnostic measurements.
- For Task 3: XGBoost demonstrates its effectiveness in time series forecasting, as seen in the visualization of expected vs predicted values.


## Conclusion
- XGBoost is a powerful algorithm suitable for various tasks, including classification, regression, and time series forecasting.
- The provided code demonstrates the versatility and performance of XGBoost in different machine learning scenarios, highlighting its speed and effectiveness in predictive modeling.
