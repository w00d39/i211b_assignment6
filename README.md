# Diabetes Dataset Regression Analysis

This project performs regression analysis on the diabetes dataset using various machine learning models, including Random Forest, Support Vector Machine (SVM), and Ridge Regression. The goal is to predict the target variable (disease progression) based on the features in the dataset.

## Dataset

The dataset used in this project is the diabetes dataset from the `sklearn.datasets` module. It contains 10 baseline variables, age, sex, body mass index, average blood pressure, and six blood serum measurements, obtained for each of 442 diabetes patients, as well as the target variable, a quantitative measure of disease progression one year after baseline.

## Models

The following models are used in this project:

1. **Random Forest Regressor**
2. **Support Vector Machine (SVM) Regressor**
3. **Ridge Regression**

## Installation

To run this project, you need to have Python and the following libraries installed:

- pandas
- numpy
- scikit-learn

##Code Explanation

Load the Dataset: The diabetes dataset is loaded using sklearn.datasets.load_diabetes().

Data Preprocessing: The dataset is converted to a pandas DataFrame for easier manipulation and analysis.

Train-Test Split: The data is split into training and testing sets with an 80/20 split.

Random Forest Regressor:

A Random Forest Regressor is initialized and trained on the training data.
The model's performance is evaluated using Mean Squared Error (MSE), Explained Variance, and R^2 Score.
Support Vector Machine (SVM) Regressor:

A parameter grid is defined for hyperparameter tuning using GridSearchCV.
The best parameters are found, and the model is trained on the training data.
The model's performance is evaluated using MSE, Explained Variance, and R^2 Score.
Ridge Regression:

Ridge Regression with cross-validation is performed to find the best alpha value.
The model is trained on the training data.
The model's performance is evaluated using MSE, Explained Variance, and R^2 Score.

##Results
The SVM is the clear winner here out of my three models hands down. By tuning it I was able to get 
the lowest MSE, the hightest r2, and the hightest explained variance. The ridge regression was the worst
model by far. The random forest was the second best model but the SVM was the best.

The SVM model was the best because it was able to tune the model to the best parameters for the data utilizing 
a grid search. The random forest was the second best model because I was able to hand tune the model to get the best
metrics I could. The ridge regression was the worst model because it was not able to tune the model to the the results
to my accepted level. 
