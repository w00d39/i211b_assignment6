import pandas as pd # lil tasting library
import numpy as np #need it to calculate some stuff in the lasso model
import sklearn # actually using this library
from sklearn import datasets, model_selection, metrics # general modules from sklearn
from sklearn import ensemble, linear_model, svm # specific modules from sklearn for ml

"""
The SVM is the clear winner here out of my three models hands down. By tuning it I was able to get 
the lowest MSE, the hightest r2, and the hightest explained variance. The ridge regression was the worst
model by far. The random forest was the second best model but the SVM was the best.

The SVM model was the best because it was able to tune the model to the best parameters for the data utilizing 
a grid search. The random forest was the second best model because I was able to hand tune the model to get the best
metrics I could. The ridge regression was the worst model because it was not able to tune the model to the the results
to my accepted level. 
"""

#load diabetes dataset and seeing names and targets to see what I'm working with
diabetes_data = datasets.load_diabetes()
# print(diabetes_data.feature_names)
# print(diabetes_data.target)


#dataset tasting
diabetes_df = pd.DataFrame(diabetes_data.data, columns=diabetes_data.feature_names)
#print(diabetes_df.describe())

#correlation
diabetes_corr = diabetes_df.corr()
#print(diabetes_corr)

x_data = diabetes_data.data #data
y_data = diabetes_data.target #target

#splitting the data into training and testing sets with an 80/20 split and a random state of 301
x_train, x_test, y_train, y_test = model_selection.train_test_split(x_data, y_data, test_size=0.2, random_state=301)


# #random forest

#I hand tuned the model to get the best metrics I could
diabaetes_forest = ensemble.RandomForestRegressor(
    n_estimators=75, min_samples_split= 2,min_impurity_decrease= 0.2,  random_state=301, warm_start=True,  ccp_alpha=0.8
    )
diabaetes_forest.fit(x_train, y_train) #fitting the data to the model
diabaetes_forest_predictions = diabaetes_forest.predict(x_test) #training the model

#metrics
print("Random Forest Mean Squared Error: ", metrics.mean_squared_error(y_test, diabaetes_forest_predictions))
print("Random Forest Explained Variance: ", metrics.explained_variance_score(y_test, diabaetes_forest_predictions))
print("Random Forest R2 Score: ", metrics.r2_score(y_test, diabaetes_forest_predictions), "\n\n\n")


#support vector machine

# Define the parameter grid for SVM
param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], #the kernel
    'C': [0.1, 1, 10, 100], #regularization parameter
    'degree': [2, 3, 4, 5], #degree of the polynomial kernel
    'gamma': ['scale', 'auto'], #kernel coefficient
    'epsilon': [0.1, 0.2, 0.5, 1.0], #margin of tolerance
}

# Initialize the model
svr = svm.SVR() #support vector regression

# Initialize GridSearchCV with parallel processing
# cv=5 is the number of folds in cross-validation, n_jobs=-1 is for parallel processing, and the rest are the parameters
grid_search = model_selection.GridSearchCV(estimator=svr, param_grid=param_grid, cv=5, n_jobs=-1)

#fitting
grid_search.fit(x_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_


# Train the model with the best parameters
best_svr = grid_search.best_estimator_
best_svr.fit(x_train, y_train) #fitting the data to the model
diabetes_svm_predictions = best_svr.predict(x_test) 

#metrics
print("Tuned SVM R^2 Score: ", metrics.r2_score(y_test, diabetes_svm_predictions)) 
print("Tuned SVM Mean Squared Error: ", metrics.mean_squared_error(y_test, diabetes_svm_predictions))
print("Tuned SVM Explained Variance: ", metrics.explained_variance_score(y_test, diabetes_svm_predictions), "\n\n\n")


# Ridge Regression with cross-validation
alphas = np.logspace(-4, 0, 50) #alphas to test
ridge_cv = linear_model.RidgeCV(alphas=alphas, cv=5) #ridge regression with cross validation
ridge_cv.fit(x_train, y_train) #fitting the data to the model
diabetes_ridge_predictions = ridge_cv.predict(x_test)

#metrics
print("Ridge Regression R^2 Score: ", metrics.r2_score(y_test, diabetes_ridge_predictions))
print("Ridge Regression Mean Squared Error: ", metrics.mean_squared_error(y_test, diabetes_ridge_predictions))
print("Ridge Regression Explained Variance: ", metrics.explained_variance_score(y_test, diabetes_ridge_predictions), "\n\n\n")
