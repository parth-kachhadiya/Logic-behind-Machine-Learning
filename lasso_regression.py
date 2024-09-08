"""
Generally ridge regression is usefull for solving overfitting problems
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV # For cross validation

# NOTE: For cross validation
from sklearn.model_selection import cross_val_score 


data = fetch_california_housing()

dataset = pd.DataFrame(data.data,columns=data.feature_names)

X = dataset
Y = data.target

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=42)

scalar = StandardScaler()
x_train = scalar.fit_transform(x_train)

# print(scalar.inverse_transform(x_train))

model = LinearRegression()
model.fit(x_train,y_train)
cv = cross_val_score(model,x_train,y_train,scoring='neg_mean_squared_error',cv=5)

predicted_value = np.array(model.predict(x_test))

# Lasso regression algorithm ----------------------------------- 
lasso_regressor = Lasso()
parametter = {'alpha' : [1,2,3,4,5,6,7,8,9,10]} # for hyperparameter tuning

lasso_cv = GridSearchCV(estimator = lasso_regressor,
                        param_grid = parametter,
                        scoring = 'neg_mean_squared_error',
                        cv = 5)

lasso_cv.fit(x_train,y_train)
print(lasso_cv.best_params_)
print(lasso_cv.best_score_) # best MSE

# lasso_cv_prediction = lasso_cv.predict(x_test)