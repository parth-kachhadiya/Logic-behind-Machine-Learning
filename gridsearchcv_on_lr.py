""" Cross validation through GridsearchCV on Multilinear regression """

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV


data = pd.read_csv("D:\\Machine_Learning\\codes\\Own_model\\data_csv\\Student_Performance.csv")
X = data.iloc[:,:-1]
Y = data.iloc[:,-1]

x_train = data.iloc[:8001,:-1]
y_train = data.iloc[:8001,-1]

x_test = data.iloc[8001:,:-1]
y_test = data.iloc[8001:,-1].to_numpy()

ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[2])],remainder='passthrough')
x_train = np.array(ct.fit_transform(x_train))
x_test = np.array(ct.transform(x_test))

scalar = StandardScaler()
x_train[:,2:] = scalar.fit_transform(x_train[:,2:])
x_test[:,2:] = scalar.transform(x_test[:,2:])

params = {'fit_intercept' : [True, False], 'positive' : [True, False]}

model = LinearRegression()

grid_serarch_cv = GridSearchCV(estimator = model,
                               param_grid = params,
                               cv = 5,
                               verbose = 1,
                               scoring = 'neg_mean_squared_error')

grid_serarch_cv.fit(x_train,y_train)

final_result = {
    'best_parameter' : grid_serarch_cv.best_params_,
    'best_model' : grid_serarch_cv.best_estimator_,
    'best_score' : grid_serarch_cv.best_score_,
    'report_card' : grid_serarch_cv.cv_results_,
    'scoring_method' : grid_serarch_cv.scorer_
}

print(final_result['best_model'])
print(final_result['best_parameter'])
print(final_result['scoring_method'])

# for k, v in final_result['report_card'].items():
#     print(f"{k} : {v}")
