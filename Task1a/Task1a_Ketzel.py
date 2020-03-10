#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


## Reading data from csv-file and compartment into input and output
train = pd.read_csv("train.csv")
X = np.array(train.drop(columns = ["Id", "y"]))
y = np.array(train.get("y"))

##Data Preprocessing

#Ridge Regression with K-Fold
cv = KFold(n_splits=10)
cv.get_n_splits(X)
alphas = np.array([0.01, 0.1 , 1 , 10 , 100])
RMSE_avg = []

for alph in alphas:
    RMSE = []
    clf = Ridge(alpha= alph, normalize= True, fit_intercept=False)
    for train_index, test_index in cv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        RMSE.append(np.sqrt(sk.metrics.mean_squared_error(y_test, y_pred)))
    
    RMSE_avg.append(np.mean(RMSE))
    print(np.mean(RMSE), '\n', alph, '\n', RMSE)


result = pd.concat([pd.DataFrame(RMSE_avg)])
print(result)
pd.DataFrame(result).to_csv("submit.csv", index=False, header=False, decimal='.', sep=' ', float_format='%.1f')



