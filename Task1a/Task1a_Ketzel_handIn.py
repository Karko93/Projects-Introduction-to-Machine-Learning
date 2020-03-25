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
##Data Preprocessing (t.b.a.)
scaler = StandardScaler(with_mean= False, with_std= True)
scaler.fit(X)
#X = scaler.transform(X)


#Ridge Regression with K-Fold
cv = KFold(n_splits=10, shuffle = False)
cv.get_n_splits(X)
alphas = np.array([0.01, 0.1 , 1 , 10 , 100])
RMSE_avg = []

for alph in alphas:
    RMSE = []
    clf = Ridge(alpha= alph, normalize= True, fit_intercept=False,max_iter = 2**31-1, tol = 1e-8)
    for train_index, test_index in cv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        RMSE.append(sk.metrics.mean_squared_error(y_test, y_pred, squared = False))
    
    RMSE_avg.append(np.mean(RMSE))
    print(np.mean(RMSE), '\n', alph, '\n', RMSE)


result = pd.concat([pd.DataFrame(RMSE_avg)])
print(result)
pd.DataFrame(result).to_csv("submit.csv", index=False, header=False, decimal='.', sep=' ') #float_format='%.1f')


#0  4.980966
#1  4.980783
#2  4.980050
#3  4.985353
#4  5.166496


