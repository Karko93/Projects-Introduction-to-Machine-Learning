#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


# In[2]:


## Reading data from csv-file and compart into input and output
train = pd.read_csv("train.csv")
X = np.array(train.drop(columns = ["Id", "y"]))
y = np.array(train.get("y"))



## data preprocessing

#Ridge Regression
cv = KFold(n_splits=10,shuffle=True)
cv.get_n_splits(X)
alphas = np.array([0.01])
RMSE_avg = []
for alph in alphas:
    RMSE = []
    
    for train_index, test_index in cv.split(X):
        #print('\n', test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = Ridge(alpha= alph, normalize= True, fit_intercept=False)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        RMSE.append(sk.metrics.mean_squared_error(y_test, y_pred))
    
    RMSE_avg.append(np.mean(RMSE))
    
   #print(np.mean(RMSE), '\n', alph, '\n', RMSE)
    
####################
# Error evaluation #
####################


#print(X_train.shape,X_test.shape, '\n', y_train.shape,y_test.shape)


#print(sk.metrics.mean_squared_error(y_test, y_pred))
#print(RMSE)
result = pd.concat([pd.DataFrame(RMSE_avg)])
#print(result)
pd.DataFrame(result).to_csv("submit.csv", index=False, header=False, decimal='.', sep=' ', float_format='%.1f')



