# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 22:48:46 2020

@author: HP
"""
#####################
# include libraries #
#####################
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import pandas as pd

################
# read in data #   
################
file = pd.read_csv("train.csv")
x = np.array(file.drop(columns = ["Id", "y"]))
y = np.array(file.get("y"))
                                
####################
# Ridge Regression # 
####################
n_fold = 10 # number of folds
kf = KFold(n_splits = n_fold)          

exp_min = -2
exp_max = 2
xhi = 10**exp_min
avg_rmse = []

while xhi <= 10**exp_max:
    rmse = []
    for train_index, test_index in kf.split(x):
        X_train, X_val = x[train_index], x[test_index]
        y_train, y_val = y[train_index], y[test_index]
                        
        # Ridge
        clf = Ridge(alpha=xhi, fit_intercept=False, tol=1e-8)
        clf.fit(X_train, y_train)
        y_hat = clf.predict(X_val)
        rmse.append(mean_squared_error(y_hat, y_val, squared = False))
        
    avg_rmse.append(np.mean(rmse))
    xhi *= 10
    
#########################
# Saving results in csv #
#########################
result = pd.concat([pd.DataFrame(avg_rmse)])
pd.DataFrame(result).to_csv("submit.csv", index=False, header=False, decimal='.', sep=' ')    