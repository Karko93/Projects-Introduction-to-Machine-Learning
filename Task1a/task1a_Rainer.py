# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 22:48:46 2020

@author: HP
"""
#####################
# include libraries #
#####################
import numpy as np
import csv
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import pandas as pd

################
# read in data #   
################
n_feat = 13 # number of features
n_fold = 10 # number of folds
idx = 0 # index of current data point
y = []
x = []

with open('train.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    next(csvReader) # skip first row
    for row in csvReader:
        idx = int(row[0])
        y.append(float(row[1]))
        x.append([]) # new data point
        for i in range(n_feat):
            x[idx].append(float(row[i+2]))
                                
####################
# Ridge Regression # 
####################
            
x = np.array(x)
y = np.array(y)
    
kf = KFold(n_splits = n_fold)          

exp_min = -2
exp_max = 3
xhi = 10**exp_min
j = 0
avg_rmse = []

while xhi < 10**exp_max:
    avg_rmse.append(0)
    for train_index, test_index in kf.split(x):
        X_train, X_val = x[train_index], x[test_index]
        y_train, y_val = y[train_index], y[test_index]
                        
        # Ridge
        clf = Ridge(alpha=xhi, fit_intercept=False, max_iter = 2**31-1, tol = 1e-8)
        clf.fit(X_train, y_train)
        y_hat = clf.predict(X_val)
        avg_rmse[j] += mean_squared_error(y_val, y_hat, squared = False)/n_fold
        
    #avg_rmse[j] /= n_fold    
    
    xhi *= 10
    j += 1
    
#########################
# Saving results in csv #
#########################
result = pd.concat([pd.DataFrame(avg_rmse)])
pd.DataFrame(result).to_csv("submit.csv", index=False, header=False, decimal='.', sep=' ')   

print(result) 