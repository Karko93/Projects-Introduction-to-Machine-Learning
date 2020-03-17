# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 16:52:51 2020

@author: HP
"""

#####################
# include libraries #
#####################
import numpy as np
import csv
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold

################
# read in data #   
################
n_feat = 5 # number of features
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
              
##########################
# Feature Transformation #
##########################
x_bar = []

for j in range(idx+1):
    x_bar.append([])
    # linear
    for i in np.arange(0, 5):
        a = x[j][i]
        x_bar[j].append(a)
    
    # quadratic
    for i in np.arange(0, 5):
        a = x[j][i]
        x_bar[j].append(a**2)
        
    # exponential
    for i in np.arange(0, 5):
        a = x[j][i]
        x_bar[j].append(np.exp(a))
    
    # cosine
    for i in np.arange(0, 5):
        a = x[j][i]
        x_bar[j].append(np.cos(a))

    # constant
    x_bar[j].append(1)
      
###################
# Model Selection #
###################
x_bar = np.array(x_bar)
y = np.array(y)
    
n_fold = 10
kf = KFold(n_splits = n_fold)          

exp_min = -3
exp_max = 6
xhi = 10**exp_min
j = 0
avg_rmse_ridge = []
avg_rmse_lasso = []

while xhi < 10**exp_max:
    avg_rmse_ridge.append(0)
    avg_rmse_lasso.append(0)
    for train_index, test_index in kf.split(x_bar):
        X_train, X_val = x_bar[train_index], x_bar[test_index]
        y_train, y_val = y[train_index], y[test_index]
                        
        # Ridge
        clf = Ridge(alpha=xhi, fit_intercept=False, normalize=False, max_iter = 2**31-1, tol = 1e-8)
        clf.fit(X_train, y_train)
        w = clf.coef_
        avg_rmse_ridge[j] += np.sqrt(mean_squared_error(y_val, X_val.dot(w)))
        
        # Lasso
        lasso = Lasso(alpha=xhi, fit_intercept = False, max_iter = 2**31-1, tol = 1e-8)
        lasso.fit(X_train, y_train)
        w = lasso.coef_
        avg_rmse_lasso[j] += np.sqrt(mean_squared_error(y_val, X_val.dot(w))) ## bug in squared
               
    avg_rmse_ridge[j] /= n_fold    
    avg_rmse_lasso[j] /= n_fold
    xhi *= 10
    j += 1

#####################
# Linear Regression #
##################### 
model = min(avg_rmse_ridge) < min(avg_rmse_lasso)    

# Ridge
if(model):    
    exp_best = avg_rmse_ridge.index(min(avg_rmse_ridge)) + exp_min    
    xhi_best = 10**exp_best
    clf = Ridge(alpha=xhi_best, fit_intercept=False, normalize=False, max_iter = 2**31-1, tol = 1e-8)
    clf.fit(x_bar, y)
    w = clf.coef_

# Lasso
else:
    exp_best = avg_rmse_lasso.index(min(avg_rmse_lasso)) + exp_min    
    xhi_best = 10**exp_best
    lasso = Lasso(alpha=xhi_best, fit_intercept = False, max_iter = 2**31-1, tol = 1e-8)
    lasso.fit(x_bar, y)
    w = lasso.coef_
    
#########################
# Saving weights in csv #
#########################
result = pd.concat([pd.DataFrame(w)])
pd.DataFrame(result).to_csv("submit.csv", index=False, header=False,decimal='.', sep=' ')

print(w)