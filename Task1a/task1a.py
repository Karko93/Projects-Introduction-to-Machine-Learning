# -*- coding: utf-8 -*-
"""
Spyder Editor

Dies ist eine temporäre Skriptdatei.
"""

#####################
# include libraries #
#####################
import numpy as np
import csv
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt

################
# read in data #   
################
n_feat = 13 # number of features
n_fold = 10 # number of folds
idx = 0 # index of current data point
y = [[] for k in range(n_fold)]
x = [[] for k in range(n_fold)]
id_fold = 0

with open('train.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    next(csvReader) # skip first row
    for row in csvReader:
        idx = int(row[0])
        id_fold = idx % 10
        y[id_fold].append(float(row[1]))
        x[id_fold].append([]) # new data point
        id_dp = idx // 10 # index of data point in fold
        for i in range(n_feat):
            x[id_fold][id_dp].append(float(row[i+2]))
                             
###################################
# Preprocessing / Standardization #
##################################                
#scaler = StandardScaler()
#for k in range(n_fold):
#    scaler.fit(x[k])
#    scaler.transform(x[k])

####################
# Gradient descent #
####################
y_hat = [[[] for k in range(n_fold)] for z in range(5)]
xhi = 0.01
j = 0
while xhi <= 100:
    for k in range(n_fold):
        x_train = []
        y_train = []
        x_val = x[k]
        for i in range(n_fold):
            if i == k:
                continue
            x_train.extend(x[i])
            y_train.extend(y[i])
                
        clf = Ridge(alpha=xhi, fit_intercept=False, normalize=True)
        clf.fit(x_train, y_train)
        y_hat[j][k] = clf.predict(x_val)
    xhi *= 10
    j += 1
    
####################
# Error evaluation #
####################
avg_rmse = []
for i in range(5): # loop over different regularization parameters
    avg_rmse.append(0)
    for k in range(n_fold):
        rmse = sqrt(mean_squared_error(y[k], y_hat[i][k]))
        avg_rmse[i] += rmse
    avg_rmse[i] /= n_fold

