# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 23:34:05 2020

@author: HP
"""


# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 01:37:59 2020

@author: HP
"""

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import sklearn.metrics as metrics
import math
from sklearn.model_selection import train_test_split

##################################
# Preprocessing of training data #
##################################

# Read in data
train_features = pd.read_csv("train_features.csv")
test_features = pd.read_csv("test_features.csv")
train_labels = pd.read_csv("train_labels.csv")

# extract unique pids
all_pids = train_features["pid"].unique()
all_pids_test = test_features["pid"].unique()

# partition data into rare and common
headers = list(train_features)
rare = []
common = []
tot = train_features.shape[0] + test_features.shape[0] # total entries per column
for head in headers:
    # get number of missing values in this column
    missing = train_features[head].isna().sum() + test_features[head].isna().sum()
    if(missing/tot >= 0.9):
        rare.append(head)
    else: 
        common.append(head)

# impute and compute features for each patient
imp = IterativeImputer()
features = ["pid", "Age"] + rare

common.remove("pid")
common.remove("Age")

for c in common:
    features.append(c + "_mean")
    features.append(c + "_min")
    features.append(c + "_max")
    features.append(c + "_median")
 
    
X_feat = pd.DataFrame(index = all_pids, data = {"pid": all_pids}, columns = features)

skip = False
for pid in all_pids:
    patient_data = train_features[train_features.pid == pid]
    data_update = {'Age': np.min(patient_data["Age"])}
    # can always do this part            
    for head in rare:
        missing = patient_data[head].isna().sum()
        if(missing > 11):
            data_update.update({head: -1}) # no test conducted
        else:
            data_update.update({head: 1}) # test conducted
        patient_data[head] = patient_data[head].fillna(0)
        
    # check whether we can impute this patient's common data (i.e. no empty columns)
    for head in common:
        missing = patient_data[head].isna().sum()
        if(missing > 11):
            # can't impute for now
            skip = True
            break
    
    if(skip):
        skip = False
    else:
        patient_data = pd.DataFrame(data = imp.fit_transform(np.array(patient_data)), columns = headers)
        # calculate some useful statistics
        for head in common:
            dic = {head + "_mean": np.mean(patient_data[head]),
                   head + "_min": np.min(patient_data[head]),
                   head + "_max": np.max(patient_data[head]),
                   head + "_median": np.median(patient_data[head])}
            data_update.update(dic)    
    
    patient_update = pd.DataFrame(index = {pid}, data = data_update, columns = features)
    X_feat.update(patient_update)
    print(pid)  

############################
# Preprocess test_features #
############################
Test_feat = pd.DataFrame(index = all_pids_test, data = {"pid": all_pids_test}, columns = features)

for pid in all_pids_test:
    patient_data = train_features[train_features.pid == pid]
    data_update = {'Age': np.min(patient_data["Age"])}
    # can always do this part            
    for head in rare:
        missing = patient_data[head].isna().sum()
        if(missing > 11):
            data_update.update({head: -1}) # no test conducted
        else:
            data_update.update({head: 1}) # test conducted
        patient_data[head] = patient_data[head].fillna(0)
        
    # check whether we can impute this patient's common data (i.e. no empty columns)
    for head in common:
        missing = patient_data[head].isna().sum()
        if(missing > 11):
            # can't impute for now
            skip = True
            break
    
    if(skip):
        skip = False
    else:
        patient_data = pd.DataFrame(data = imp.fit_transform(np.array(patient_data)), columns = headers)
        # calculate some useful statistics
        for head in common:
            dic = {head + "_mean": np.mean(patient_data[head]),
                   head + "_min": np.min(patient_data[head]),
                   head + "_max": np.max(patient_data[head]),
                   head + "_median": np.median(patient_data[head])}
            data_update.update(dic)    
    
    patient_update = pd.DataFrame(index = {pid}, data = data_update, columns = features)
    Test_feat.update(patient_update)
    print(pid)
 
#######################################################    
# Standardize and set missing values to overall means #
#######################################################
scaler = StandardScaler()
frames = [X_feat, Test_feat]
all_data = pd.concat(frames)
scaled = scaler.fit_transform(all_data.drop("pid", axis = 1))
scaled = pd.DataFrame(index = all_data.pid, data = scaled, columns = features)

X_feat.update(scaled)
X_feat = X_feat.fillna(0)

pd.DataFrame(X_feat).to_csv("X_feat_standardized.csv", index=False, header=True)  

Test_feat.update(scaled)
Test_feat = Test_feat.fillna(0)

pd.DataFrame(Test_feat).to_csv("Test_feat_standardized.csv", index=False, header=True) 

#################################
# Model selection via KFold CV  #
#################################
tests_to_predict = ["LABEL_BaseExcess", "LABEL_Fibrinogen", "LABEL_AST", "LABEL_Alkalinephos", "LABEL_Bilirubin_total", "LABEL_Lactate", "LABEL_TroponinI", "LABEL_SaO2", "LABEL_Bilirubin_direct", "LABEL_EtCO2"]
sepsis = ["LABEL_Sepsis"]
vital_signs_to_predict = ["LABEL_RRate", "LABEL_ABPm", "LABEL_SpO2", "LABEL_Heartrate"]

n_fold = 5
kf = KFold(n_splits = n_fold) 

model_col = []
for m in tests_to_predict + sepsis + vital_signs_to_predict:
    model_col.append(m + "_kernel")
    model_col.append(m + "_xhi")
    
model = pd.DataFrame(index = {0}, columns = model_col)
#model = pd.read_csv("modelselection.csv")

score_col = tests_to_predict + sepsis + vital_signs_to_predict
rocauc = pd.DataFrame(index = {0}, columns = score_col)

# Subtask 1 and 2
for test in tests_to_predict + sepsis:
    if(rocauc[test].isna().sum() == 0): continue
    print(test, "started")
    exp_min = -3
    exp_max = 2
    xhi = 10**exp_min
    j = 0
    avg_roc_auc_score_rbf = []
    avg_roc_auc_score_poly = []
    
    while xhi <= 10**exp_max:
        clf_rbf = SVC(C = xhi, kernel = 'rbf', class_weight = 'balanced', cache_size = 1000, probability = True)
        clf_poly = SVC(C = xhi, kernel = 'poly', class_weight = 'balanced', cache_size = 1000, probability = True)
        avg_roc_auc_score_rbf.append(0)
        avg_roc_auc_score_poly.append(0)
        
        fold = 0
        for train_pid, val_pid in kf.split(all_pids):
            fold += 1
            print("Fold = ", fold)
            X_train, X_val = X_feat[X_feat["pid"].isin(train_pid)], X_feat[X_feat["pid"].isin(val_pid)]
            y_train, y_val = train_labels[train_labels["pid"].isin(train_pid)].get(test), train_labels[train_labels["pid"].isin(val_pid)].get(test)
            X_train = X_train.drop("pid", axis = 1)
            X_val = X_val.drop("pid", axis = 1)
            
            # RBF
            clf_rbf.fit(X_train, y_train)
            predict = clf_rbf.predict_proba(X_val)
            avg_roc_auc_score_rbf[j] +=  metrics.roc_auc_score(y_val, predict[:,1])           
            
            print("RBF done")
            
            # Poly 
            clf_poly.fit(X_train, y_train)
            predict = clf_poly.predict_proba(X_val)
            avg_roc_auc_score_poly[j] +=  metrics.roc_auc_score(y_val, predict[:,1])
            
            print("Poly done")
         
        avg_roc_auc_score_rbf[j] /= n_fold
        avg_roc_auc_score_poly[j] /= n_fold
        print("xhi = ", xhi)  
        print("avg_roc_auc_score_rbf = ", avg_roc_auc_score_rbf)
        print("avg_roc_auc_score_poly = ", avg_roc_auc_score_poly)
    
        xhi *= 10
        j += 1
        
    xhi_rbf_best = avg_roc_auc_score_rbf.index(np.max(avg_roc_auc_score_rbf))
    max_auc_rbf = np.max(avg_roc_auc_score_rbf)
    xhi_poly_best = avg_roc_auc_score_poly.index(np.max(avg_roc_auc_score_poly))
    max_auc_poly = np.max(avg_roc_auc_score_poly)
    print("optimal xhi of rbf: ", max_auc_rbf, " at ", xhi_rbf_best)
    print("optimal xhi of poly: ", max_auc_poly, " at ", xhi_poly_best)
    
    if(max_auc_rbf > max_auc_poly):
        kernel = "rbf"
        xhi_best = 10**(exp_min + xhi_rbf_best)
    else:
        kernel = "poly"
        xhi_best = 10**(exp_min + xhi_poly_best)
    
    # save best strategy in df model
    m = pd.DataFrame(data = [[kernel, xhi_best]], columns = [test + "_kernel", test + "_xhi"])
    model.update(m)
    
    # save best ROC AUC value in df rocauc (for manual review)
    r = pd.DataFrame(index = {0}, data = {test: np.max([max_auc_rbf, max_auc_poly])})
    rocauc.update(r)
  
    pd.DataFrame(model).to_csv("modelselection.csv", index=False, header=True)
    pd.DataFrame(rocauc).to_csv("ROCAUC.csv", index=False, header=True)  
    print(test, " finished")
    
# Subtask 3
for sign in vital_signs_to_predict:
    if(rocauc[sign].isna().sum() == 0): continue
    print(sign, "started")
    exp_min = -3
    exp_max = 2
    xhi = 10**exp_min
    j = 0
    avg_score_rbf = []
    avg_score_poly = []
    while xhi <= 10**exp_max:
        reg_rbf = SVR(C = xhi, kernel = 'rbf', cache_size = 1000)
        reg_poly = SVR(C = xhi, kernel = 'poly', cache_size = 1000)
        avg_score_rbf.append(0)
        avg_score_poly.append(0)
        
        fold = 0
        for train_pid, val_pid in kf.split(all_pids):
            fold += 1
            print("Fold = ", fold)
            X_train, X_val = X_feat[X_feat["pid"].isin(train_pid)], X_feat[X_feat["pid"].isin(val_pid)]
            y_train, y_val = train_labels[train_labels["pid"].isin(train_pid)].get(sign), train_labels[train_labels["pid"].isin(val_pid)].get(sign)
            X_train = X_train.drop("pid", axis = 1)
            X_val = X_val.drop("pid", axis = 1)
            
            # RBF
            reg_rbf.fit(X_train, y_train)
            predict = reg_rbf.predict(X_val)
            avg_score_rbf[j] +=  metrics.r2_score(y_val, predict)           
            
            print("RBF done")
            
            # Poly 
            reg_poly.fit(X_train, y_train)
            predict = reg_poly.predict(X_val)
            avg_score_poly[j] +=  metrics.r2_score(y_val, predict)  
            
            print("Poly done")
         
        avg_score_rbf[j] /= n_fold
        avg_score_poly[j] /= n_fold
        print("xhi = ", xhi)  
        print("avg_score_rbf = ", avg_score_rbf)
        print("avg_score_poly = ", avg_score_poly)
    
        xhi *= 10
        j += 1
        
    xhi_rbf_best = avg_score_rbf.index(np.max(avg_score_rbf))
    max_rbf = np.max(avg_score_rbf)
    xhi_poly_best = avg_score_poly.index(np.max(avg_score_poly))
    max_poly = np.max(avg_score_poly)
    print("optimal xhi of rbf: ", max_rbf, " at ", xhi_rbf_best)
    print("optimal xhi of poly: ", max_poly, " at ", xhi_poly_best)
    
    
    if(max_rbf > max_poly):
        kernel = "rbf"
        xhi_best = 10**(exp_min + xhi_rbf_best)
    else:
        kernel = "poly"
        xhi_best = 10**(exp_min + xhi_poly_best)
    
    # save best strategy in df model
    m = pd.DataFrame(data = [[kernel, xhi_best]], columns = [sign + "_kernel", sign + "_xhi"])
    model.update(m)
    
    # save best score value in df rocauc (for manual review)
    r = pd.DataFrame(index = {0}, data = {sign: np.max([max_rbf, max_poly])})
    rocauc.update(r)
  
    pd.DataFrame(model).to_csv("modelselection.csv", index=False, header=True)
    pd.DataFrame(rocauc).to_csv("ROCAUC.csv", index=False, header=True)  
    print(sign, " finished")    
    
##################################
# Model training and predictions #
##################################
predictions = pd.DataFrame(index = all_pids_test, data = {"pid": all_pids_test}, columns = list(train_labels))
for test in tests_to_predict + sepsis:
    if(predictions[test].isna().sum() == 0): continue
    print(test, "started")
    kernel = list(model[test + "_kernel"])[0]
    xhi = list(model[test + "_xhi"])[0]
    X = X_feat.drop("pid", axis = 1)
    Y = train_labels.get(test)
    clf = SVC(C = xhi, kernel = kernel, class_weight = 'balanced', cache_size = 500, probability = True)
    clf.fit(X, Y)
    predict = pd.DataFrame(index = all_pids_test, data = np.round(clf.predict_proba(Test_feat.drop("pid", axis = 1)), 3)[:,1], columns = [test])
    predictions.update(predict)
    print(test, "finished")

for sign in vital_signs_to_predict:
    if(predictions[sign].isna().sum() == 0): continue
    print(sign, "started")
    kernel = list(model[test + "_kernel"])[0]
    xhi = list(model[test + "_xhi"])[0]
    reg = SVR(C = xhi, kernel = kernel, cache_size = 500)
    reg.fit(X_feat.drop("pid", axis = 1), train_labels[sign])
    predict = pd.DataFrame(index = all_pids_test, data = np.round(reg.predict(Test_feat.drop("pid", axis = 1)), 3), columns = [sign])
    predictions.update(predict)
    print(sign, "finished")
    
# save predictions df   
predictions.to_csv("submit.csv", index=False, header=True)         

"""
clf = SVC(C = xhi_best, kernel = kernel, class_weight = 'balanced')
clf.fit(X_feat, train_labels["LABEL_BaseExcess"])
alpha = clf.decision_function(X_feat)

###########
# Predict #
###########

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

gamma2 = (1/(np.size(features)-1))**2
predicted = []
for t_pid in Test_feat["pid"]:
    y = y_val[y_val.pid == t_pid].get("LABEL_BaseExcess")
    t = np.array(Test_feat[Test_feat.pid == t_pid].drop("pid", axis = 1))
    #compute kernels
    k = []
    for x_pid in X_feat["pid"]:
        x = np.array(X_feat[X_feat.pid == x_pid].drop("pid", axis = 1))
        euc_dist = np.linalg.norm(x - t)
        rbf = math.exp(-euc_dist**2 * gamma2)
        k.append(y*rbf)
    
    predicted.append(sigmoid(np.dot(alpha, k)))

##############
# Testground #
##############
x_train, x_val, y_train, y_val = train_test_split(X_feat, train_labels[["pid", "LABEL_BaseExcess"]], train_size= 0.8)
clf = SVC(C = xhi_best, kernel = kernel, class_weight = 'balanced', cache_size = 1000)
clf.fit(x_train.drop("pid", axis = 1), y_train.drop("pid", axis = 1))
alpha = clf.decision_function(x_train.drop("pid", axis = 1))

print(np.mean(metrics.roc_auc_score(y_val.drop("pid", axis = 1), predicted)))
"""