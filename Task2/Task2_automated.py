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
tot = train_features.shape[0] # total entries per column
for head in headers:
    # get number of missing values in this column
    missing = train_features[head].isna().sum()
    if(missing/tot >= 0.9):
        rare.append(head)
    else: 
        common.append(head)

# impute and compute features for each patient
simp = SimpleImputer()
features = ["pid", "Age"] + rare

common.remove("pid")
common.remove("Age")

for c in common:
    features.append(c + "_mean")
    features.append(c + "_min")
    features.append(c + "_max")
    features.append(c + "_median")
 
    
X_feat = pd.DataFrame(index = list(range(all_pids.shape[0])), columns = features)
X_feat.update(pd.DataFrame(all_pids, columns = ["pid"]))

for pid in all_pids:
    # already done:
    break

    patient_data = train_features[train_features.pid == pid]
    for head in common:
        missing = patient_data[head].isna().sum()
        if(missing > 11):
            # can't impute for now
            continue
        else:
            patient_data.loc[:, head] = simp.fit_transform(np.array(patient_data[head]).reshape(-1,1))
            # calculate some useful statistics
            X_feat.loc[X_feat.pid == pid, head + "_mean"] = np.mean(patient_data[head])
            X_feat.loc[X_feat.pid == pid, head + "_min"] = np.min(patient_data[head])
            X_feat.loc[X_feat.pid == pid, head + "_max"] = np.max(patient_data[head])
            X_feat.loc[X_feat.pid == pid, head + "_median"] = np.median(patient_data[head])          
            
    
    for head in rare:
        missing = patient_data[head].isna().sum()
        if(missing > 11):
            X_feat.loc[X_feat.pid == pid, head] = -1 # no test conducted
        else:
            X_feat.loc[X_feat.pid == pid, head] = 1 # test conducted
    
    
    X_feat.loc[X_feat.pid == pid, "Age"] = np.min(patient_data["Age"])
    print(pid)
    
# Standardize and set missing values to overall means
X_feat = pd.read_csv("X_feat.csv")
scaler = StandardScaler()
X_feat = scaler.fit_transform(X_feat)
X_feat = pd.DataFrame(X_feat, columns = features)
X_feat.update(pd.DataFrame(all_pids, columns = ["pid"]))
X_feat = X_feat.fillna(0)

# pd.DataFrame(X_feat).to_csv("X_feat_standardized.csv", index=False, header=True)    

############################
# Preprocess test_features #
############################
Test_feat = pd.DataFrame(index = list(range(all_pids_test.shape[0])), columns = features)
Test_feat.update(pd.DataFrame(all_pids_test, columns = ["pid"]))

for pid in all_pids_test:
    break
    patient_data = test_features[test_features.pid == pid]
    for head in common:
        missing = patient_data[head].isna().sum()
        if(missing > 11):
            # can't impute for now
            continue
        else:
            patient_data.loc[:, head] = simp.fit_transform(np.array(patient_data[head]).reshape(-1,1))
            # calculate some useful statistics
            Test_feat.loc[Test_feat.pid == pid, head + "_mean"] = np.mean(patient_data[head])
            Test_feat.loc[Test_feat.pid == pid, head + "_min"] = np.min(patient_data[head])
            Test_feat.loc[Test_feat.pid == pid, head + "_max"] = np.max(patient_data[head])
            Test_feat.loc[Test_feat.pid == pid, head + "_median"] = np.median(patient_data[head])          
            
    
    for head in rare:
        missing = patient_data[head].isna().sum()
        if(missing > 11):
            Test_feat.loc[Test_feat.pid == pid, head] = -1 # no test conducted
        else:
            Test_feat.loc[Test_feat.pid == pid, head] = 1 # test conducted
    
    
    Test_feat.loc[Test_feat.pid == pid, "Age"] = np.min(patient_data["Age"])
    print(pid)
    
# Standardize and set missing values to overall means
Test_feat = pd.read_csv("Test_feat.csv")
Test_feat = scaler.fit_transform(Test_feat)
Test_feat = pd.DataFrame(Test_feat, columns = features)
Test_feat.update(pd.DataFrame(all_pids_test, columns = ["pid"]))
Test_feat = Test_feat.fillna(0)

# pd.DataFrame(Test_feat).to_csv("Test_feat_standardized.csv", index=False, header=True) 

#################################
# Model selection via KFold CV  #
#################################
tests_to_predict = ["LABEL_BaseExcess", "LABEL_Fibrinogen", "LABEL_AST", "LABEL_Alkalinephos", "LABEL_Bilirubin_total", "LABEL_Lactate", "LABEL_TroponinI", "LABEL_SaO2", "LABEL_Bilirubin_direct", "LABEL_EtCO2"]
sepsis = ["LABEL_Sepsis"]
vital_signs_to_predict = ["LABEL_RRate", "LABEL_ABPm", "LABEL_SpO2", "LABEL_Heartrate"]

n_fold = 10
kf = KFold(n_splits = n_fold) 

model_col = []
for m in tests_to_predict + sepsis:
    model_col.append(m + "_kernel")
    model_col.append(m + "_xhi")
    
model = pd.DataFrame(index = {0}, columns = model_col)

rocauc_col = tests_to_predict + sepsis + vital_signs_to_predict
rocauc = pd.DataFrame(index = {0}, columns = rocauc_col)

for test in tests_to_predict + sepsis:
    print(test, " started")
    exp_min = -4
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
        
        for train_pid, val_pid in kf.split(all_pids):        
            X_train, X_val = X_feat[X_feat["pid"].isin(train_pid)], X_feat[X_feat["pid"].isin(val_pid)]
            y_train, y_val = train_labels[train_labels["pid"].isin(train_pid)].get(test), train_labels[train_labels["pid"].isin(val_pid)].get(test)
            X_train = scaler.fit_transform(X_train.drop("pid", axis = 1))
            X_val = scaler.fit_transform(X_val.drop("pid", axis = 1))
            
            # RBF
            clf_rbf.fit(X_train, y_train)
            predict = clf_rbf.predict_proba(X_val)
            avg_roc_auc_score_rbf[j] +=  metrics.roc_auc_score(y_val, predict[:,1])           
            
            # Poly 
            clf_poly.fit(X_train, y_train)
            predict = clf_poly.predict_proba(X_val)
            avg_roc_auc_score_poly[j] +=  metrics.roc_auc_score(y_val, predict[:,1])
         
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
##################
# Model training #
##################
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