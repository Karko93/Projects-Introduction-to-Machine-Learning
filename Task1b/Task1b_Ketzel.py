import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV



#Reading data
train = pd.read_csv("train.csv")
X = np.array(train.drop(columns = ["Id", "y"]))
y = np.array(train.get("y"))
print(X.shape)

## Data Preprocessing (Adding new features)
x_extend = np.append(X, X**2, axis=1)
x_extend = np.append(x_extend, np.exp(X), axis=1)
x_extend = np.append(x_extend, np.cos(X), axis=1)
x_extend = np.hstack((x_extend, np.ones((X.shape[0], 1), dtype= float)))
print(x_extend.shape)

#x_train, x_val, y_train, y_val = train_test_split(x_extend, y, test_size=0.2, shuffle = True)

x_train = x_extend
y_train = y

### Scoring function
def RMSE_avg(rmse_list):
    return np.mean(rmse_list)

##Crossvalidation
n_fold = 10
kf = KFold(n_splits = n_fold)
rmse_ridge = []
rmse_lasso = []

alphas = np.logspace(-3, 2, 100)

#for alph in alphas:

    
#    for train_index, test_index in kf.split(x_train):
#        X_tr, X_test = x_train[train_index], x_train[test_index]
#        y_tr, y_test = y_train[train_index], y_train[test_index]
#                                    
#        # Ridge
#        clf = Ridge(alpha= alph , fit_intercept=False, normalize=False, max_iter = 2**31-1, tol = 1e-8)
#        clf.fit(X_tr, y_tr)
#        w = clf.coef_
#        rmse_ridge.append(np.sqrt(mean_squared_error(y_test, X_test.dot(w))))
#                    
#        # Lasso
#        lasso = Lasso(alpha= alph , fit_intercept = False, max_iter = 2**31-1, tol = 1e-8)
#        lasso.fit(X_tr, y_tr)
#        w = lasso.coef_
#        rmse_lasso.append(np.sqrt(mean_squared_error(y_test, X_test.dot(w)))) ## bug in squared


clf = LassoCV(alphas=alphas, fit_intercept=False, normalize=False, max_iter = 2**31-1, tol = 1e-8)   
clf.fit(x_train, y_train) 
Coeffs = clf.coef_
#rmse_lasso_avg = RMSE_avg(rmse_lasso)
#rmse_ridge_avg = RMSE_avg(rmse_ridge)

#if rmse_lasso_avg > rmse_ridge_avg:
#    print("Ridge is better")
#    Coeffs = clf.coef_
#else:
#    print("lasso is better")
#    Coeffs = lasso.coef_
    

## Save Coefficient in a csv - file
result = pd.concat([pd.DataFrame(Coeffs)])
np.set_printoptions(precision=3)
pd.DataFrame(result).to_csv("submit.csv", index=False, header=False, decimal='.', sep=' ')
Coeffs_Pre = np.array(pd.read_csv("submit_pre.csv", header = None))

print('\n', Coeffs_Pre.T,'\n', Coeffs)




