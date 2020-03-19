import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import datasets
from sklearn.metrics import mean_squared_error 
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import time


#Print messages
printMessages = True
timeOn = False
pm = printMessages
plotcompl = True
start_time = time.time()
def PrintFunc(text,doesPrint=True):
    if doesPrint:
        print(text)
        if timeOn:
            print("Current time:" ,(time.time() - start_time))
if printMessages:
    print('Print messages activated')


#Functions
#####Used for meansquare error, use
def rmsep(x):
    rmse= np.sqrt(np.absolute(x))
    return np.mean(rmse)
def rmse(x):
    return np.sqrt(np.absolute(x))
def annot_min(x,y, ax=None):
    xmin = x[np.argmin(y)]
    ymin = y.min()
    text= "x={:.3f}, y={:.3f}".format(xmin, ymin)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmin, ymin), xytext=(0.94,0.96), **kw)
    
def plotEst(param,evaluater,methode):
    scores = rmse(evaluater.cv_results_['mean_test_score'])
    scores_std = evaluater.cv_results_['std_test_score']

    plt.figure().set_size_inches(8, 6)
    plt.semilogx(param, scores)
    # plot error lines showing +/- std. errors of the scores
    std_error = scores_std / np.sqrt(cv)
    
    plt.semilogx(param, scores + std_error, 'b--')
    plt.semilogx(param, scores - std_error, 'b--')
    
    # alpha=0.2 controls the translucency of the fill color
    plt.fill_between(param, scores + std_error, scores - std_error, alpha=0.2)
    plt.title(methode)
    plt.ylabel('CV score +/- std error')
    plt.xlabel('alpha')
    plt.axhline(np.max(scores), linestyle='--', color='.5')
    plt.xlim([param[0], param[-1]])
    min_ind = np.where(scores == scores.min())
    print('Method: ' + methode + '\n Minium:' + str(scores[min_ind]) + '\n alpha: ' + str(param[min_ind]))
    annot_min(scores,param)
    plt.show()

    ##




#Initialization
subNumber = '1'
filenameSub = "Task1b_" + subNumber + ".csv"

#Import data
train = pd.read_csv("train.csv",float_precision='high') 
PrintFunc("Data imported",printMessages)

#Convert Data
#-  To NP
trainNp = train.values

#-  To X-y
x_trainnp = trainNp[:,2:]
y_trainnp = trainNp[:,1]


#Preprocessing data resulted in worse coefficient neglected therefor
##Removing outlier with quantile range 20-80
#transformer = RobustScaler(copy=True, quantile_range =(20.0,80.0),with_centering=False).fit(x_trainnp)
#transformer
#x_train_prepr = transformer.transform(x_trainnp)





#Feature transformation
#x_quad = x_trainnp**2
x_quad = np.square(x_trainnp)#higher precision
x_exp = np.exp(x_trainnp)
x_cos = np.cos(x_trainnp)
x_const = np.ones((1000,1))


#Put all together in one matrix y_hut = omega_1_Phi_1(x) + ... +  omega_21_Phi_21(x)
x_traintrf = np.concatenate((x_trainnp,x_quad,x_exp,x_cos,x_const), axis=1)









#Cross validation: 
PrintFunc("--Cross validation--", pm)  
#Step 1: First split data in training data and test data (see sklearn chapter 3.1)
#Step 2: use training data and cross validations to predict best coefficients
#step 3: use test set for final evaluation

#step1
PrintFunc("Step 1: Splitting data in test/train data", pm)  
X_train, X_test, y_train, y_test = train_test_split(x_traintrf, y_trainnp, test_size=0.3, random_state=1)


#step2
#Cross Validation
cv = 10
PrintFunc("Step 2: K-Fold for all the models", pm)  

#No Preprocesing - linearRegression
regr = linear_model.LinearRegression(fit_intercept=False)
scores=cross_val_score(regr, X_train, y_train, cv=cv,scoring='neg_mean_squared_error')
#PrintFunc("Score of the different cross_validations: " + str(np.sqrt(np.absolute(scores))),pm) 
PrintFunc("Expected score for Linear Regression: " +str(rmsep(scores)), pm)  
regr.fit(X_train,y_train)


#Preprocessing included in LinearRegressionmodel - does not make any change in the score
#regr_prep = linear_model.LinearRegression(fit_intercept=True,normalize=True)
#scores_prep=cross_val_score(regr_prep, X_train, y_train, cv=cv,scoring='neg_mean_squared_error')
#PrintFunc("Score of the different cross_validations using preprocessing: " + str(np.sqrt(np.absolute(scores_prep))),pm)   
#PrintFunc("This should be the expected score using preprocessing: " +str(rmsep(scores_prep)), pm) 
#regr_prep.fit(X_train,y_train)
#PrintFunc("Score using fit: " +str(regr_prep.score(X_test,y_test)), pm)  

##Ridge Regression
lambdas = [0.01,0.1,1,10,100,1000,10000,100000]
averageRMSE = np.zeros(len(lambdas))
for i in range(len(lambdas)):
    lmRR = linear_model.Ridge(alpha=lambdas[i],fit_intercept=False)
    scores = cross_val_score(lmRR, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
    averageRMSE[i] = rmsep(scores)
   # PrintFunc("Expected score for Ridge Regression using lambda "+ str(lambdas[i]) +" :" +str(rmsep(scores)), pm)

##Ridge Regression CV
rrcv = RidgeCV(cv=cv,alphas=np.linspace(start=250, stop=500, num=50),fit_intercept=False).fit(X_train, y_train)

##Ridge Regression GridSearch
rr = linear_model.Ridge(max_iter=2000,tol=0.001,fit_intercept = False, random_state=0)
lambdas = np.logspace(-2, 4, 200)
param_grid = [{'alpha': lambdas}]
gs_rr = GridSearchCV(rr, param_grid, cv=cv, refit=False, scoring='neg_mean_squared_error')
gs_rr.fit(X_train, y_train)


#Stochastik Gradient Descent Regressor
sgdr = linear_model.SGDRegressor(fit_intercept=False,alpha=0.05,max_iter=100,tol=1e-4)
scores=cross_val_score(sgdr, X_train, y_train, cv=cv,scoring='neg_mean_squared_error')
#PrintFunc("Score of the different cross_validations: " + str(np.sqrt(np.absolute(scores))),pm) 
PrintFunc("Expected score for Stochastik gradient descent regressor: " +str(rmsep(scores)), pm)  
sgdr.fit(X_train,y_train)

#Lasso CV
lasCV = LassoCV(cv=cv, random_state=0,fit_intercept=False).fit(X_train, y_train)
 
#Lasso GridSearch
lasso = Lasso(max_iter=3000,tol=0.001,fit_intercept = False, random_state=0)
alphas = np.logspace(-3, 2, 100)
param_grid = [{'alpha': alphas}]
gs_las = GridSearchCV(lasso, param_grid, cv=cv, refit=True, scoring='neg_mean_squared_error')
gs_las.fit(X_train, y_train)






#Step3 Evaluating with test data
PrintFunc("Step 3: Evaluating models with test data", pm)  
    #Linear Regression
PrintFunc("RMSE-score Linear Regression for test-data: " +str(np.sqrt(mean_squared_error(y_test, regr.predict(X_test)))), pm) 
    #Ridge Regression
for i in range(len(lambdas)):
    lmRR = linear_model.Ridge(alpha=lambdas[i],fit_intercept=False)
    lmRR.fit(X_train,y_train)
   # PrintFunc("RMSE-score for Ridge Regression for test-data, lambda = " + str(lambdas[i]) + ": "  +str(np.sqrt(mean_squared_error(y_test, lmRR.predict(X_test)))), pm)
    
    #Stochastik Gradient descent regressor
PrintFunc("RMSE-score Stochastik Gradient Descent Regressor for test-data: " +str(np.sqrt(mean_squared_error(y_test, sgdr.predict(X_test)))), pm) 
    #Lasso
PrintFunc("RMSE-score Lasso for test-data using lambda =" + str(lasCV.alpha_) +": " +str(np.sqrt(mean_squared_error(y_test, lasCV.predict(X_test)))), pm) 
PrintFunc("RMSE-score Lasso Random_search for test-data using lambda =" + str(lasCV.alpha_) +": " +str(np.sqrt(mean_squared_error(y_test, lasCV.predict(X_test)))), pm) 

PrintFunc("RMSE-score RidgeCV for test-data using lambda =" + str(rrcv.alpha_) +": " +str(np.sqrt(mean_squared_error(y_test, rrcv.predict(X_test)))), pm) 






### Plotting:
#Using grid Search and Plotting to estimate best parameters

##
methode = 'Ridge Regression with GridSearch, TrainingSet'
evaluater = gs_rr
evaluater.fit(x_traintrf, y_trainnp)
param = lambdas
plotEst(param,evaluater,methode)
###



###
methode = 'Lasso with GridSearch, TrainingSet'
evaluater = gs_las
param = alphas
plotEst(param,evaluater,methode)
###

##Plotting Complete Set
if plotcompl:
    ##
    methode = 'Lasso with GridSearch, CompleteSet'
    evaluater = gs_las
    evaluater.fit(x_traintrf, y_trainnp)
    param = alphas
    plotEst(param,evaluater,methode)
    bestModel = evaluater
    ###
    
    ##
    methode = 'Ridge Regression with GridSearch, CompleteSet'
    evaluater = gs_rr
    param = lambdas
    plotEst(param,evaluater,methode)
    
    ###

    
#SelectBestModel
#bestModel = LassoCV(cv=cv,alphas=np.linspace(start=0.1, stop=1, num=5000),fit_intercept=False)


#PrintFunc("Final score for testfile using alpha " + str(bestModel.best_estimator_.alpha) +" : " +str(rmsep(bestModel.best_score_)), pm) 
bestModel = Lasso(alpha= 0.6, max_iter=3000,tol=0.001,fit_intercept = False, random_state=0)
bestModel.fit(x_traintrf, y_trainnp)
    
bestEstimator = bestModel








#Save in csv file:
dataForFile  = pd.DataFrame(bestModel.coef_)
dataForFile.to_csv(filenameSub,index = False,header = False)
PrintFunc('Saved in file: \n' + str(filenameSub),printMessages)
