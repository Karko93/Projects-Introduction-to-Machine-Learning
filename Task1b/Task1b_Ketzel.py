#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# In[8]:


#Reading data
train = pd.read_csv("train.csv")
X = np.array(train.drop(columns = ["Id", "y"]))
y = np.array(train.get("y"))
print(X.shape)

#X = preprocessing.normalize(X)



# In[9]:


## Data Preprocessing (Adding new features)
x_extend = np.append(X, X**2, axis=1)
x_extend = np.append(x_extend, np.exp(X), axis=1)
x_extend = np.append(x_extend, np.cos(X), axis=1)
x_extend = np.hstack((x_extend, np.ones((X.shape[0], 1), dtype= float)))
print(x_extend.shape)

x_train, x_test, y_train, y_test = train_test_split(x_extend, y, test_size=0.1, shuffle = True)

# In[14]:


## Linear Regression and Getting Coefficients of Models
#reg = lm.LinearRegression(fit_intercept= False, normalize= False)
reg = lm.RidgeCV(fit_intercept=False, alphas = [0.1,1,10,100])
reg.fit(x_train,y_train)
y_hat = reg.predict(x_extend)
Coeffs = reg.coef_
print(Coeffs.shape)


# In[15]:


## Save Coefficient in a csv
result = pd.concat([pd.DataFrame(Coeffs)])
np.set_printoptions(precision=3)
pd.DataFrame(result).to_csv("submit.csv", index=False, header=False, decimal='.', sep=' ' ,float_format='%.0f')
Coeffs_Pre = np.array(pd.read_csv("submit_pre.csv", header = None))

print('\n', Coeffs_Pre.T,'\n', Coeffs)
print('\n','RMSE = ',np.sqrt(mean_squared_error(y,y_hat)))

# In[ ]:




