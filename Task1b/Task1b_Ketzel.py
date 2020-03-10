#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# In[8]:


#Reading data
train = pd.read_csv("train.csv")
X = np.array(train.drop(columns = ["Id", "y"]))
y = np.array(train.get("y"))
print(X.shape)


# In[9]:


## Data Preprocessing (Adding new features)
x_extend = np.append(X, X**2, axis=1)
x_extend = np.append(x_extend, np.exp(X), axis=1)
x_extend = np.append(x_extend, np.cos(X), axis=1)
x_extend = np.hstack((x_extend, np.ones((X.shape[0], 1), dtype=float)))
print(x_extend.shape)


# In[14]:


## Linear Regression and Getting Coefficients of Models
reg = LinearRegression(fit_intercept= False, normalize=True)
reg.fit(x_extend,y)
Coeffs = reg.coef_
print(Coeffs.shape)


# In[15]:


## Save Coefficient in a csv
result = pd.concat([pd.DataFrame(Coeffs)])
np.set_printoptions(precision=3)
print(Coeffs)
pd.DataFrame(result).to_csv("submit.csv", index=False, header=False, decimal='.', sep=' ', float_format='%.0f')

Coeffs_Pre = np.array(pd.read_csv("submit_pre.csv", header = None))

print('\n', Coeffs_Pre.T - [Coeffs])

# In[ ]:




