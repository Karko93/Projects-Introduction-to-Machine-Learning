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
reg = LinearRegression(fit_intercept= False, normalize= False)
reg.fit(x_extend,y)
coeffs = reg.coef_
print(coeffs.shape)


# In[15]:


## Save Coefficient in a csv
result = pd.concat([pd.DataFrame(coeffs)])
print(result)
pd.DataFrame(result).to_csv("submit.csv", index=False, header=False, decimal='.', sep=' ', float_format='%.0f')


# In[ ]:




