#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# In[2]:


## Reading data from csv-file and compart into input and output
train = pd.read_csv("train.csv")
X = np.array(train.drop(columns = ["Id", "y"]))
y = np.array(train.get("y"))
#y_train = np.array(y_train)
print(X.shape)


# In[3]:


a = 12
fig, ax =plt.subplots(X.shape[1],3)
ax=fig.add_axes([0,0,1,1])
ax.scatter(X[:,a], y , color='r')
ax.set_xlabel(f'x{a}_Value')
ax.set_ylabel('y_Value')
ax.set_title('scatter plot')


# In[4]:


## data preprocessing
scaler = StandardScaler()
X = X[:,(a-1):a]
scaler.fit(X)
#print(x)


# In[5]:


## k-fold

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=True)

#for train_index, test_index in cv.split(x):
    #print("Train Index: ", train_index, "\n")
    #print("Test Index: ", test_index)


# In[6]:


#Ridge Regression
cv = KFold(n_splits=10)
cv.get_n_splits(X)
alphas = np.array([0.01, 0.1 , 1 , 10 , 100])
RMSE_avg = []

for alph in alphas:
    RMSE = []
    clf = Ridge(alpha= alph, normalize= True, solver = 'auto')
    for train_index, test_index in cv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        RMSE.append(sk.metrics.mean_squared_error(y_test, y_pred))
    
    RMSE_avg.append(np.mean(RMSE))
    print(np.mean(RMSE), '\n', alph, '\n', RMSE)


# In[7]:


print(X_train.shape,X_test.shape, '\n', y_train.shape,y_test.shape)


# In[8]:


print(sk.metrics.mean_squared_error(y_test, y_pred))


# In[9]:


print(RMSE)


# In[10]:


print(RMSE_avg)


# In[11]:


result = pd.concat([pd.DataFrame(RMSE_avg)])
print(result)
pd.DataFrame(result).to_csv("submit.csv", index=False, header=False, decimal='.', sep=' ', float_format='%.1f')


# In[12]:


print(max(RMSE_avg))


# In[ ]:





# In[ ]:




