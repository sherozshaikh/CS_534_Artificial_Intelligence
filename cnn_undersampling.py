#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math


# In[2]:


transactions = pd.read_csv('transactions.csv')


# In[3]:


from imblearn.under_sampling import CondensedNearestNeighbour


# In[4]:


print(transactions.columns)


# In[5]:


transactions['IsFraud?'] = transactions['Is Fraud?_No'].replace({1: 0, 0: 1}).fillna(transactions['Is Fraud?_Yes'])
cols = ['Is Fraud?_Yes', 'Is Fraud?_No']
transactions = transactions.drop(columns=cols)


# In[6]:


transactions["Amount"] = transactions["Amount"].str.replace("$", "", regex=False).astype(float)
cols = ['Merchant State', 'Errors?']
transactions = transactions.drop(columns=cols)


# In[9]:


X = transactions.drop(["IsFraud?"], axis=1)
y = transactions["IsFraud?"]


# In[14]:


y = y.to_numpy()


# In[15]:


X.shape


# In[16]:


y.shape


# In[ ]:


#timestamp_columns = transactions.select_dtypes(include=['datetime64']).columns
#transformed_data = transactions.drop(columns=timestamp_columns)


# In[17]:


transactions.dtypes


# In[18]:


cnn = CondensedNearestNeighbour(sampling_strategy='auto', random_state=0)
X_resampled, y_resampled = cnn.fit_resample(X, y)
resampled_data = pd.concat([X_resampled, y_resampled], axis=1)


# In[19]:


y_resampled = pd.DataFrame(data=y_resampled, columns=["IsFraud"])


# In[20]:


resampled_data = pd.concat([X_resampled, y_resampled], axis=1)


# In[21]:


resampled_data.head()


# In[22]:


resampled_data.shape


# In[23]:


resampled_data.to_csv('resampled_transactions.csv', index=False)


# In[ ]:




