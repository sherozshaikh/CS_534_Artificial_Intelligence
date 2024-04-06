#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math


# In[2]:


chunks = []

for i in range(25):
    filename = f'transactions/transaction_chunk_{i}.csv'
    
    chunk = pd.read_csv(filename)
    chunks.append(chunk)

combined_dataframe = pd.concat(chunks, ignore_index=True)


# In[3]:


combined_dataframe.describe()


# In[4]:


from datetime import datetime
combined_dataframe['Time'] = combined_dataframe['Time'].apply(lambda x: datetime.strptime(x, '%H:%M'))
combined_dataframe['Hour'] = combined_dataframe['Time'].apply(lambda x: x.hour)


# In[5]:


from feature_engine.creation import CyclicalFeatures


# In[6]:


cyc_cols = ['Month', 'Day', 'Hour']
cyclical = CyclicalFeatures(variables=cyc_cols, drop_original=True)
transformed_data = cyclical.fit_transform(combined_dataframe)
print(cyclical.max_values_)
print(transformed_data.head())


# In[7]:


transformed_data.to_csv('transactions.csv', index=False)


# In[ ]:




