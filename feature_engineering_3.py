#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import feature_engine


# In[2]:


transactions = pd.read_csv('cleaned_transactions (1).csv')


# In[3]:


transactions.columns


# In[4]:


transactions['Card Combined'] = transactions['Card Brand'] + ' ' + transactions['Card Type']


# In[5]:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()


# In[6]:


transactions['Card Combination_LabelEncoded'] = label_encoder.fit_transform(transactions['Card Combined'])


# In[7]:


ohe_card = pd.get_dummies(transactions['Card Combined'], prefix='Card_Combined', dtype=int)
transactions = pd.concat([transactions, ohe_card], axis=1)


# In[8]:


pd.set_option('display.max_columns', None)
transactions.head(5)


# In[9]:


avg_amount_per_mcc = transactions.groupby('MCC')['Amount'].mean().reset_index()
avg_amount_per_mcc.columns = ['MCC', 'Avg_MCC']
transactions = transactions.merge(avg_amount_per_mcc, on='MCC', how='left')


# In[10]:


avg_amount_per_state = transactions.groupby('Merchant State')['Amount'].mean().reset_index()
avg_amount_per_state.columns = ['Merchant State', 'Avg_State']
transactions = transactions.merge(avg_amount_per_state, on='Merchant State', how='left')


# In[11]:


pd.set_option('display.max_columns', None)
transactions.head(5)


# In[12]:


transactions['Transaction_Date'] = pd.to_datetime(transactions[['Year', 'Month', 'Day', 'Hour']])

transactions = transactions.sort_values(by=['User', 'Transaction_Date'])

def calculate_spending_n_period(df, n):
    df['User_Active_Past_{}_Months'.format(n)] = df.groupby('User')['Transaction_Date'].diff().dt.days.le(n*30).astype(int)
    df['User_Amount_Spent_Past_{}_Months'.format(n)] = df.groupby('User')['Amount'].rolling(n*30, min_periods=1).sum().reset_index(level=0, drop=True)

calculate_spending_n_period(transactions, 3)
calculate_spending_n_period(transactions, 6)
calculate_spending_n_period(transactions, 9)
calculate_spending_n_period(transactions, 12)
calculate_spending_n_period(transactions, 15)
calculate_spending_n_period(transactions, 18)
calculate_spending_n_period(transactions, 24)

transactions.drop('Transaction_Date', axis=1, inplace=True)


# In[13]:


pd.set_option('display.max_columns', None)
transactions.head(5)


# In[14]:


transactions.to_csv('cleaned_transactions (1).csv')


# In[ ]:




