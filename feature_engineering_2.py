#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import feature_engine as fe


# In[3]:


transactions = pd.read_csv('cleaned_transactions (1).csv')


# In[6]:


transactions['Acct Open Date'] = pd.to_datetime(transactions['Acct Open Date'])
transactions['Months_Since_Acct_Open'] = ((transactions['Year'] - transactions['Acct Open Date'].dt.year)*12 + (transactions['Month'] - transactions['Acct Open Date'].dt.month))


# In[7]:


print(transactions.head(5).to_string())


# In[8]:


transactions['Debt_Credit_Limit_Ratio'] = transactions['Total Debt']/transactions['Credit Limit']


# In[9]:


print(transactions.head(5).to_string())


# In[10]:


transactions['Yearly_Income_To_Total_Debt_Ratio'] = transactions['Yearly Income - Person']/transactions['Total Debt']


# In[11]:


print(transactions.head(5).to_string())


# In[12]:


transactions['Yearly_Income_To_Credit_Limit_Ratio'] = transactions['Yearly Income - Person']/transactions['Credit Limit']


# In[13]:


print(transactions.head(5).to_string())


# In[15]:


transactions['Num_Of_Cards_Relative_To_Median'] = transactions['Num Credit Cards'] - transactions['Num Credit Cards'].median()


# In[16]:


print(transactions.head(5).to_string())


# In[19]:


transactions['FICO_Score_Binned'] = transactions['FICO Score'].apply(lambda x: (5 if x >= 800 else 4 if x >= 740 else 3 if x >= 670 else 2 if x >= 580 else 1 if x >= 300 else 0))


# In[20]:


print(transactions.head(5).to_string())


# In[25]:


transactions['FICO_Score_Binned_Ordinal'] = transactions['FICO_Score_Binned']
transactions = pd.get_dummies(transactions, columns=['FICO_Score_Binned'])


# In[21]:


transactions['User_Age_Binned'] = transactions['Current Age'].apply(lambda x: (4 if x >= 65 else 3 if x >= 50 else 2 if x >= 35 else 1 if x >= 20 else 0))


# In[27]:


transactions['User_Age_Binned_Ordinal'] = transactions['User_Age_Binned']
transactions = pd.get_dummies(transactions, columns=['User_Age_Binned'])


# In[34]:


print(transactions.head(5).to_string())


# In[29]:


transactions.describe()


# In[37]:


transactions.to_csv('cleaned_transactions (1).csv')

