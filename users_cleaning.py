#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


users = pd.read_csv('sd254_users.csv')


# In[4]:


print(users.head(1).to_string())


# # Cleaning User data

# In[5]:


users = users.drop(columns=['Person', 'Birth Month', 'Address', 'Apartment', 'City', 'Zipcode', 'Birth Year'])


# In[6]:


print(users.head(1).to_string())


# In[7]:


gender = pd.get_dummies(data=users, columns=['Gender'])
gender = gender.drop(columns=['Gender_Male'])
print(gender.head(5).to_string())


# In[8]:


gender['Per Capita Income - Zipcode'] = gender['Per Capita Income - Zipcode'].str.replace("$", '', regex=True)
gender['Yearly Income - Person'] = gender['Yearly Income - Person'].str.replace("$", '', regex=True)
gender['Total Debt'] = gender['Total Debt'].str.replace("$", '', regex=True)


# In[9]:


print(gender.head(5).to_string())


# In[10]:


state_encoding = pd.get_dummies(data=gender, columns=['State'])
print(state_encoding.head(5).to_string())


# In[11]:


state_encoding.to_csv('cleaned_users.csv')


# In[1]:


from imblearn.under_sampling import CondensedNearestNeighbour


# In[ ]:




