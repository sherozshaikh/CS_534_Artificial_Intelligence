#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


cards = pd.read_csv('sd254_cards.csv')


# In[3]:


print(cards.head(1).to_string())


# In[4]:


dropped_cards = cards.drop(columns=['Card Number', 'CVV', 'Card on Dark Web'])


# In[5]:


print(dropped_cards.head(1).to_string())


# In[6]:


card_features_encoded = pd.get_dummies(data=dropped_cards, columns=['Card Brand', 'Card Type', 'Has Chip'])


# In[8]:


card_features_encoded = card_features_encoded.drop(columns=['Has Chip_NO'])
print(card_features_encoded.head(5).to_string())


# In[10]:


card_features_encoded['Credit Limit'] = card_features_encoded['Credit Limit'].str.replace('$','', regex=True)


# In[11]:


print(card_features_encoded.head(5).to_string())


# In[12]:


card_features_encoded['EXPIRATION_YEAR'] = pd.to_datetime(card_features_encoded['Expires']).dt.year
card_features_encoded['OPENED_YEAR'] = pd.to_datetime(card_features_encoded['Acct Open Date']).dt.year


# In[14]:


dates_formatted = card_features_encoded.drop(columns=['Expires', 'Acct Open Date'])
print(dates_formatted.head(5).to_string())


# In[15]:


dates_formatted.to_csv('cleaned_cards.csv')

