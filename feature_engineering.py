#!/usr/bin/env python
# coding: utf-8

# In[287]:


import pandas as pd

transactions = pd.read_csv('/Users/christinaberthiaume/Documents/School/Fall 2023/CS 534 - Artificial Intelligence/Credit Card Fraud Project/resampled_transactions.csv')


# In[288]:


transactions.head()


# In[289]:


transactions.columns


# In[290]:


# Drop unwanted columns
transactions = transactions.drop(columns = ['Unnamed: 0', 'Person', 'Birth Month', 'Address', 'Apartment', 'City', 'Zipcode', 'Card Number', 'CVV', 'Card on Dark Web', 'Merchant Name', 'Merchant City', 'Zip'])


# In[291]:


# Encode Is Fraud? column
transactions["Is Fraud?"] = transactions["Is Fraud?"].apply(lambda x: 1 if x == 'Yes' else 0)


# In[292]:


# Fill missing entries and encode Errors? column
transactions['Errors?'] = transactions['Errors?'].fillna('None')
transactions['Error_binary'] = transactions['Errors?'].apply(lambda x: 0 if x == 'None' else 1)


# In[293]:


def err_fraud_enc(transaction):
    if transaction['Errors?'] == 'None':
        transaction['Error_fraud'] = 0
    elif transaction['Is Fraud?'] == 0:
        transaction['Error_fraud'] = 1
    else:
        transaction['Error_fraud'] = 2
    return transaction


# In[294]:


transactions = transactions.apply(lambda x: err_fraud_enc(x), axis=1)


# In[295]:


def err_fraud_count(transaction):
    counts = transactions.groupby('Errors?')['Is Fraud?'].sum()
    for error, count in counts.iteritems():
        if transaction['Errors?'] == error:
            transaction['Error_count'] = count
    return transaction


# In[296]:


transactions = transactions.apply(lambda x: err_fraud_count(x), axis=1)


# In[297]:


def err_fraud_ratio(transaction):
    counts = transactions.groupby('Errors?')['Is Fraud?'].count()
    for error, count in counts.iteritems():
        if transaction['Errors?'] == error:
            transaction['Error_ratio'] = transaction['Error_count']/count
    return transaction


# In[298]:


transactions = transactions.apply(lambda x: err_fraud_ratio(x), axis=1)


# In[299]:


# Encode Gender column
transactions['Female'] = transactions['Gender'].apply(lambda x: 1 if x == 'Female' else 0)


# In[300]:


# Encode Has Chip column
transactions['Has Chip'] = transactions['Has Chip'].apply(lambda x: 1 if x == 'YES' else 0)


# In[301]:


# Turn money values into floats
transactions['Per Capita Income - Zipcode'] = transactions['Per Capita Income - Zipcode'].str.replace("$", '', regex=True).astype(float)
transactions['Yearly Income - Person'] = transactions['Yearly Income - Person'].str.replace("$", '', regex=True).astype(float)
transactions['Total Debt'] = transactions['Total Debt'].str.replace("$", '', regex=True).astype(float)
transactions['Credit Limit'] = transactions['Credit Limit'].str.replace('$','', regex=True).astype(float)
transactions['Amount'] = transactions['Amount'].str.replace('$', '', regex=True).astype(float)


# In[302]:


# Map states to region
# https://www.faa.gov/air_traffic/publications/atpubs/cnt_html/appendix_a.html
# https://www.mappr.co/political-maps/us-regions-map/
state_region_mapping = {
  'AL' : 'Southeast',
  'AK' : 'West',
  'AZ' : 'Southwest',
  'AR' : 'Southeast',
  'CA' : 'West',
  'CO' : 'West',
  'CT' : 'Northeast',
  'DE' : 'Northeast',
  'DC' : 'Southeast',
  'FL' : 'Southeast',
  'GA' : 'Southeast',
  'HI' : 'West',
  'ID' : 'West',
  'IL' : 'Midwest',
  'IN' : 'Midwest',
  'IA' : 'Midwest',
  'KS' : 'Midwest',
  'KY' : 'Southeast',
  'LA' : 'Southeast',
  'ME' : 'Northeast',
  'MD' : 'Northeast',
  'MA' : 'Northeast',
  'MI' : 'Midwest',
  'MN' : 'Midwest',
  'MS' : 'Southeast',
  'MO' : 'Midwest',
  'MT' : 'West',
  'NE' : 'Midwest',
  'NV' : 'West',
  'NH' : 'Northeast',
  'NJ' : 'Northeast',
  'NM' : 'Southwest',
  'NY' : 'Northeast',
  'NC' : 'Southeast',
  'ND' : 'Midwest',
  'OH' : 'Midwest',
  'OK' : 'Southwest',
  'OR' : 'West',
  'PA' : 'Northeast',
  'RI' : 'Northeast',
  'SC' : 'Southeast',
  'SD' : 'Midwest',
  'TN' : 'Southeast',
  'TX' : 'Southwest',
  'UT' : 'West',
  'VT' : 'Northeast',
  'VA' : 'Southeast',
  'WA' : 'West',
  'WV' : 'Southeast',
  'WI' : 'Midwest',
  'WY' : 'West'
}


# In[303]:


# Function to map states to region
def map_merchant_state(transaction: pd.Series):
    mapped_state = transaction['Merchant State']
    state = transaction['Merchant State']
    #print(state)
    if state != state:
        mapped_state = 'Online'
    elif len(state) > 2:
        mapped_state = 'Foreign'
    elif state in state_region_mapping.keys():
        mapped_state = state_region_mapping[state]
    else:
        mapped_state = state
    transaction['Merchant State'] = mapped_state
    return transaction


# In[304]:


# Function to map merchant state to country
def merchant_country(mer_state):
    if mer_state in state_region_mapping:
        return 'US'
    elif pd.isna(mer_state):
        return 'Online'
    else:
        return mer_state


# In[305]:


# Function to map MCC to industry
def map_mcc(transaction: pd.Series):
    mcc = transaction['MCC']
    if mcc < 1500:
        mapped_mcc = 'Agricultural Services'
    elif 1500 <= mcc < 3000:
        mapped_mcc = 'Contracted Services'
    elif 3000 <= mcc < 3300:
        mapped_mcc = 'Airlines'
    elif 3300 <= mcc < 3500:
        mapped_mcc = 'Car Rental'
    elif 3500 <= mcc < 4000:
        mapped_mcc = 'Lodging'
    elif 4000 <= mcc < 4800:
        mapped_mcc = 'Transportation Services'
    elif 4800 <= mcc < 5000:
        mapped_mcc = 'Utility Services'
    elif 5000 <= mcc < 5600:
        mapped_mcc = 'Retail Outlet Services'
    elif 5600 <= mcc < 5700:
        mapped_mcc = 'Clothing Stores'
    elif 5700 <= mcc < 7300:
        mapped_mcc = 'Micellaneous Stores'
    elif 7300 <= mcc < 8000:
        mapped_mcc = 'Business Services'
    elif 8000 <= mcc < 9000:
        mapped_mcc = 'Professional Services and Membership Organizations'
    elif 9000 <= mcc < 10000:
        mapped_mcc = 'Government Services'
    transaction['MCC'] = mapped_mcc
    return transaction


# In[306]:


transactions['State'] = transactions['State'].map(state_region_mapping)
transactions['Merchant Country'] = transactions['Merchant State'].map(merchant_country)
transactions = transactions.apply(lambda x: map_merchant_state(x), axis=1)
transactions = transactions.apply(lambda x: map_mcc(x), axis=1)


# In[307]:


# Extract day of week from date
transactions['Date'] = pd.to_datetime(transactions[['Year', 'Month', 'Day']])
transactions['Day of Week'] = transactions['Date'].dt.dayofweek


# In[308]:


transactions['Hour'] = pd.to_datetime(transactions['Time']).dt.hour


# In[309]:


# Create cyclical features
from feature_engine.creation import CyclicalFeatures

cyc_cols = ['Month', 'Day', 'Day of Week', 'Hour']
cyclical = CyclicalFeatures(variables=cyc_cols, drop_original=False)
transactions = cyclical.fit_transform(transactions)
print(cyclical.max_values_)
print(transactions.head())


# In[310]:


transactions['Expiration Year'] = pd.to_datetime(transactions['Expires']).dt.year
transactions['Year Opened'] = pd.to_datetime(transactions['Acct Open Date']).dt.year


# In[311]:


transactions.columns


# In[312]:


transactions['Acct Open Age'] = transactions['Year Opened'] - transactions['Birth Year']
transactions['Years til Card Expires'] = transactions['Expiration Year'] - transactions['Year Opened']
transactions['Years til Retirement'] = transactions['Retirement Age'] - transactions['Current Age']
transactions['Credit Limit Utilization'] = transactions['Amount'] / transactions['Credit Limit']
transactions['Years til PIN Change'] = transactions['Year PIN last Changed'] - transactions['Year Opened']


# In[313]:


transactions = transactions.drop(columns=['Time', 'Date'])


# In[314]:


transactions.head()


# In[315]:


transactions.dtypes


# In[316]:


transactions.to_csv('cleaned_transactions.csv')


# In[ ]:




