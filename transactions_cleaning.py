#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import math


# In[10]:


transactions = pd.read_csv('../Data/archive/credit_card_transactions-ibm_v2.csv', chunksize=1000000)


# In[11]:


# Dropping unwanted columns 
i = 0
for chunk in transactions:
    dropped_columns = chunk.drop(columns=['Merchant Name', 'Merchant City','Zip'])
    dropped_columns.to_csv('transactions/transaction_chunk_' + str(i)+ '.csv')
    i += 1


# In[12]:


# Turning 'Use Chip' and 'Is Fraud?' into dummy variables
for n in range(0,25):
    transaction_chunk = pd.read_csv('transactions/transaction_chunk_' + str(n)+ '.csv')
    dummy = pd.get_dummies(data=transaction_chunk, columns=['Use Chip', 'Is Fraud?'])
    
    dummy.to_csv('transactions/transaction_chunk_' + str(n)+ '.csv')


# In[13]:


# Categorizing states by region
state_map = dict()

state_map["WA"] = "Northwest"
state_map["OR"] = "Northwest"
state_map["ID"] = "Northwest"
state_map["MT"] = "Northwest"
state_map["WY"] = "Northwest"

state_map["CA"] = "West"
state_map["NV"] = "West"
state_map["AZ"] = "West"
state_map["CO"] = "West"
state_map["NM"] = "West"
state_map["UT"] = "West"

state_map["ND"] = "Northcentral"
state_map["SD"] = "Northcentral"
state_map["NE"] = "Northcentral"
state_map["IA"] = "Northcentral"
state_map["MN"] = "Northcentral"

state_map["TX"] = "Southcentral"
state_map["OK"] = "Southcentral"
state_map["KS"] = "Southcentral"
state_map["MO"] = "Southcentral"
state_map["AR"] = "Southcentral"
state_map["LA"] = "Southcentral"

state_map["MS"] = "Southeast"
state_map["AL"] = "Southeast"
state_map["GA"] = "Southeast"
state_map["FL"] = "Southeast"
state_map["SC"] = "Southeast"
state_map["NC"] = "Southeast"
state_map["TN"] = "Southeast"

state_map["WI"] = "GreatLakes"
state_map["MI"] = "GreatLakes"
state_map["IN"] = "GreatLakes"
state_map["IL"] = "GreatLakes"
state_map["KY"] = "GreatLakes"

state_map["OH"] = "Midatlantic"
state_map["WV"] = "Midatlantic"
state_map["VA"] = "Midatlantic"
state_map["MD"] = "Midatlantic"
state_map["DC"] = "Midatlantic"
state_map["DE"] = "Midatlantic"
state_map["PA"] = "Midatlantic"
state_map["NJ"] = "Midatlantic"

state_map["NY"] = "Northeast"
state_map["CT"] = "Northeast"
state_map["RI"] = "Northeast"
state_map["MA"] = "Northeast"
state_map["VT"] = "Northeast"
state_map["NH"] = "Northeast"
state_map["ME"] = "Northeast"

state_map["AK"] = "AK"
state_map["HI"] = "HI"


# In[14]:


print(state_map["OR"])


# In[15]:


# Maps states to region
def map_merchant_state(transaction: pd.Series):
    mapped_state = transaction['Merchant State']
    state = transaction['Merchant State']
    #print(state)
    if state != state:
        mapped_state = 'Online'
    elif len(state) > 2:
        mapped_state = 'Foreign'
    elif state in state_map.keys():
        mapped_state = state_map[state]
    else:
        mapped_state = state
    transaction['Merchant State'] = mapped_state
    return transaction
    


# In[ ]:


# Mapping states to regions
for i in range(0,25):
    transaction_chunk = pd.read_csv('transactions/transaction_chunk_' + str(i)+ '.csv')
    if 'Is Fraud?_No' in transaction_chunk.columns: 
        dummy = transaction_chunk.drop(columns=['Is Fraud?_No'])
    else:
        dummy = transaction_chunk
    dummy = dummy.apply(lambda x: map_merchant_state(x), axis=1)
    
    dummy.to_csv('transactions/transaction_chunk_' + str(i)+ '.csv')


# In[12]:


# Turning 'Merchant State' into a dummy variable
for i in range(0,25):
    transaction_chunk = pd.read_csv('transactions/transaction_chunk_' + str(i)+ '.csv')
    transaction_chunk = pd.get_dummies(data=transaction_chunk, columns=['Merchant State'])
    transaction_chunk.to_csv('transactions/transaction_chunk_' + str(i)+ '.csv')


# In[14]:


# Getting rid of $ in amount
# Transforming 'Time' into the hour
for i in range(0,25):
    transaction_chunk = pd.read_csv('transactions/transaction_chunk_' + str(i)+ '.csv')
    #print(type(transaction_chunk['Amount'].iloc[0]))
    try:
        transaction_chunk['Amount'] = transaction_chunk['Amount'].str.replace('$', '', regex=True)
    except:
        pass
    #print(transaction_chunk['Amount'])
    transaction_chunk['Time'] = pd.to_datetime(transaction_chunk['Time']).dt.hour

    transaction_chunk.to_csv('transactions/transaction_chunk_' + str(i)+ '.csv')


# In[21]:


# Transforming 'Errors?' into dummy variable
for i in range(0,25):
    transaction_chunk = pd.read_csv('transactions/transaction_chunk_' + str(i)+ '.csv')
    transaction_chunk['Errors?'] = transaction_chunk['Errors?'].fillna('None')
    transaction_chunk = pd.concat((transaction_chunk, transaction_chunk['Errors?'].str.get_dummies(sep=',').add_prefix('Error_')), axis=1)
    transaction_chunk.to_csv('transactions/transaction_chunk_' + str(i)+ '.csv')


# In[24]:


# Drop unwanted columns
for i in range(0,25):
    transaction_chunk = pd.read_csv('transactions/transaction_chunk_' + str(i)+ '.csv')
    transaction_chunk = transaction_chunk.drop(columns=['Errors?', 'Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.2', 'Unnamed: 0.3', 'Unnamed: 0.4', 'Unnamed: 0.5'])
    transaction_chunk.to_csv('transactions/transaction_chunk_' + str(i)+ '.csv')


# In[27]:


# Maps MCC to Industry
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


# In[28]:


# Mapping MCC to Industry
for i in range(0,25):
    transaction_chunk = pd.read_csv('transactions/transaction_chunk_' + str(i)+ '.csv')
    transaction_chunk = transaction_chunk.apply(lambda x: map_mcc(x), axis=1)
    
    transaction_chunk.to_csv('transactions/transaction_chunk_' + str(i)+ '.csv')


# In[30]:


# Transforming 'MCC' into dummy variables
for i in range(0,25):
    transaction_chunk = pd.read_csv('transactions/transaction_chunk_' + str(i)+ '.csv')
    transaction_chunk = pd.get_dummies(data=transaction_chunk, columns=['MCC'])
    transaction_chunk.to_csv('transactions/transaction_chunk_' + str(i)+ '.csv')


# In[33]:


# Dropping unwanted columns
for i in range(0,25):
    transaction_chunk = pd.read_csv('transactions/transaction_chunk_' + str(i)+ '.csv')
    transaction_chunk = transaction_chunk.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.2'])
    transaction_chunk.to_csv('transactions/transaction_chunk_' + str(i)+ '.csv')


# In[38]:


# Idk why it turned everything into floats
for i in range(0,25):
    transaction_chunk = pd.read_csv('transactions/transaction_chunk_' + str(i)+ '.csv')
    transaction_chunk = transaction_chunk.drop(columns=['Unnamed: 0'])
    for var in ['User', 'Card', 'Year', 'Month', 'Day', 'Time', 'Use Chip_Chip Transaction', 'Use Chip_Online Transaction', 'Use Chip_Swipe Transaction', 'Is Fraud?_Yes', 'Merchant State_AK', 'Merchant State_Foreign', 'Merchant State_GreatLakes', 'Merchant State_HI', 'Merchant State_Midatlantic', 'Merchant State_Northcentral', 'Merchant State_Northeast', 'Merchant State_Northwest', 'Merchant State_Online', 'Merchant State_Southcentral', 'Merchant State_Southeast', 'Merchant State_West', 'Error_Bad CVV', 'Error_Bad Card Number', 'Error_Bad Expiration', 'Error_Bad PIN', 'Error_Bad Zipcode', 'Error_Insufficient Balance', 'Error_None', 'Error_Technical Glitch', 'MCC_Airlines', 'MCC_Business Services', 'MCC_Car Rental', 'MCC_Clothing Stores', 'MCC_Contracted Services', 'MCC_Government Services', 'MCC_Lodging', 'MCC_Micellaneous Stores', 'MCC_Professional Services and Membership Organizations', 'MCC_Retail Outlet Services', 'MCC_Transportation Services', 'MCC_Utility Services']:
        transaction_chunk[var] = transaction_chunk[var].astype('int')
    transaction_chunk.to_csv('transactions/transaction_chunk_' + str(i)+ '.csv')


# In[ ]:




