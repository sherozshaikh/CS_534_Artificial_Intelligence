#!/usr/bin/env python
# coding: utf-8

# In[104]:


import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder


# In[105]:


resampled_data = pd.read_csv('re_sampled_fraud_dect.csv')


# In[106]:


resampled_data = resampled_data.drop(labels=['Card Number', 'CVV', 'Card on Dark Web', 'Person', 'Address', 'Apartment', 'City', 'Zipcode', 'Birth Year', 'Merchant Name', 'Merchant City','Zip'], axis=1)


# In[87]:


print(resampled_data.head(5).to_string())


# In[107]:


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


# In[108]:


resampled_data['Credit Limit'] = resampled_data['Credit Limit'].str.replace('$','', regex=True)
resampled_data['Expiration Year'] = pd.to_datetime(resampled_data['Expires']).dt.year
resampled_data['Opened Year'] = pd.to_datetime(resampled_data['Acct Open Date']).dt.year
resampled_data['Expiration Month'] = pd.to_datetime(resampled_data['Expires']).dt.month
resampled_data['Opened Month'] = pd.to_datetime(resampled_data['Acct Open Date']).dt.month
resampled_data['Amount'] = resampled_data['Amount'].str.replace('$', '', regex=True)
resampled_data['Errors?'] = resampled_data['Errors?'].fillna('None')
resampled_data['Per Capita Income - Zipcode'] = resampled_data['Per Capita Income - Zipcode'].str.replace("$", '', regex=True)
resampled_data['Yearly Income - Person'] = resampled_data['Yearly Income - Person'].str.replace("$", '', regex=True)
resampled_data['Merchant State'] = resampled_data['Merchant State'].fillna('Online')
resampled_data['Total Debt'] = resampled_data['Total Debt'].str.replace("$", '', regex=True)
resampled_data = resampled_data.apply(lambda x: map_mcc(x), axis=1)


# In[90]:


print(resampled_data.head(5).to_string())


# In[109]:


cleaned_data = resampled_data.drop(labels=['Expires', 'Acct Open Date'], axis=1)


# In[110]:


from datetime import datetime
cleaned_data['Time'] = cleaned_data['Time'].apply(lambda x: datetime.strptime(x, '%H:%M'))
cleaned_data['Hour'] = cleaned_data['Time'].apply(lambda x: x.hour)


# In[111]:


from feature_engine.creation import CyclicalFeatures


# In[112]:


cyc_cols = ['Birth Month', 'Month', 'Day', 'Hour', 'Expiration Month', 'Opened Month']
cyclical = CyclicalFeatures(variables=cyc_cols, drop_original=True)
transformed_data = cyclical.fit_transform(cleaned_data)
print(cyclical.max_values_)
print(transformed_data.head())


# In[113]:


transformed_data.to_csv('cleaned_resampled_data.csv')


# In[114]:


le = LabelEncoder()
is_fraud = le.fit_transform(transformed_data['Is Fraud?'])
print(is_fraud)


# In[115]:


transformed_data['Is Fraud?'] = is_fraud


# In[116]:


def label_encode_categories(df: pd.DataFrame, category):
    df_x = df[category]
    oe = LabelEncoder()
    encoded = oe.fit_transform(df_x)

    df_others = df.drop(labels=[str(category)], axis=1)
    df_others[category] = encoded

    return df_others


# In[117]:


gender_encoded = label_encode_categories(transformed_data, 'Gender')
has_chip_encoded = label_encode_categories(gender_encoded, 'Has Chip')


# In[57]:


dummy_encoded = pd.get_dummies(has_chip_encoded, columns=['Card Brand', 'Card Type'])


# In[118]:


has_chip_encoded['Male'] = has_chip_encoded['Gender']
has_chip_encoded = has_chip_encoded.drop(labels=['Gender'], axis=1)


# In[119]:


print(has_chip_encoded.head(5).to_string())


# In[120]:


has_chip_encoded.to_csv('encoded_resampled_data.csv')

