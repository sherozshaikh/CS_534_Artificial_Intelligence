#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from feature_engine.encoding import MeanEncoder


# In[2]:


training_data = pd.read_csv('../transactions_data/transactions_train.csv')


# In[3]:


print(training_data.head(3).to_string())


# In[8]:


dummy_encoded_training_data = pd.get_dummies(training_data, columns=['Card Brand', 'Card Type', 'Use Chip'])


# In[9]:


print(dummy_encoded_training_data.head(3).to_string())


# In[11]:


def mean_encode_category(df: pd.DataFrame, categories, target):
    df_x = df.filter(items=categories)
    df_y = df[target]

    me = MeanEncoder(ignore_format=True)

    me.fit(df_x, df_y)
    transformed = me.transform(df_x)

    df_others = df.drop(labels=categories, axis=1)
    joined = pd.concat([df_others, transformed], axis=1)
    return joined


# In[12]:


mean_encoded_training_data = mean_encode_category(dummy_encoded_training_data, ['State', 'Merchant State', 'Year PIN last Changed', 'Year', 'MCC', 'Errors?', 'Expiration Year', 'Opened Year'], 'Is Fraud?')


# In[13]:


print(mean_encoded_training_data.head(3).to_string())


# In[14]:


def sequential_feature_selection(df: pd.DataFrame, target: str, dropped_columns, direction='forward', tol=None):
    df = df.drop(columns=dropped_columns, axis=1, errors='ignore')
    df_y = df[target]
    df_x = df.drop(columns=[target, 'Unnamed: 0'], axis=1, errors='ignore')

    dt = DecisionTreeClassifier()

    sfs = SequentialFeatureSelector(estimator=dt, direction=direction, scoring='roc_auc', tol=tol)
    sfs = sfs.fit(df_x, df_y)
    df_x_transformed = sfs.transform(df_x)
    return sfs, df_x_transformed



# In[30]:


ffs, f_X = sequential_feature_selection(mean_encoded_training_data, 'Is Fraud?', ['Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0', 'User', 'Card', 'Time'], tol=0.000005)


# In[31]:


print(ffs.feature_names_in_)
print()
print(ffs.get_feature_names_out())


# In[24]:


bfs, b_X = sequential_feature_selection(mean_encoded_training_data, 'Is Fraud?', ['Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0', 'User', 'Card', 'Time'], direction='backward', tol=-0.0001)


# In[25]:


print(bfs.feature_names_in_)
print()
print(bfs.get_feature_names_out())


# In[19]:


from sklearn.feature_selection import RFE


# In[20]:


def recursive_feature_elimination(df: pd.DataFrame, target, dropped_columns, num_of_features=10):
    df = df.drop(columns=dropped_columns, axis=1, errors='ignore')
    df_y = df[target]
    df_x = df.drop(columns=[target], axis=1, errors='ignore')

    dt = DecisionTreeClassifier()

    rfe = RFE(dt, step=1, verbose=1, n_features_to_select=num_of_features)
    rfe = rfe.fit(df_x, df_y)

    df_transformed = rfe.transform(df_x)

    return rfe, df_transformed
    


# In[22]:


rfe_instance, rfe_X = recursive_feature_elimination(mean_encoded_training_data, 'Is Fraud?', ['Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0', 'User', 'Card', 'Time'], num_of_features=10)


# In[23]:


print(rfe_instance.get_feature_names_out())


# In[54]:


print(rfe_instance.feature_names_in_)
print()
print(rfe_instance.ranking_)

