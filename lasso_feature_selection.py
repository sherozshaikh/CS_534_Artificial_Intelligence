#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[2]:


df = pd.read_pickle('../features_files_20231029/X_train_all_20231028.pkl')
validation = pd.read_pickle('../features_files_20231029/X_test_all_20231028.pkl')
y_valid = pd.read_pickle('../features_files_20231029/y_test_20231028.pkl')
y_valid = pd.DataFrame(y_valid, columns=['Is Fraud?'])
validation = pd.concat([validation, y_valid], axis=1)

#data = pd.read_csv('../feature_engineering_dataset.csv')
#df, validation = train_test_split(data, test_size=0.2, random_state=42)


# In[3]:


pd.set_option('display.max_columns', None)
df.head(2)


# In[4]:


df = df.drop(columns = ['Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0'])
validation = validation.drop(columns = ['Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0'])


# In[5]:


# Fill missing entries
df['Errors?'] = df['Errors?'].fillna('None')
validation['Errors?'] = validation['Errors?'].fillna('None')


# In[6]:


# Map states to region
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


# In[7]:


# Function to map states to region
def map_merchant_state(df: pd.Series):
    mapped_state = df['Merchant State']
    state = df['Merchant State']
    #print(state)
    if state != state:
        mapped_state = 'Online'
    elif len(state) > 2:
        mapped_state = 'Foreign'
    elif state in state_map.keys():
        mapped_state = state_map[state]
    else:
        mapped_state = state
    df['Merchant State'] = mapped_state
    return df


# In[8]:


df = df.apply(lambda x: map_merchant_state(x), axis=1)
validation = validation.apply(lambda x: map_merchant_state(x), axis=1)


# In[9]:


pd.set_option('display.max_columns', None)
df.head(2)


# In[10]:


#columns_to_encode = ['Errors?', 'Card Brand', 'Card Type', 'State', 'Use Chip', 'Merchant State', 'MCC', 'Year PIN last Changed', 'Year', 'Expiration Year', 'Gender', 'Merchant Country', 'Has Chip', 'Card Brand Type', 'State_US_region']
columns_to_encode = ['Errors?', 'Card Brand', 'Card Type', 'State', 'Use Chip', 'Merchant State', 'MCC', 'Year PIN last Changed', 'Year', 'Expiration Year', 'Gender', 'Merchant Country', 'Has Chip', 'Card Combined', 'Acct Open Date', 'Expires']

label_encoder = LabelEncoder()

for col in columns_to_encode:
    label_encoder.fit(df[col])
    df[col] = label_encoder.transform(df[col])
    label_encoder.fit(validation[col])
    validation[col] = label_encoder.transform(validation[col])


# In[11]:


df.columns


# In[12]:


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    return df[indices_to_keep].astype(np.float64)

df = clean_dataset(df)
validation = clean_dataset(validation)


# In[13]:


#X = df.drop(columns=['Is Fraud?'])
#y = df['Is Fraud?']
X = df
y = pd.read_pickle('../features_files_20231029/y_train_20231028.pkl')#df['Is Fraud?']


# In[14]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[15]:


lasso = Lasso(alpha=0.01)
lasso.fit(X_scaled, y)


# In[16]:


selected_features = X.columns[lasso.coef_ != 0]


# In[17]:


selected_features


# In[18]:


from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

validation_selected_features = validation[selected_features]
model = LogisticRegression()
model.fit(X[selected_features], y)
y_pred = model.predict(validation_selected_features)
validation_accuracy = accuracy_score(validation['Is Fraud?'], y_pred)
validation_accuracy


# In[19]:


# End of Lasso


# In[20]:


# Ridge (not good because seems to be selecting all features)


# In[21]:


from sklearn.linear_model import RidgeCV
from sklearn.metrics import accuracy_score


#X = df.drop(columns=['Is Fraud?'])
#y = df['Is Fraud?']
X = df # df.drop(columns=['Is Fraud?'])
y = pd.read_pickle('../features_files_20231029/y_train_20231028.pkl') #df['Is Fraud?']

X_validation = validation.drop(columns=['Is Fraud?'])
y_validation = validation['Is Fraud?']
print("y val", y_validation)


# In[22]:


alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0] # List of alpha values to test
ridge = RidgeCV(alphas=alphas, store_cv_values=True)

ridge.fit(X, y)

best_alpha = ridge.alpha_

final_ridge_model = Ridge(alpha=best_alpha)
final_ridge_model.fit(X, y)

feature_importances = final_ridge_model.coef_
feature_importances = feature_importances.ravel()

selected_features = X.columns[feature_importances != 0]
print("Selected Features:", selected_features)


# In[23]:


X_validation = validation.drop(columns=['Is Fraud?'])
y_validation = validation['Is Fraud?']

y_pred_validation = final_ridge_model.predict(X_validation)
print("y pred val", y_pred_validation)

threshold = 0.5
y_pred_validation_binary = [1 if pred >= threshold else 0 for pred in y_pred_validation]

validation_accuracy = accuracy_score(y_validation, y_pred_validation_binary)

print("Validation Accuracy:", validation_accuracy)


# In[25]:


from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import column_or_1d

X2 = df.drop(columns=['Is Fraud?'])
y2 = df['Is Fraud?']
#X2 = df
#y2 = pd.read_pickle('../features_files_20231029/y_train_20231028.pkl')

X2_validation = validation.drop(columns=['Is Fraud?'])
y2_validation = validation['Is Fraud?']
#y2 = column_or_1d(y2, warn=True)

alphas = [0.1, 0.5, 1.0, 10.0, 50.0]  # Larger alpha values

l1_ratios = [0.1, 0.2, 0.5, 0.7, 0.9]  # L1 ratio values (0.0 corresponds to Ridge, 1.0 to Lasso)
elastic_net = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratios, max_iter=10000)

elastic_net.fit(X2, y2)

best_alpha = elastic_net.alpha_
best_l1_ratio = elastic_net.l1_ratio_

final_elastic_net_model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio, max_iter=10000)
final_elastic_net_model.fit(X2, y2)

feature_importances = final_elastic_net_model.coef_

selected_features = X2.columns[feature_importances != 0]
print("Selected Features:", selected_features)

y_pred_validation2 = final_elastic_net_model.predict(X2_validation)

threshold = 0.5
y_pred_validation_binary2 = [1 if pred >= threshold else 0 for pred in y_pred_validation2]

validation_accuracy2 = accuracy_score(y2_validation, y_pred_validation_binary2)

print("Validation Accuracy:", validation_accuracy2)


# In[ ]:




