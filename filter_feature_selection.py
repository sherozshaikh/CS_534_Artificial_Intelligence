#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

X_train = pd.read_pickle('/Users/christinaberthiaume/Downloads/features_files_20231029/features_files_20231029/X_train_all_20231028.pkl')
y_train = pd.read_pickle('/Users/christinaberthiaume/Downloads/features_files_20231029/features_files_20231029/y_train_20231028.pkl')


# In[3]:


X_train.head()


# In[4]:


X_train_num = X_train.select_dtypes(include='number')


# In[5]:


for col in X_train_num:
    if (col[:7] == 'oh_enc_') or (col[:7] == 'ss_enc_') or (col[:7] == 'mm_enc_'):
        X_train_num = X_train_num.drop(columns=[col])


# In[6]:


X_train_num.columns


# In[7]:


X_train_cont = X_train[['Credit Limit', 'Latitude', 'Longitude', 'Per Capita Income - Zipcode',
                        'Yearly Income - Person', 'Total Debt', 'FICO Score', 'Amount', 'Month_sin', 'Month_cos',
                        'Day_sin', 'Day_cos', 'Day_of_Week_sin', 'Day_of_Week_cos', 'Hour_sin', 'Hour_cos',
                        'Avg_MCC', 'Avg_State', 'User_Amount_Spent_Past_3_Months',
                        'User_Amount_Spent_Past_6_Months', 'User_Amount_Spent_Past_9_Months',
                        'User_Amount_Spent_Past_12_Months', 'User_Amount_Spent_Past_15_Months',
                        'User_Amount_Spent_Past_18_Months', 'User_Amount_Spent_Past_21_Months',
                        'User_Amount_Spent_Past_24_Months', 'PIN last Changed', 'Retirement to Current',
                        'Num_Of_Cards_Relative_To_Median', 'Months_Since_Account_Open', 'Credit_Limit_Utilization',
                        'Debt_Credit_Limit_Ratio', 'Yearly_Income_To_Total_Debt_Ratio', 'Yearly_Income_To_Credit_Limit_Ratio']]


# In[9]:


from sklearn.feature_selection import mutual_info_classif

mutual_info = mutual_info_classif(X_train_cont, y_train, discrete_features=False)

mutual_info = pd.Series(mutual_info)
mutual_info.index = X_train_cont.columns
mutual_info = mutual_info.sort_values(ascending=False)
print(mutual_info)


# In[10]:


mutual_info.plot.bar(figsize=(20, 8))


# In[11]:


from sklearn.feature_selection import SelectKBest, f_classif

fs = SelectKBest(score_func=f_classif, k='all')

# Applying feature selection
fit = fs.fit(X_train_cont, y_train)

features_score = pd.DataFrame(fit.scores_)
features = pd.DataFrame(X_train_cont.columns)
feature_score = pd.concat([features, features_score],axis=1)

# Assigning column names
feature_score.columns = ["Input_Features", "F_Score"]
print(feature_score.nlargest(34, columns="F_Score"))


# In[12]:


feature_score.nlargest(34, columns="F_Score").plot.bar(figsize=(20, 8))


# In[13]:


X_train_dis = X_train[['Cards Issued', 'Current Age', 'Retirement Age', 'User_Active_Past_3_Months',
                       'User_Active_Past_6_Months', 'User_Active_Past_9_Months', 'User_Active_Past_12_Months',
                       'User_Active_Past_15_Months', 'User_Active_Past_18_Months', 'User_Active_Past_21_Months',
                       'User_Active_Past_24_Months', 'Num Credit Cards', '#Pin Changed by User',
                       '#Transaction by User', 'Account Open Age', 'Years in Card Expiration',
                       'Years PIN Change After Account Open', 'lb_enc_Card Brand', 'lb_enc_Card Type',
                       'lb_enc_Has Chip', 'lb_enc_Gender', 'lb_enc_State', 'lb_enc_Use Chip',
                       'lb_enc_Merchant State', 'lb_enc_Errors?', 'lb_enc_State_US_region', 
                       'lb_enc_Merchant Country', 'lb_enc_MCC_CAT', 'lb_enc_Card Brand Type']]


# In[14]:


mutual_info = mutual_info_classif(X_train_dis, y_train, discrete_features=True)

mutual_info = pd.Series(mutual_info)
mutual_info.index = X_train_dis.columns
mutual_info = mutual_info.sort_values(ascending=False)
print(mutual_info)


# In[15]:


mutual_info.plot.bar(figsize=(20, 8))


# In[16]:


from sklearn.feature_selection import chi2

fs = SelectKBest(score_func=chi2, k='all')
fit = fs.fit(X_train_dis, y_train)

features_score = pd.DataFrame(fit.scores_)
features = pd.DataFrame(X_train_dis.columns)
feature_score = pd.concat([features, features_score],axis=1)

# Assigning column names
feature_score.columns = ["Input_Features", "Chi2_Score"]
print(feature_score.nlargest(33, columns="Chi2_Score"))


# In[21]:


feature_score["Chi2_Score"].sort_values(ascending=False).plot.bar(figsize=(20, 8))


# In[27]:


chi2 = list(feature_score["Chi2_Score"].sort_values(ascending=False))


# In[30]:


pd.Series(chi2[4:]).plot.bar(figsize=(20, 8))


# In[ ]:




