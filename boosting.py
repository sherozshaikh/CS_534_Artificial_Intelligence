#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

X_train = pd.read_pickle('/Users/christinaberthiaume/Downloads/features_files_20231029/features_files_20231029/X_train_all_20231028.pkl')
y_train = pd.read_pickle('/Users/christinaberthiaume/Downloads/features_files_20231029/features_files_20231029/y_train_20231028.pkl')

X_test = pd.read_pickle('/Users/christinaberthiaume/Downloads/features_files_20231029/features_files_20231029/X_test_all_20231028.pkl')
y_test = pd.read_pickle('/Users/christinaberthiaume/Downloads/features_files_20231029/features_files_20231029/y_test_20231028.pkl')


# In[2]:


X_train = X_train[['User_Amount_Spent_Past_3_Months', 'Per Capita Income - Zipcode',
                   'Yearly Income - Person', 'Avg_State', 'User_Amount_Spent_Past_24_Months',
                   'User_Amount_Spent_Past_6_Months', 'User_Amount_Spent_Past_9_Months',
                   'User_Amount_Spent_Past_15_Months', 'User_Amount_Spent_Past_18_Months',
                   'Current Age', 'Total Debt', 'Hour_cos', 'User_Amount_Spent_Past_12_Months']]

X_test = X_test[['User_Amount_Spent_Past_3_Months', 'Per Capita Income - Zipcode',
                   'Yearly Income - Person', 'Avg_State', 'User_Amount_Spent_Past_24_Months',
                   'User_Amount_Spent_Past_6_Months', 'User_Amount_Spent_Past_9_Months',
                   'User_Amount_Spent_Past_15_Months', 'User_Amount_Spent_Past_18_Months',
                   'Current Age', 'Total Debt', 'Hour_cos', 'User_Amount_Spent_Past_12_Months']]


# In[3]:


import pickle

with open('/Users/christinaberthiaume/Downloads/Models/KNN_Model.pkl', 'rb') as file:
    knn_model = pickle.load(file)

with open('/Users/christinaberthiaume/Downloads/Models/Logistic_Regression_Model.pkl', 'rb') as file:
    lr_model = pickle.load(file)

with open('/Users/christinaberthiaume/Downloads/Models/Naive_Bayes_Model.pkl', 'rb') as file:
    nb_model = pickle.load(file)

with open('/Users/christinaberthiaume/Downloads/Models/Support_Vector_Model.pkl', 'rb') as file:
    svm_model = pickle.load(file)


# In[4]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, confusion_matrix

#boost_1=AdaBoostClassifier(estimator=knn_model, n_estimators=10, random_state=123, algorithm='SAMME')

boost_2=AdaBoostClassifier(estimator=svm_model, n_estimators=25, random_state=123, algorithm='SAMME')

boost_3=AdaBoostClassifier(estimator=lr_model, n_estimators=10, random_state=123)

boost_4=AdaBoostClassifier(estimator=nb_model, n_estimators=10, random_state=123)


# In[5]:


boost_2=AdaBoostClassifier(estimator=svm_model, n_estimators=50, random_state=123, algorithm='SAMME')
boost_2_fit=boost_2.fit(X_train, y_train)
boost_2_pred=boost_2_fit.predict(X_test)
print('SVM Accuracy: ', accuracy_score(y_test, boost_2_pred))
print('SVM MCC: ', matthews_corrcoef(y_test, boost_2_pred))
print('SVM F1 Score: ', f1_score(y_test, boost_2_pred))


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

parameters = {'n_estimators': [15, 50, 100, 250, 500], 'learning_rate': [0.001, 0.01, 0.1]}
adb = AdaBoostClassifier(estimator=svm_model, random_state=123, algorithm='SAMME')
clf = GridSearchCV(adb, parameters, cv = 3, scoring='accuracy')
clf.fit(X_train, y_train)
print(clf.best_params_)
boost_2=AdaBoostClassifier(estimator=svm_model, random_state=123, algorithm='SAMME',
                           n_estimators=clf.best_params_['n_estimators'],
                           learing_rate=clf.best_params_['learning_rate'])
boost_2_fit=boost_2.fit(X_train, y_train)
boost_2_pred=boost_2_fit.predict(X_test)
print('SVM Accuracy: ', accuracy_score(y_test, boost_2_pred))
print('SVM MCC: ', matthews_corrcoef(y_test, boost_2_pred))
print('SVM F1 Score: ', f1_score(y_test, boost_2_pred))


# In[5]:


"""
boost_1_fit=boost_1.fit(X_train, y_train)
boost_1_pred=boost_1_fit.predict(X_test)
print('KNN Accuracy: ', accuracy_score(boost_1_pred, y_test))
print('KNN MCC: ', matthews_corrcoef(boost_1_pred, y_test))
print('KNN F1 Score: ', f1_score(boost_1_pred, y_test))


boost_2_fit=boost_2.fit(X_train, y_train)
boost_2_pred=boost_2_fit.predict(X_test)
print('SVM Accuracy: ', accuracy_score(y_test, boost_2_pred))
print('SVM MCC: ', matthews_corrcoef(y_test, boost_2_pred))
print('SVM F1 Score: ', f1_score(y_test, boost_2_pred))
"""

boost_3_fit=boost_3.fit(X_train, y_train)
boost_3_pred=boost_3_fit.predict(X_test)
print('Logistic Regression Accuracy: ', accuracy_score(y_test, boost_3_pred))
print('Logistic Regression MCC: ', matthews_corrcoef(y_test, boost_3_pred))
print('Logistic Regression F1 Score: ', f1_score(y_test, boost_3_pred))

boost_4_fit=boost_4.fit(X_train, y_train)
boost_4_pred=boost_4_fit.predict(X_test)
print('Naive Bayes Accuracy: ', accuracy_score(y_test, boost_4_pred))
print('Naive Bayes MCC: ', matthews_corrcoef(y_test, boost_4_pred))
print('Naive Bayes F1 Score: ', f1_score(y_test, boost_4_pred))


# In[19]:


import matplotlib.pyplot as plt
from sklearn import metrics

#cm_1 = metrics.confusion_matrix(boost_1_pred, y_test)
#cm_2 = metrics.confusion_matrix(y_test, boost_2_pred)
cm_3 = metrics.confusion_matrix(y_test, boost_3_pred)
cm_4 = metrics.confusion_matrix(y_test, boost_4_pred)

#display_1 = metrics.ConfusionMatrixDisplay(confusion_matrix = cm_1, display_labels = [False, True])
#display_2 = metrics.ConfusionMatrixDisplay(confusion_matrix = cm_2, display_labels = [False, True])
display_3 = metrics.ConfusionMatrixDisplay(confusion_matrix = cm_3, display_labels = [False, True])
display_4 = metrics.ConfusionMatrixDisplay(confusion_matrix = cm_4, display_labels = [False, True])

#display_1.plot()
#display_2.plot(cmap=plt.cm.Blues)
display_3.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix for Boosted Model\n with Logistic Regression as Base Estimator')
display_4.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix for Boosted Model\n with Naive Bayes as Base Estimator')
plt.show()


# In[8]:


from sklearn.ensemble import StackingClassifier

estimators = [('knn', knn_model), ('svm', svm_model), ('lr', lr_model), ('nb', nb_model)]

stacker = AdaBoostClassifier()   
model = StackingClassifier(estimators=estimators, final_estimator= stacker)
stack = model.fit(X_train, y_train)
stack_pred = stack.predict(X_test)
print('Accuracy: ', accuracy_score(y_test, stack_pred))
print('MCC: ', matthews_corrcoef(y_test, stack_pred))
print('F1 Score: ', f1_score(y_test, stack_pred))


# In[21]:


cm = metrics.confusion_matrix(y_test, stack_pred)

display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [False, True])

display.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Stacked and\n Boosted Model')
plt.show()


# In[ ]:




