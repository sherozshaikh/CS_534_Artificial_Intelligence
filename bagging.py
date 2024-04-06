import pandas as pd
import pickle
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import joblib
from sklearn.metrics import matthews_corrcoef, f1_score, precision_score, recall_score, accuracy_score

# Load your data
X_train = pd.read_pickle('./features_files_20231029/X_train_all_20231028.pkl')
y_train = pd.read_pickle('./features_files_20231029/y_train_20231028.pkl')
X_test = pd.read_pickle('./features_files_20231029/X_test_all_20231028.pkl')
y_test = pd.read_pickle('./features_files_20231029/y_test_20231028.pkl')

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

svm_model_filename = './Models/Support_Vector_Model.pkl'
naive_bayes_model_filename = './Models/Naive_Bayes_Model.pkl'
log_reg_model_filename = './Models/Logistic_Regression_Model.pkl'
knn_model_filename = './Models/KNN_Model.pkl'

with open(svm_model_filename, 'rb') as file:
    svm = pickle.load(file)

with open(naive_bayes_model_filename, 'rb') as file:
    naive_bayes = pickle.load(file)

with open(log_reg_model_filename, 'rb') as file:
    log_reg = pickle.load(file)

with open(knn_model_filename, 'rb') as file:
    knn = pickle.load(file)

bag_1 = BaggingClassifier(base_estimator=knn, n_estimators=10, random_state=123)
bag_2 = BaggingClassifier(base_estimator=svm, n_estimators=10, random_state=123)
bag_3 = BaggingClassifier(base_estimator=log_reg, n_estimators=10, random_state=123)
bag_4 = BaggingClassifier(base_estimator=naive_bayes, n_estimators=10, random_state=123)

bag_1.fit(X_train, y_train)
bag_2.fit(X_train, y_train)
bag_3.fit(X_train, y_train)
bag_4.fit(X_train, y_train)

bag_1_prediction = bag_1.predict(X_test)
bag_2_prediction = bag_2.predict(X_test)
bag_3_prediction = bag_3.predict(X_test)
bag_4_prediction = bag_4.predict(X_test)

bags = [bag_1, bag_2, bag_3, bag_4]
bag_names = ['KNN Bag', 'SVM Bag', 'Logistic Regression Bag', 'Naive Bayes Bag']

for bag, bag_name in zip(bags, bag_names):
    bag_prediction = bag.predict(X_test)

    binary_prediction = (bag_prediction >= 0.5).astype(int)

    mcc = matthews_corrcoef(y_test, binary_prediction)
    f1 = f1_score(y_test, binary_prediction)
    precision = precision_score(y_test, binary_prediction)
    recall = recall_score(y_test, binary_prediction)
    accuracy = accuracy_score(y_test, binary_prediction)

    print(f'\nMetrics for {bag_name}:')
    print(f'MCC: {mcc}')
    print(f'F1 Score: {f1}')
    print(f'Precision: {precision}')
    print(f'Sensitivity (Recall): {recall}')
    print(f'Accuracy: {accuracy}')


combined_prediction = (bag_1_prediction + bag_2_prediction + bag_3_prediction + bag_4_prediction) / 4
print(combined_prediction)

binary_prediction = (combined_prediction >= 0.5).astype(int)

mcc = matthews_corrcoef(y_test, binary_prediction)
print(f'MCC: {mcc}')

f1 = f1_score(y_test, binary_prediction)
print(f'F1 Score: {f1}')

precision = precision_score(y_test, binary_prediction)
print(f'Precision: {precision}')

recall = recall_score(y_test, binary_prediction)
print(f'Sensitivity (Recall): {recall}')

with open('./Models/Bags/knn_bag.pkl', 'wb') as bag1:
    pickle.dump(bag_1, bag1)

with open('./Models/Bags/svm_bag.pkl', 'wb') as bag2:
    pickle.dump(bag_2, bag2)

with open('./Models/Bags/lr_bag.pkl', 'wb') as bag3:
    pickle.dump(bag_3, bag3)

with open('./Models/Bags/nb_bag.pkl', 'wb') as bag4:
    pickle.dump(bag_4, bag4)