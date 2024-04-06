import pickle
from sklearn.ensemble import VotingClassifier

with open("Models/Naive_Bayes_Model.pkl",'rb') as file:
    clf_naive_bayes=pickle.load(file)

with open("Models/KNN_Model.pkl",'rb') as file:
    clf_knn=pickle.load(file)

with open("Models/Logistic_Regression_Model.pkl",'rb') as file:
    clf_log_reg=pickle.load(file)

with open("Models/Support_Vector_Model.pkl",'rb') as file:
    clf_svm=pickle.load(file)

models=[
    ('knn',clf_knn),
    ('svm',clf_svm),
    ('nb',clf_naive_bayes),
    ('knn',clf_knn),
    ('log_reg',clf_log_reg),
    ('knn',clf_knn),
]

X_train=pd.read_pickle('train_test_split/X_train_all_20231028.pkl')
y_train=pd.read_pickle('train_test_split/y_train_20231028.pkl')

X_train=X_train[
    ['User_Amount_Spent_Past_3_Months','Per Capita Income - Zipcode','Yearly Income - Person','Avg_State',
     'User_Amount_Spent_Past_24_Months','User_Amount_Spent_Past_6_Months','User_Amount_Spent_Past_9_Months',
     'User_Amount_Spent_Past_15_Months','User_Amount_Spent_Past_18_Months','Current Age','Total Debt','Hour_cos',
     'User_Amount_Spent_Past_12_Months']]

votingclassifier_1=VotingClassifier(estimators=models,voting='hard')
votingclassifier_1.fit(X_train,y_train)
with open("Models/VotingClassifier.pkl",'wb') as file:
    pickle.dump(votingclassifier_1,file)

