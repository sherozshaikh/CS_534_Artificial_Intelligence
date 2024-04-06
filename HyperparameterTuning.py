
import pandas as pd
import sklearn.metrics as metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

print('Reading in data...')

X_train = pd.read_pickle('train_test_split/X_train_all_20231028.pkl')
X_test = pd.read_pickle('train_test_split/X_test_all_20231028.pkl')

y_train = pd.read_pickle('train_test_split/y_train_20231028.pkl')
y_test = pd.read_pickle('train_test_split/y_test_20231028.pkl')

X_train = X_train[
    ['User_Amount_Spent_Past_3_Months', 'Per Capita Income - Zipcode', 'Yearly Income - Person', 'Avg_State',
     'User_Amount_Spent_Past_24_Months', 'User_Amount_Spent_Past_6_Months', 'User_Amount_Spent_Past_9_Months',
     'User_Amount_Spent_Past_15_Months', 'User_Amount_Spent_Past_18_Months', 'Current Age', 'Total Debt', 'Hour_cos',
     'User_Amount_Spent_Past_12_Months']]
X_test = X_test[
    ['User_Amount_Spent_Past_3_Months', 'Per Capita Income - Zipcode', 'Yearly Income - Person', 'Avg_State',
     'User_Amount_Spent_Past_24_Months', 'User_Amount_Spent_Past_6_Months', 'User_Amount_Spent_Past_9_Months',
     'User_Amount_Spent_Past_15_Months', 'User_Amount_Spent_Past_18_Months', 'Current Age', 'Total Debt', 'Hour_cos',
     'User_Amount_Spent_Past_12_Months']]

print('Fitting models...')

nb_classifier = GaussianNB()
gnb_distr = dict(var_smoothing=[5e-10, 6e-10, 7e-10, 8e-10, 9e-10, 1e-9, 2e-9, 3e-9, 4e-9, 5e-9])
gnb_cv = GridSearchCV(nb_classifier, gnb_distr, cv=5, scoring=metrics.make_scorer(metrics.matthews_corrcoef))

gnb_cv.fit(X_train, y_train)

print(f'Best MCC for Gaussian Naive Bayes: {gnb_cv.best_score_}')
print(f'Best set of parameters for Gaussian Naive Bayes: {gnb_cv.best_params_}')

knn_model = KNeighborsClassifier()
knn_distr = dict(n_neighbors=[1, 3, 5, 7, 9], p=[1, 2], algorithm=['auto', 'ball_tree', 'kd_tree', 'brute'])
knn_cv = GridSearchCV(knn_model, knn_distr, cv=5, scoring=metrics.make_scorer(metrics.matthews_corrcoef))

knn_cv.fit(X_train, y_train)

print(f'Best MCC for KNN: {knn_cv.best_score_}')
print(f'Best set of parameters for KNN: {knn_cv.best_params_}')

lr_model = LogisticRegression(random_state=123)
lr_distr = dict(penalty=['l1', 'l2', 'elasticnet', None], C=[0.5, 1, 1.5, 2, 2.5, 3],
                solver=['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'])
lr_cv = GridSearchCV(lr_model, lr_distr, cv=5, scoring=metrics.make_scorer(metrics.matthews_corrcoef))

lr_cv.fit(X_train, y_train)

print(f'Best MCC for Logistic Regression: {lr_cv.best_score_}')
print(f'Best set of parameters for Logistic Regression: {lr_cv.best_params_}')

print('Fitting parameters...')
svm_model = SVC(random_state=123)
svm_distr = dict(C=[0.5, 1, 1.5], kernel=['rbf', 'sigmoid'],
                 gamma=['scale', 'auto', 0.1])
svm_cv = GridSearchCV(svm_model, svm_distr, cv=5, scoring=metrics.make_scorer(metrics.matthews_corrcoef), verbose=10)

svm_cv.fit(X_train, y_train)

print(f'Best MCC for SVM: {svm_cv.best_score_}')
print(f'Best set of parameters for SVM: {svm_cv.best_params_}')
