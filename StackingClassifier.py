import pickle
import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print('loading models...')
with open("Models/Naive_Bayes_Model.pkl", 'rb') as file:
    naive_bayes = pickle.load(file)
with open("Models/KNN_Model.pkl", 'rb') as file:
    knn = pickle.load(file)
with open("Models/Logistic_Regression_Model.pkl", 'rb') as file:
    log_reg = pickle.load(file)
with open("Models/Support_Vector_Model.pkl", 'rb') as file:
    svm = pickle.load(file)
# assuming we have the below data
# X_train,y_train,X_test,y_test

# assuming we have the model loaded
# knn,svm,log_reg,naive_bayes

# read your own fine tuned model here


models=[
          ('knn',knn),
          ('svm',svm),
          ('log_reg',log_reg),
          ('naive_bayes',naive_bayes)
      ]

print('loading data...')

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

# C. Stacking
print('Stacking models...')
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier

# change the final_estimator model can be any of our trained model or a new model like random forest etc.
lr_stacked_model=StackingClassifier(estimators=models,final_estimator=log_reg,cv=5)
lr_stacked_model.fit(X_train, y_train)
y_pred_lr = lr_stacked_model.predict(X_test)
accuracy_lr = metrics.accuracy_score(y_test,y_pred_lr)
print('Logistic Regression model done')

nb_stacked_model=StackingClassifier(estimators=models,final_estimator=naive_bayes,cv=5)
nb_stacked_model.fit(X_train, y_train)
y_pred_nb = nb_stacked_model.predict(X_test)
accuracy_nb = metrics.accuracy_score(y_test,y_pred_nb)
print('Naive Bayes model done')

knn_stacked_model=StackingClassifier(estimators=models,final_estimator=knn,cv=5)
knn_stacked_model.fit(X_train, y_train)
y_pred_knn = knn_stacked_model.predict(X_test)
accuracy_knn = metrics.accuracy_score(y_test,y_pred_knn)

print('KNN model done')

svm_stacked_model=StackingClassifier(estimators=models,final_estimator=svm,cv=5)
svm_stacked_model.fit(X_train, y_train)
y_pred_svm = svm_stacked_model.predict(X_test)
accuracy_svm = metrics.accuracy_score(y_test,y_pred_svm)
print('SVM model done')

rf_stacked_model=StackingClassifier(estimators=models,final_estimator=RandomForestClassifier(random_state=123),cv=5)
rf_stacked_model.fit(X_train, y_train)
y_pred_rf = rf_stacked_model.predict(X_test)
accuracy_rf = metrics.accuracy_score(y_test,y_pred_rf)
print('Random Forest model done')

print('Model results:')
# Accuracy, Precision, Sensitivity

print('Naive Bayes Accuracy: ' + str(accuracy_nb))
print('KNN Accuracy: ' + str(accuracy_knn))
print('Logistic Regression Accuracy: ' + str(accuracy_lr))
print('Support Vector Machine Accuracy: ' + str(accuracy_svm))
print(f'Random Forest Accuracy: {accuracy_rf}')

nb_cm = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred_nb)
knn_cm = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred_knn)
lr_cm = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred_lr)
svm_cm = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred_svm)
rf_cm = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred_rf)

nb_cm_display = metrics.ConfusionMatrixDisplay(nb_cm)
nb_cm_display.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Stacked Model\nWith Naive Bayes Classifier As The Final Estimator')
plt.show()

knn_cm_display = metrics.ConfusionMatrixDisplay(knn_cm)
knn_cm_display.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Stacked Model\nWith K-Nearest Neighbors Classifier As The Final Estimator')
plt.show()

lr_cm_display = metrics.ConfusionMatrixDisplay(lr_cm)
lr_cm_display.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Stacked Model\nWith Logistic Regression Classifier As The Final Estimator')
plt.show()

svm_cm_display = metrics.ConfusionMatrixDisplay(svm_cm)
svm_cm_display.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Stacked Model\nWith Support Vector Machine Classifier As The Final Estimator')
plt.show()

rf_cm_display = metrics.ConfusionMatrixDisplay(rf_cm)
rf_cm_display.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Stacked Model\nWith Random Forest Classifier As The Final Estimator')
plt.show()

# Matthews Correlation Coefficient

nb_mcc = metrics.matthews_corrcoef(y_true=y_test, y_pred=y_pred_nb)
knn_mcc = metrics.matthews_corrcoef(y_true=y_test, y_pred=y_pred_knn)
lr_mcc = metrics.matthews_corrcoef(y_true=y_test, y_pred=y_pred_lr)
svm_mcc = metrics.matthews_corrcoef(y_true=y_test, y_pred=y_pred_svm)
rf_mcc = metrics.matthews_corrcoef(y_true=y_test, y_pred=y_pred_rf)

print('Naive Bayes MCC: ' + str(nb_mcc))
print('KNN MCC: ' + str(knn_mcc))
print('Logistic Regression MCC: ' + str(lr_mcc))
print('Support Vector Machine MCC: ' + str(svm_mcc))
print(f'Random Forest MCC: {rf_mcc}')

# F1 Score

nb_f1 = metrics.f1_score(y_true=y_test, y_pred=y_pred_nb)
knn_f1 = metrics.f1_score(y_true=y_test, y_pred=y_pred_knn)
lr_f1 = metrics.f1_score(y_true=y_test, y_pred=y_pred_lr)
svm_f1 = metrics.f1_score(y_true=y_test, y_pred=y_pred_svm)
rf_f1 = metrics.f1_score(y_true=y_test, y_pred=y_pred_rf)

print('Naive Bayes F1 Score: ' + str(nb_f1))
print('KNN F1 Score: ' + str(knn_f1))
print('Logistic Regression F1 Score: ' + str(lr_f1))
print('Support Vector Machine F1 Score: ' + str(svm_f1))
print(f'Random Forest F1 Score: {rf_f1}')
# ROC Curves

fig, ax = plt.subplots()
ax.set_title('ROC Curves for Stacked Models')
nb_roc = metrics.RocCurveDisplay.from_predictions(y_true=y_test, y_pred=y_pred_nb, ax=ax, name='Naive Bayes')
knn_roc = metrics.RocCurveDisplay.from_predictions(y_true=y_test, y_pred=y_pred_knn, ax=ax, name='K-Nearest Neighbors')
lr_roc = metrics.RocCurveDisplay.from_predictions(y_true=y_test, y_pred=y_pred_lr, ax=ax, name='Logistic Regression')
svm_roc = metrics.RocCurveDisplay.from_predictions(y_true=y_test, y_pred=y_pred_svm, ax=ax, name='Support Vector Machine')
rf_roc = metrics.RocCurveDisplay.from_predictions(y_true=y_test, y_pred=y_pred_rf, ax=ax, name='Random Forest')

plt.show()

with open("Models/Naive_Bayes_Model_stacked.pkl", 'wb') as file:
    pickle.dump(nb_stacked_model, file)
with open("Models/KNN_Model_stacked.pkl", 'wb') as file:
    pickle.dump(knn_stacked_model, file)
with open("Models/Logistic_Regression_Model_stacked.pkl", 'wb') as file:
    pickle.dump(lr_stacked_model, file)
with open("Models/Support_Vector_Model_stacked.pkl", 'wb') as file:
    pickle.dump(svm_stacked_model, file)
with open("Models/Random_Forest_Model_stacked.pkl", 'wb') as file:
    pickle.dump(rf_stacked_model, file)