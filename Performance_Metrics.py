
import pandas as pd
import sklearn.metrics as metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


X_train = pd.read_pickle('train_test_split/X_train_20231028.pkl')
X_test = pd.read_pickle('train_test_split/X_test_20231028.pkl')

y_train = pd.read_pickle('train_test_split/y_train_20231028.pkl')
y_test = pd.read_pickle('train_test_split/y_test_20231028.pkl')


nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred_nb = nb_classifier.predict(X_test)
accuracy_nb = metrics.accuracy_score(y_test,y_pred_nb)


knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
y_pred_knn = knn.predict(X_test)
accuracy_knn = metrics.accuracy_score(y_test, y_pred_knn)


lr_classifier = LogisticRegression(random_state=123,solver='lbfgs',C=0.1,max_iter=1000)
lr_classifier.fit(X_train, y_train)
y_pred_lr = lr_classifier.predict(X_test)
accuracy_lr = metrics.accuracy_score(y_test,y_pred_lr)


rf_classifier = RandomForestClassifier(random_state=123)
rf_classifier.fit(X_train,y_train)
y_pred_rf = rf_classifier.predict(X_test)
accuracy_rf = metrics.accuracy_score(y_test,y_pred_rf)

df = pd.DataFrame({'Feature': X_train.columns, 'Feature Importance': rf_classifier.feature_importances_})
df = df.sort_values(by='Feature Importance', ascending=False)
df.head(13)

# Accuracy, Precision, Sensitivity

print('Naive Bayes Accuracy: ' + str(accuracy_nb))
print('KNN Accuracy: ' + str(accuracy_knn))
print('Logistic Regression Accuracy: ' + str(accuracy_lr))
print('Random Forest Accuracy: ' + str(accuracy_rf))

nb_cm = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred_nb)
knn_cm = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred_knn)
lr_cm = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred_lr)
rf_cm = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred_rf)

nb_cm_display = metrics.ConfusionMatrixDisplay(nb_cm)
nb_cm_display.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Naive Bayes Classifier')
plt.show()

knn_cm_display = metrics.ConfusionMatrixDisplay(knn_cm)
knn_cm_display.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for K-Nearest Neighbors Classifier')
plt.show()

lr_cm_display = metrics.ConfusionMatrixDisplay(lr_cm)
lr_cm_display.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Logistic Regression Classifier')
plt.show()

rf_cm_display = metrics.ConfusionMatrixDisplay(rf_cm)
rf_cm_display.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Random Forest Classifier')
plt.show()

# Matthews Correlation Coefficient

nb_mcc = metrics.matthews_corrcoef(y_true=y_test, y_pred=y_pred_nb)
knn_mcc = metrics.matthews_corrcoef(y_true=y_test, y_pred=y_pred_knn)
lr_mcc = metrics.matthews_corrcoef(y_true=y_test, y_pred=y_pred_lr)
rf_mcc = metrics.matthews_corrcoef(y_true=y_test, y_pred=y_pred_rf)

print('Naive Bayes MCC: ' + str(nb_mcc))
print('KNN MCC: ' + str(knn_mcc))
print('Logistic Regression MCC: ' + str(lr_mcc))
print('Random Forest MCC: ' + str(rf_mcc))

# F1 Score

nb_f1 = metrics.f1_score(y_true=y_test, y_pred=y_pred_nb)
knn_f1 = metrics.f1_score(y_true=y_test, y_pred=y_pred_knn)
lr_f1 = metrics.f1_score(y_true=y_test, y_pred=y_pred_lr)
rf_f1 = metrics.f1_score(y_true=y_test, y_pred=y_pred_rf)

print('Naive Bayes F1 Score: ' + str(nb_f1))
print('KNN F1 Score: ' + str(knn_f1))
print('Logistic Regression F1 Score: ' + str(lr_f1))
print('Random Forest F1 Score: ' + str(rf_f1))

# ROC Curves

fig, ax = plt.subplots()
ax.set_title('ROC Curves for Baseline Models')
nb_roc = metrics.RocCurveDisplay.from_predictions(y_true=y_test, y_pred=y_pred_nb, ax=ax, name='Naive Bayes')
knn_roc = metrics.RocCurveDisplay.from_predictions(y_true=y_test, y_pred=y_pred_knn, ax=ax, name='K-Nearest Neighbors')
lr_roc = metrics.RocCurveDisplay.from_predictions(y_true=y_test, y_pred=y_pred_lr, ax=ax, name='Logistic Regression')
rf_roc = metrics.RocCurveDisplay.from_predictions(y_true=y_test, y_pred=y_pred_rf, ax=ax, name='Random Forest')


