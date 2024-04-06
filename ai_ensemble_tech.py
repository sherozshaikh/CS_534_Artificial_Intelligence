
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




# A. Bagging

from sklearn.ensemble import BaggingClassifier

bag_1=BaggingClassifier(base_estimator=knn,n_estimators=10,random_state=123)
bag_1.fit(X_train,y_train)

bag_2=BaggingClassifier(base_estimator=svm,n_estimators=10,random_state=123)

bag_3=BaggingClassifier(base_estimator=log_reg,n_estimators=10,random_state=123)

bag_4=BaggingClassifier(base_estimator=naive_bayes,n_estimators=10,random_state=123)

# Assume you have a BaggingClassifier for each model
bag_1=BaggingClassifier(base_estimator=knn,n_estimators=10,random_state=123)
bag_2=BaggingClassifier(base_estimator=svm,n_estimators=10,random_state=123)
bag_3=BaggingClassifier(base_estimator=log_reg,n_estimators=10,random_state=123)
bag_4=BaggingClassifier(base_estimator=naive_bayes,n_estimators=10,random_state=123)

# 1 - we can do a lot here before sending out the final predicitons
(
    bag_1.predict(X_test) +
    bag_2.predict(X_test) +
    bag_3.predict(X_test) +
    bag_4.predict(X_test)
) / 4

# 2 - use the above models or read your own fine tuned model here
bag_5=BaggingClassifier([knn,svm,log_reg,naive_bayes],random_state=123)








# B. Boosting

from sklearn.ensemble import AdaBoostClassifier

boost_1=AdaBoostClassifier(base_estimator=knn,n_estimators=10,random_state=123)

boost_2=AdaBoostClassifier(base_estimator=svm,n_estimators=10,random_state=123)

boost_3=AdaBoostClassifier(base_estimator=log_reg,n_estimators=10,random_state=123)

boost_4=AdaBoostClassifier(base_estimator=naive_bayes,n_estimators=10,random_state=123)

# Assume you have a AdaBoostClassifier for each model
boost_1=AdaBoostClassifier(base_estimator=knn,n_estimators=10,random_state=123)
boost_2=AdaBoostClassifier(base_estimator=svm,n_estimators=10,random_state=123)
boost_3=AdaBoostClassifier(base_estimator=log_reg,n_estimators=10,random_state=123)
boost_4=AdaBoostClassifier(base_estimator=naive_bayes,n_estimators=10,random_state=123)

# 1 - we can do a lot here before sending out the final predicitons
(
    boost_1.predict(X_test) +
    boost_2.predict(X_test) +
    boost_3.predict(X_test) +
    boost_4.predict(X_test)
) / 4

# 2 - use the above models or read your own fine tuned model here
boost_6=AdaBoostClassifier([knn,svm,log_reg,naive_bayes],random_state=123)






# C. Stacking

from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier

# change the final_estimator model can be any of our trained model or a new model like random forest etc.
stacked_model=StackingClassifier(estimators=models,final_estimator=log_reg,cv=5)

stacked_model=StackingClassifier(estimators=models,final_estimator=RandomForestClassifier(),cv=5)










# D. Voting Classifier

from sklearn.ensemble import VotingClassifier

# 1. Hard Voting
voting_clf=VotingClassifier(estimators=models,voting='hard')

# 2. Soft Voting
voting_clf=VotingClassifier(estimators=models,voting='soft')



# all the above models

# training
.fit(X_train,y_train)

# prediction
.predict(X_train,y_train)




