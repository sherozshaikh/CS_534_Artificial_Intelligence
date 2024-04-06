# Credit Card Fraud Detection Using Machine Learning

## University & Course Details
- Course: CS 534 - Artificial Intelligence
- University: Worcester Polytechnic Institute
- Semester: Fall 2023

## Authors
- Christina Berthiaume
- Nathaniel Itty
- Owen Radcliffe
- Sheroz Shaikh

## Abstract
As digital transactions become increasingly prevalent, the risk of credit card fraud has grown substantially, necessitating the development of robust and effective fraud detection systems. This paper presents a comprehensive study that evaluates and compares the performance of four prominent machine learning methods, Logistic Regression, K-Nearest Neighbors (KNN), Naive Bayes, and Support Vector Machine (SVM), in the context of credit card fraud classification. The dataset we are using to train these models contains information on over 2.4 million transactions, including information on the card and the user. Since most of the transactions are not fraudulent, we have a very imbalanced dataset. Therefore, we first resample the data before training our classification models. Next, we train our four machine learning models and evaluate them, choosing the best model for our goal. In this case, we must minimize the number of false negatives or cases where the model predicts that there is no credit card fraud when there is because we want to correctly classify all the instances of fraud. We will choose the model that not only has good accuracy but also has good sensitivity.

## Keywords
Classification, Imbalanced dataset, Undersampling, Logistic Regression, K-Nearest Neighbors, Naïve Bayes, Support Vector Machine, Credit Card Fraud, Ensemble methods, Bagging, Boosting, Stacking
