# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 06:25:14 2025

@author: bunny
"""

import pandas as pd
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age',
'label']
pima = pd.read_csv(r"diabetes.csv", header=None, names=col_names)
print(pima)

for col in col_names :
    pima[col] = pd.to_numeric(pima[col],errors = 'coerce')
    
pima.dropna(inplace=True)

feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp']
X = pima[feature_cols]
y = pima.label

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(random_state = 0,solver = 'lbfgs',max_iter = 1000)

logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)

from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test,y_pred)
print(cnf_matrix)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))