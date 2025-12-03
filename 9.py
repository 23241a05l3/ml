# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 06:48:19 2025

@author: bunny
"""

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split

iris_dataset = load_iris()
print("\n iris features \n target names:",iris_dataset.target_names)
for i in range(len(iris_dataset.target_names)):
    print("\n [{0}] : [{1}]".format(i,iris_dataset.target_names[i]))

X_train, X_test, y_train, y_test = train_test_split(iris_dataset["data"], iris_dataset["target"],random_state=0)

kn = KNeighborsClassifier(n_neighbors=1)
kn.fit(X_train,y_train)
x_new = np.array([[5, 2.9, 1, 0.2]])
print("\n XNEW \n",x_new)
prediction = kn.predict(x_new)
print("\n Predicted target value: {}\n".format(prediction))
print("\n Predicted feature name: {}\n".format(iris_dataset["target_names"][prediction][0]))
for i in range(len(X_test)) :
    x = X_test[i]
    x_new = np.array([x])
    prediction = kn.predict(x_new)
    print("\n Actual : {0} {1}, Predicted :{2} {3}".format(y_test[i],iris_dataset["target_names"][y_test[i]],prediction,iris_dataset["target_names"][prediction][0]))
print("\n TEST SCORE[ACCURACY]: {:.2f}\n".format(kn.score(X_test, y_test)))