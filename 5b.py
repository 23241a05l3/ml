# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 11:49:28 2025

@author: bunny
"""

import numpy as np
import pandas as pd

dataset = pd.read_csv(r'50_Startups.csv')
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,4]

states = pd.get_dummies(x['State'])
x = x.drop('State',axis = 1)
x = pd.concat([x,states],axis =1)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
from sklearn.metrics import r2_score
score = r2_score(y_test,y_pred)
print(score)
print('mean_sqrd_error is==',mean_squared_error(y_test,y_pred))
print('root_mean_squared error of is==',np.sqrt(mean_squared_error(y_test,y_pred)))