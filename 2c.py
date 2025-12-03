# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 10:41:30 2025

@author: bunny
"""
from sklearn.preprocessing import MinMaxScaler
import pandas
import numpy
 
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(r"C:\Users\bunny\Downloads\diabetes.csv",names = names)
array = dataframe.values[1:]

x = array[:,0:8]
y = array[:,8]

scaler = MinMaxScaler(feature_range = (0,1))
rescaledx = scaler.fit_transform(x)
numpy.set_printoptions(precision=2)
print(rescaledx)