# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 10:31:09 2025

@author: bunny
"""
from sklearn.preprocessing import Binarizer
import pandas 
import numpy
names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
dataframe = pandas.read_csv(r"C:\Users\bunny\Downloads\diabetes.csv",names = names)
array = dataframe.values[1:]
x = array[:,0:8]
y = array[:,8]
binarizer = Binarizer(threshold = 0.0).fit(x)
binaryx = binarizer.transform(x)
numpy.set_printoptions(precision = 2)
print(binaryx[0:5,:])
