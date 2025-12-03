# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 11:01:43 2025

@author: bunny
"""

from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig

A = array([[1,2],[3,4],[5,6]])
print(A )

M = mean(A.T, axis = 1)
print(M)
C = A - M
V = cov(C.T)
print(V)

values,vectors = eig(V)

print(vectors)
print(values)

P = vectors.T.dot(C.T)
print(P.T)