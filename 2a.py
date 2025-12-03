from sklearn.preprocessing import StandardScaler 
import pandas
import numpy

dataframe = pandas.read_csv(r"C:\Users\bunny\Downloads\diabetes.csv",names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'],header = None )
array = dataframe.values[1:]
print(array)
x = array [:,0:8]
y = array[:,8]
scaler = StandardScaler().fit(x)
rescaledx = scaler.transform(x)

numpy.set_printoptions(precision=7)
print(rescaledx[0:5,:])