from sklearn import datasets
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

ir = datasets.load_iris()
features = ir.data
labels = ir.target
print(ir.DESCR)

clf = KNeighborsClassifier()
clf.fit(features, labels)

sl = eval(input("Enter sepal length: "))
sw = eval(input("Enter sepal width: "))
pl = eval(input("Enter petal length: "))
pw = eval(input("Enter petal width: "))

pred = clf.predict([[sl , sw , pl , pw]])
print(pred)

