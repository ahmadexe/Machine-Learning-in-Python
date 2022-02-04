from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier


dib = datasets.load_diabetes()

features = dib.data
labels = dib.target
print(dib.DESCR)


clf = KNeighborsClassifier()
clf.fit(features, labels)
pred = clf.predict([[20, 0, 20, 2, 12, 1,1,1,1,1]])
print(pred)
