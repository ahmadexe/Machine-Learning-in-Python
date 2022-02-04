from sklearn import datasets
from sklearn.linear_model import LogisticRegression
ir = datasets.load_iris()
#print(ir.data.shape)
# print(ir['target'])
print(ir['DESCR'])

x = ir.data
y = ir.target

clf = LogisticRegression()
clf.fit(x, y)
pred = clf.predict([[5.84,3.05,3.76,1.2]])
print(pred)