from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from tkinter import *
from tkinter.messagebox import showinfo
import numpy as np



root = Tk()
root.title("Machine Learning Project 1")
root.geometry('800x450')

ir = datasets.load_iris()
features = ir.data
labels = ir.target
clf = KNeighborsClassifier()
clf.fit(features, labels)


esl = Entry(root)
esw = Entry(root)
epl = Entry(root)
epw = Entry(root)
esl.place(x = 360, y = 50)
esw.place(x = 360, y = 80)
epl.place(x = 360, y = 110)
epw.place(x = 360, y = 140)

def dis():
    showinfo('Info', f"{ir.DESCR}")

def pred(spl_len,spl_width,petal_len,petal_width):
    pred = clf.predict([[esl.get(),esw.get(),epl.get(),epw.get()]])
    if pred == [0]:
        x = 'Iris-Setosa'
    if pred == [1]:
        x = 'Iris-Versicolour'
    if pred == [2]:
        x = 'Iris-Virginica'

    showinfo('Plant Type', x)



Label(root, text = 'Sepal Length').place(x = 270, y = 50)
Label(root, text = 'Sepal Width').place(x = 270, y = 80)
Label(root, text = 'Petal Length').place(x = 270, y = 110)
Label(root, text = 'Petal Width').place(x = 270, y = 140)

Button(root, text = "Get Discription", command = dis).place(x = 285, y = 180)
Button(root, text = "Get Prediction", command = lambda: pred(esl.get(),esw.get(),epl.get(),epw.get())).place(x = 385, y = 180)


root.mainloop()