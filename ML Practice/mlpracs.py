from sklearn import linear_model, datasets
from sklearn.metrics import mean_squared_error




bc = datasets.load_breast_cancer()
bc_X = bc.data

bc_x_train = bc.data[:-20]
bc_y_train = bc.target[:-20]
bc_x_test = bc.data[-20:]
bc_y_test = bc.target[-20:]


model = linear_model.LinearRegression()
model.fit(bc_x_train, bc_y_train)
bc_y_predict = model.predict(bc_x_test)

print("Mean squared error: ", mean_squared_error(bc_y_test, bc_y_predict))
print("Weights: ", model.coef_)
print("Intercept: ", model.intercept_)



