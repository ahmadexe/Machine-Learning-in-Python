from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
housing = pd.read_csv('data.csv')
train_set, test_set = train_test_split(housing, test_size= 0.2)
print(len(train_set))
print(len(test_set))
s = StratifiedShuffleSplit(n_splits= 1, test_size= 0.2, random_state= 42)
for train_index, test_index in s.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
print(strat_test_set.info())
print(strat_test_set['CHAS'].value_counts())
hcm = housing.corr()
print(hcm)
housing = strat_train_set.drop("MEDV", axis= 1)

housing_lable = strat_train_set["MEDV"].copy()


print(housing.info())
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

mypipeline = Pipeline([
('imputer', SimpleImputer(strategy='median')),
('std_scalar', StandardScaler())
])


housing_w = mypipeline.fit_transform(housing)
print(housing_w.shape)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
#ml_model = LinearRegression()
ml_model = DecisionTreeRegressor()
ml_model.fit(housing_w, housing_lable)



test_random_data = housing.iloc[:5]
test_random_lables = housing_lable.iloc[:5]


prep_data = mypipeline.transform(test_random_data)
mypre = ml_model.predict(prep_data)


print(mypre)

print(list(test_random_lables))

mse = mean_squared_error(mypre, test_random_lables)
print(mse)


sc = cross_val_score(ml_model, prep_data, prep_data, test_random_lables, scoring="neg_mean_squared_error")
rmse = np.sqrt(-sc)
print(rmse)

