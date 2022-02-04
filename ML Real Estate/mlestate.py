import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# def split_train_set(data, test_ratio):
#     np.random.seed(42)
#     shuffled = np.random.permutation(len(data))
#     print(shuffled)
#     test_set_length = int(len(data)*test_ratio)
#     test_indices = shuffled[:test_set_length]
#     train_indices = shuffled[test_set_length:]
#     return data.iloc[train_indices], data.iloc[test_indices]

housing = pd.read_csv('data.csv')
# print(housing.head())
# print(housing.info())
# print(housing['CHAS'])
# print(housing['CHAS'].value_counts())
# print(housing.describe())
# housing.hist(bins = 50,figsize = (100,80))
# plt.show()

# test_set, train_set = split_train_set(housing, .2)
# print(len(test_set))
# print(len(train_set))



from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
train_set, test_set = train_test_split(housing, test_size= 0.2, random_state= 42)
print(len(train_set))
print(len(test_set))