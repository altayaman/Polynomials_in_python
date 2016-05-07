'''
sklearn.preprocessing.PolynomialFeatures

Generate polynomial and interaction features.

Generate a new feature matrix consisting of all polynomial combinations of the features 
with degree less than or equal to the specified degree. 

For example, if an input sample is two dimensional and of the form [a, b], 
the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pn

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def f(x):
    """ function to approximate by polynomial interpolation"""
    return x * np.sin(x)

# ------  Read 1st sheet data from excel file  ------------------------
xls_file = 'E:/IQtest_3 - y=-2+1x.xlsx';
xls_file = 'E:/1 - Sinusoid1.xlsx';
xls = pn.ExcelFile(xls_file)
df = xls.parse('Sheet3')
dataSetArray =np.array(df) 

# -------  Create indexes of train data and test data  ---------------------
dataSetSize = len(dataSetArray)
train_size = int(dataSetSize * 0.6)

train_indexes = list(range(train_size))
test_indexes = list(range(train_size, dataSetSize))
print(train_indexes)
print(test_indexes)

# -------  Create X and Y for training  --------------------------------
X_train = dataSetArray[train_indexes, 0]
Y_train = dataSetArray[train_indexes, 1]
X_train = X_train[:, np.newaxis]
# X_train = X_train.reshape(5,1)

# -------  Create X and Y for testing  ---------------------------------
X_test = dataSetArray[test_indexes, 0]
Y_test = dataSetArray[test_indexes, 1]
X_test = X_test[:, np.newaxis]

plt.plot(X_test, Y_test, label="ground truth")
plt.scatter(X_test, Y_test, label="training points")

print(X_test)
print(Y_test)

for degree in [1, 2]:
    model = make_pipeline(PolynomialFeatures(degree, include_bias=True), Ridge())
    model.fit(X_train, Y_train)
    y_plot = model.predict(X_test)
    print('degree = ', degree, '  =>  ', y_plot)
    plt.plot(X_test, y_plot, label="degree %d" % degree)

plt.legend(loc='upper left')
plt.grid()
plt.show()


