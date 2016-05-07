'''  
Uses numpy.polyfit
fits formula:  p(x) = p[0] * x**deg + ... + p[deg]

'''

import numpy as np
import pandas as pn
import scipy.interpolate as i 
from numpy import shape
import matplotlib.pyplot as plt

# ------  Read 1st sheet data from excel file  ------------------------
xls_file = 'E:/IQtest_2 - y=x^2.xlsx';
xls = pn.ExcelFile(xls_file)
# print('Excel file sheets: ', xls.sheet_names)

df = xls.parse('Sheet1')
# print(df['Order'][:5])

# ------  Convert data into array  -------------------------------------
# ----------------------------------------------------------------------
dataSetArray =np.array(df) 

# -------  Create indexes of train data and test data  ---------------------
dataSetSize = len(dataSetArray)
train_size = int(dataSetSize * 0.6)

train_indexes = list(range(train_size))
test_indexes = list(range(train_size, dataSetSize))
print(train_indexes)
print(test_indexes)


# -------  Create X and Y for training  --------------------------------
# ----------------------------------------------------------------------
X_train = dataSetArray[train_indexes, 0]
Y_train = dataSetArray[train_indexes, 1]
# X_train = X_train.reshape(5,1)        # Reshape array into [5,1] because 1d array will be interpreted as single sample
X_train = X_train[:, np.newaxis]


# -------  Create X and Y for testing  ---------------------------------
# ----------------------------------------------------------------------
X_test = dataSetArray[test_indexes, 0]
Y_test = dataSetArray[test_indexes, 1]
# X_test = X_test[:, np.newaxis]
# X_test = X_test.reshape(4,1) 


# print('Polynomial Regression Training...')
# # Set polynomial with degree 1 (e.g. y = 1 + theta * x)
# poly_1_coeffs = np.polyfit(df['Order'][:5], df['Result'][:5], 1)
# poly_prediction = np.poly1d(poly_1_coeffs)
# 
# print('\nPolynomail coeffs: ', poly_1_coeffs)
# 
# print('\nPolynomail Regression Predicting...')
# Poly_prediction = poly_prediction(X_test)#.astype(np.int32)
# print(Poly_prediction)



plt.plot(X_test, Y_test, label="ground truth")
plt.scatter(X_test, Y_test, label="training points")

print(X_test)
print(Y_test)

for degree in [1, 2, 3, 4]:
    poly_coeffs = np.polyfit(df['Order'][:5], df['Result'][:5], degree)
#     poly_coeffs = np.polyfit(X_train, Y_train, degree)
    poly_prediction = np.poly1d(poly_coeffs)
    Poly_prediction = poly_prediction(X_test)
    print('degree = ', degree, '  =>  ', Poly_prediction, '  Coefs: ', poly_coeffs)
    plt.plot(X_test, Poly_prediction, label="degree %d" % degree)


plt.legend(loc='upper left')
plt.show()