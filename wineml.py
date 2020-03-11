import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
dataset = pd.read_csv(dataset_url, sep = ";")
print (dataset.head())
X = dataset.iloc[:, 0:11].values
y = dataset.iloc[:, 11].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

"""from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)"""

from sklearn import tree
regressor=tree.DecisionTreeClassifier()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

confidence = regressor.score(X_test, y_test)
