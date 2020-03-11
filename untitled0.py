import pandas as pd

dataset = pd.read_csv('car.data')
dataset.columns = ['buying', 'maint', 'doors', 'persons', 'ug_boots', 'safety', 'car']

dataset = dataset.replace('vhigh',4)
dataset = dataset.replace('high',3)
dataset = dataset.replace('med',2)
dataset = dataset.replace('low',1)
dataset = dataset.replace('5more',6)
dataset = dataset.replace('more',5)
dataset = dataset.replace('small',1)
dataset = dataset.replace('med',2)
dataset = dataset.replace('big',3)
dataset = dataset.replace('unacc',1)
dataset = dataset.replace('acc',2)
dataset = dataset.replace('good',3)
dataset = dataset.replace('vgood',4)

X = dataset.iloc[:, 0:6].values
y = dataset.iloc[:, 6].values

X,y = X.astype(int), y.astype(int)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'gini', random_state = 0, n_jobs = 1)  
classifier.fit(X_train, y_train)  

accuracy=classifier.score(X_test,y_test) 
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) 

