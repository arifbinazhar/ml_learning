# -*- coding: utf-8 -*-
"""
# Support Vector Machine
"""

import numpy as np
import pandas as pd

df = pd.read_csv('Diabetes.csv')
df.head()

df.shape

df = df.dropna()

x = df.drop('Outcome', axis=1)
y = df['Outcome']

k = 5
 folds = np.random.choice(k, size = df.shape[0], replace = True)

 # k_fold split
 x_train = x[folds != 0]
 x_test = x[folds == 0]

 y_train = y[folds != 0]
 y_test = y[folds == 0]

x_test.shape

from sklearn.svm import SVC

clf = SVC(kernel='linear')
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report
a = accuracy_score(y_test, y_pred)
r = recall_score(y_test, y_pred)
p = precision_score(y_test, y_pred)
f = f1_score(y_test, y_pred)
c = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

print(a , r ,p , f)
print(c)
print(cr)

performance = {}
for kernel in ['linear', 'rbf', 'poly']:
  for c in [0.1, 1 , 10. , 100.]:
    for gamma in ['scale', 'auto',0.001, 0.01]:
      for degree in [2,3,4]:
        clf = SVC(kernel=kernel, C=c, gamma=gamma, degree=degree)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        a = accuracy_score(y_test, y_pred)
        performance[(kernel,c, gamma, degree)] = a
        print(performance)

