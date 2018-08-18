# ML
import pandas as pd
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
#
from sklearn.linear_model import LogisticRegression

x_data = np.load('titanic_x_data.npy')
y_data = np.load('titanic_y_data.npy')
print(x_data.shape) #
print(y_data.shape) #
print(x_data[:5])
print(y_data[:5]) #

x_train, x_test, y_train, y_test = model_selection.train_test_split(x_data, y_data, test_size=0.33)

#estimator = LogisticRegression()
estimator = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
estimator.fit(x_train, y_train)

print('estimator.coef_:', estimator.coef_) 
'''
estimator.coef_: [[ 0.29864911 -1.09809075]
 [-0.31026732  0.06349747]
 [-0.21938531  0.80216872]]
'''
print('estimator.intercept_:', estimator.intercept_) #estimator.intercept_: [-0.52099727  0.9849799  -0.84966982]

y_predict = estimator.predict(x_test)
score = metrics.accuracy_score(y_test, y_predict) 
print('score:', score) #
print(metrics.classification_report(y_test, y_predict))
'''
'''

print(x_test[:2])
'''
[[ 1.08412616]
 [-0.50335834]]
'''
y_predict = estimator.predict(x_test[:2])
print(y_predict) #
for y1, y2 in zip(y_test, y_predict):
    print(y1, y2, y1==y2)
    
