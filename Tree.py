# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 13:18:55 2018

@author: morten
"""

import numpy as np
from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, show, boxplot
from scipy.io import loadmat
from sklearn import model_selection, tree
import graphviz

from initData import *

# Tree complexity parameter - constraint on maximum depth
tc = np.arange(2, 21, 1)

# K-fold crossvalidation
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)

# Initialize variable
Error_train = np.empty((len(tc),K))
Error_test = np.empty((len(tc),K))

k=0
for train_index, test_index in CV.split(stdX,classY):
    print('Computing CV fold: {0}/{1}..'.format(k+1,K))

    # extract training and test set for current CV fold
    X_train, y_train = stdX[train_index,:], classY[train_index]
    X_test, y_test = stdX[test_index,:], classY[test_index]

    for i, t in enumerate(tc):
        # Fit decision tree classifier, Gini split criterion, different pruning levels
        dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
        dtc = dtc.fit(X_train,y_train.ravel())
        y_est_test = dtc.predict(X_test)
        y_est_train = dtc.predict(X_train)
        # Evaluate misclassification rate over train/test data (in this CV fold)
        misclass_rate_test = sum(np.abs(y_est_test - y_test)) / float(len(y_est_test))
        misclass_rate_train = sum(np.abs(y_est_train - y_train)) / float(len(y_est_train))
        Error_test[i,k], Error_train[i,k] = misclass_rate_test, misclass_rate_train
    k+=1

    
f = figure()
boxplot(Error_test.T)
xlabel('Model complexity (max tree depth)')
ylabel('Test error across CV folds, K={0})'.format(K))

f = figure()
plot(tc, Error_train.mean(1))
plot(tc, Error_test.mean(1))
xlabel('Model complexity (max tree depth)')
ylabel('Error (misclassification rate, CV K={0})'.format(K))
legend(['Error_train','Error_test'])
    
show()

dtcDepthGini = tree.DecisionTreeClassifier(criterion='gini', max_depth=6)
dtcDepthGini = dtcDepthGini.fit(stdX,classY)

out = tree.export_graphviz(dtcDepthGini, out_file='6DepthGini.gvz', feature_names=attributeNames)

graphviz.render('dot','png','6DepthGini.gvz')

"""
dtcGini = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=2)
dtcGini = dtcGini.fit(classX,classIndices)

out = tree.export_graphviz(dtcGini, out_file='2sampleGini.gvz', feature_names=attributeNames)

graphviz.render('dot','png','2sampleGini.gvz')
"""
