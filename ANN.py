# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 16:17:06 2018

@author: morten
"""

from matplotlib.pyplot import figure, show
from scipy.io import loadmat
from toolbox_02450 import dbplotf
import numpy as np
import sklearn.neural_network as nn

from initData import *

K = 10
CV = model_selection.KFold(K,shuffle=True)

errors = np.ones(K)
errorData = np.ones([K,3])
k=0

iStart = 5
iEnd = 20
for train_index, test_index in CV.split(stdX,classY):
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
    
    # extract training and test set for current CV fold
    X_train = stdX[train_index,:]
    y_train = classY[train_index]
    X_test = stdX[test_index,:]
    y_test = classY[test_index]
    
    for i in range(5,20):
        ## ANN Classifier, i.e. MLP with one hidden layer
        clf = nn.MLPClassifier(solver='lbfgs',alpha=1e-4,
                               hidden_layer_sizes=(i,), random_state=1)
        clf.fit(X_train,y_train)
        if (np.sum(clf.predict(X_test)!=y_test)/len(y_test))<=errors[k]:
            errors[k] = np.sum(clf.predict(X_test)!=y_test)/len(y_test)
            errorData[k,0] = i
            errorData[k,1] = np.sum(clf.predict(X_test)!=y_test)
            errorData[k,2] = len(y_test)
        # print(i,'Number of miss-classifications for ANN:\n\t {0} out of {1}'.format(np.sum(clf.predict(X_test)!=y_test),len(y_test)))
    print(i,'Number of miss-classifications for ANN:\n\t {0} out of {1}'.format(np.sum(clf.predict(X_test)!=y_test),len(y_test)))
    k+=1
   
NHiddenUnits = 19   
   
clf = nn.MLPClassifier(solver='lbfgs',alpha=1e-4,
                       hidden_layer_sizes=(NHiddenUnits,), random_state=1)
clf.fit(X_train,y_train)

figure(1)
def neval(xval):
    return np.argmax(clf.predict_proba(xval),1)

dbplotf(X_test,y_test,neval,'auto')
show()

