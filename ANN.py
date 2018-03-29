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
from sklearn import model_selection
from scipy import stats

from initData import *

#Max hidden layers
iStart = 1
iEnd = 10

K1 = 5
K2 = 10
CV1 = model_selection.KFold(n_splits=K1,shuffle=True)
CV2 = model_selection.KFold(n_splits=K2,shuffle=True)

"""
error = np.zeros([K,iEnd-1])
errors = np.ones(K)
errorData = np.ones([K,3])
bestnet = list()
train_error = list()
"""

# Initialize variable
valError = np.zeros((K2,iEnd))
sGenError = np.zeros((1,iEnd))
testError = np.zeros(K1)
previous=1e100

i = 0
#for train_index, test_index in CV2.split(stdX):
for train_index, test_index in CV1.split(stdX,classY):
    j = 0
    print('Outer loop: {0}/{1}'.format(i+1,K1))
    X_par = stdX[train_index,:]
    y_par = classY[train_index]
    X_test = stdX[test_index,:]
    y_test = classY[test_index]   
    for train_index, test_index in CV2.split(stdX,classY):
        print('\tCrossvalidation fold: {0}/{1}'.format(j+1,K2))    
        
        # extract training and test set for current CV fold
        X_train = stdX[train_index,:]
        y_train = classY[train_index]
        X_val = stdX[test_index,:]
        y_val = classY[test_index]
    
        # Fit classifier and classify the test points (consider 1 to 40 neighbors)
        for s in range(iStart,iEnd+1):
            clf = nn.MLPClassifier(solver='lbfgs',alpha=1e-4,
                               hidden_layer_sizes=(s,), random_state=1)
            clf.fit(X_train, y_train)
            y_est = clf.predict(X_val)
            
            valError[j,s-1] = np.mean(y_est!=y_val)
            #print('\t\tmiss-classifications:\t {0} out of {1}'.format(np.sum(y_est!=y_val),len(y_val)))
        j+=1
    sGenError = (len(X_val)/len(X_par))*np.sum(valError,axis=0)   
    
    figure()
    plot(100*np.mean(valError,axis=0))
    xlabel('Number of hidden layers')
    ylabel('Classification error rate (%)')
    show()    
    
    bestModel = nn.MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes = (np.argmin(sGenError)+1,), random_state=1)
    bestModel.fit(X_par,y_par)
    
    testError[i] = np.mean(bestModel.predict(X_test)!=y_test)
    
    if(testError[i]<previous):
        absBest = bestModel
        absBestPred = absBest.predict(X_test)
        previous = testError[i]
    #genE[i] = sum(valError[i],0)/len(parX)
    i+=1
genError = (len(X_test)/N)*np.sum(testError,axis=0)
print('K-fold CV done')





"""
print('Error rate: {0}%'.format(100*np.mean(error)))

figure()
plot(100*np.mean(error,axis=0))
xlabel('Number of hidden layers')
ylabel('Classification error rate (%)')
show()

#does not work? Something with dimension mismatch?
CV2 = model_selection.KFold(n_splits=2,shuffle=True)
for train_index, test_index in CV2.split(stdX,classY):
    Xtr = stdX[train_index,:]
    ytr = classY[train_index]
    Xte = stdX[test_index,:]
    yte = classY[test_index] 

NHiddenUnits = 9   
   
clf = nn.MLPClassifier(solver='lbfgs',alpha=1e-4,
                       hidden_layer_sizes=(NHiddenUnits,), random_state=1)
clf.fit(Xtr,ytr)

figure(1)
def neval(xval):
    return np.argmax(clf.predict_proba(xval),1)

dbplotf(Xte,yte,neval,'auto')
show()
"""