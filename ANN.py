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
## Cange iEnd to research a larger span
iStart = 1
iEnd = 10
tc = np.arange(iStart, iEnd+1, 1)

K1 = 5
K2 = 10
CV1 = model_selection.KFold(n_splits=K1,shuffle=True)
CV2 = model_selection.KFold(n_splits=K2,shuffle=True)

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
    
    bestModel = nn.MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes = (np.argmin(sGenError)+1,), random_state=1)
    bestModel.fit(X_par,y_par)
    
    testError[i] = np.mean(bestModel.predict(X_test)!=y_test)
    
    print('\n\tBest model: {:0.0f} hidden layers and {:1.2f}% biased test error and {:2.2f}% unbiased'.format(bestModel.hidden_layer_sizes[0],100*np.mean(valError,axis=0)[bestModel.hidden_layer_sizes[0]-1],100*testError[i]))

    
    fig=figure()
    plot(tc,100*np.mean(valError,axis=0))
    xlabel('Number of hidden layers')
    ylabel('Classification error rate (%)')
    show()
    fig.savefig('fig/annErrorFold{0}.eps'.format(i+1), format='eps', dpi=1200)
    fig.clf
    
    if(testError[i]<previous):
        absBest = bestModel
        absBestPred = absBest.predict(X_test)
        previous = testError[i]
    #genE[i] = sum(valError[i],0)/len(parX)
    i+=1
genError = (len(X_test)/N)*np.sum(testError,axis=0)
print('K-fold CV done')
print('The best model has {:0.0f} hidden layers with {:1.2f}% unbiased test error'.format(absBest.hidden_layer_sizes[0],100*previous))
print('Generalization error: {:0.2f}%'.format(100*genError))


#==============================================================================
# figure(1)
# def neval(xval):
#     return np.argmax(clf.predict_proba(xval),1)
# 
# print(X_test.shape,y_test.shape)
# 
# dbplotf(X_test,y_test,neval,'auto')
# show()
# fig.savefig('fig/annClassification.eps', format='eps', dpi=1200)
# fig.clf
#==============================================================================
