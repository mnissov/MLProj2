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

K = 10
iStart = 1
iEnd = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)

error = np.zeros([K,iEnd-1])
errors = np.ones(K)
errorData = np.ones([K,3])
bestnet = list()
train_error = list()
k=0

for train_index, test_index in CV.split(stdX,classY):
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
    
    # extract training and test set for current CV fold
    X_train = stdX[train_index,:]
    y_train = classY[train_index]
    X_test = stdX[test_index,:]
    y_test = classY[test_index]
    
    best_train_error = 1e100
    for i in range(iStart,iEnd):
        ## ANN Classifier, i.e. MLP with one hidden layer
        clf = nn.MLPClassifier(solver='lbfgs',alpha=1e-4,
                               hidden_layer_sizes=(i,), random_state=1)
        if i==1:
            bestnet.append(clf)
        
        clf.fit(X_train,y_train)
        y_est = clf.predict(X_test)
        
        """
        train_error.append(100*np.sum(y_est[0]!=y_test[0])/len(y_test))
        if train_error[-1]<best_train_error:
            bestnet[k]=clf
            best_train_error = 100*np.sum(y_est!=y_test)/len(y_test)
            #error_hist[range(len(train_error)),k] = train_error
        """
        error[k,i-1] = np.mean(y_est!=y_test)
        
        # finding lowest error
        """ 
        if (np.sum(clf.predict(X_test)!=y_test)/len(y_test))<=errors[k]:
            errors[k] = np.sum(clf.predict(X_test)!=y_test)/len(y_test)
            errorData[k,0] = i
            errorData[k,1] = np.sum(clf.predict(X_test)!=y_test)
            errorData[k,2] = len(y_test)
            print(i,'Number of miss-classifications for ANN:\n\t {0} out of {1}'.format(np.sum(clf.predict(X_test)!=y_test),len(y_test)))
        """        
        #print(i,'Number of miss-classifications for ANN:\n\t {0} out of {1}'.format(np.sum(clf.predict(X_test)!=y_test),len(y_test)))
    """
    print('Best train error: {0}...'.format(best_train_error))
    y_est = bestnet[k].predict(X_test)
    y_est = (y_est>.5).astype(int)
    error[k] = (y_est!=y_test).sum().astype(float)/y_test.shape[0]
    """
    k+=1
 
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