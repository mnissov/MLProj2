# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 16:09:34 2018

@author: morten
"""
from matplotlib.pyplot import figure, plot, xlabel, ylabel, show
import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection

from initData import *

# Maximum number of neighbors
## Change L to look over larger area
L=40
tcStart = 1
tcEnd = L+1
tc = np.arange(tcStart, tcEnd, 1)

K1 = 5
K2 = 10
CV1 = model_selection.KFold(n_splits=K1,shuffle=True)
CV2 = model_selection.KFold(n_splits=K2,shuffle=True)
CV3 = model_selection.LeaveOneOut()

# Initialize variable
valError = np.zeros((K2,L))
sGenError = np.zeros((1,L))
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
        for s in range(1,L+1):
            knclassifier = KNeighborsClassifier(n_neighbors=s)
            knclassifier.fit(X_train, y_train)
            y_est = knclassifier.predict(X_val)
            
            valError[j,s-1] = np.mean(y_est!=y_val)
        j+=1
    sGenError = (len(X_val)/len(X_par))*np.sum(valError,axis=0)
    
    bestModelKNN = KNeighborsClassifier(n_neighbors = np.argmin(sGenError)+1)
    bestModelKNN.fit(X_par,y_par)
    
    testError[i] = np.mean(bestModelKNN.predict(X_test)!=y_test)
    
    print('\n\tBest model: {:0.0f} neighbors and {:1.2f}% biased test error and {:2.2f}% unbiased'.format(bestModelKNN.n_neighbors,100*np.mean(valError,axis=0)[bestModelKNN.n_neighbors-1],100*testError[i]))
    
    
    
    fig=figure()
    plot(tc,100*np.mean(valError,axis=0))
    xlabel('Number of neighbors')
    ylabel('Classification error rate (%)')
    show()
    fig.savefig('fig/knnErrorFold{0}.eps'.format(i+1), format='eps', dpi=1200)   
    fig.clf
    
    if(testError[i]<previous):
        knnBest = bestModelKNN
        absBestPred = knnBest.predict(X_test)
        previous = testError[i]
    #genE[i] = sum(valError[i],0)/len(parX)
    i+=1
genError = (len(X_test)/N)*np.sum(testError,axis=0)
    
print('K-fold CV done')
print('The best model has {:0.0f} neighbors with {:1.2f}% unbiased test error'.format(knnBest.n_neighbors,100*previous))
print('Generalization error: {:0.2f}%'.format(100*genError))


y_est = knnBest.predict(stdX);

# Plot the classfication results
fig = figure()
styles = ['ob', 'or', 'og', 'oy']
for c in range(C):
    class_mask = (y_est==c)
    plot(stdX[class_mask,0], stdX[class_mask,1], styles[c])
title('Synthetic data classification - KNN')
legend(classNames)
fig.savefig('fig/knnClassification.eps', format='eps', dpi=1200)
fig.clf