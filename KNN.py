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
L=40

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
    
    figure()
    plot(100*np.mean(valError,axis=0))
    xlabel('Number of neighbors')
    ylabel('Classification error rate (%)')
    show()    
    
    bestModel = KNeighborsClassifier(n_neighbors = np.argmin(sGenError)+1)
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


# Plot the classification error rate
# K fold

"""
# Leave one out
figure()
plot(100*sum(errors2,0)/N)
xlabel('Number of neighbors')
ylabel('Classification error rate (%)')
show()
"""
"""
figure(1)
styles = ['.b', '.r', '.g', '.y']
for c in range(C):
    class_mask = (y_train==c)
    plot(X_train[class_mask,0], X_train[class_mask,1], styles[c])
"""

y_est = absBest.predict(stdX);

# Plot the classfication results
styles = ['ob', 'or', 'og', 'oy']
for c in range(C):
    class_mask = (y_est==c)
    plot(stdX[class_mask,0], stdX[class_mask,1], styles[c])
title('Synthetic data classification - KNN');