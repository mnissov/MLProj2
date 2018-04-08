#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 17:23:40 2018

@author: johandybkjaer-knudsen
"""

from matplotlib.pyplot import figure, boxplot, plot, xlabel, ylabel, show
import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm #Can this be deleted?
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection, tree
from scipy import stats

from initData import classX, classY
from ANN import annBest
from KNN import knnBest
from Tree import treeBest1

#%%
# Defines data
X = classX
y = classY

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)
#CV = model_selection.StratifiedKFold(n_splits=K)


# Initialize variables
Error_ANN = np.empty((K,1))
Error_KNN = np.empty((K,1))
Error_Tree = np.empty((K,1))
n_tested=0

#%%
k=0
# Note: Cannot remove train_index and X_train from this loop :/ ??
for train_index, test_index in CV.split(X,y):
    print('CV-fold {0} of {1}'.format(k+1,K))

    # extract training and test set for current CV fold
    X_train = X[train_index,:] # Should have been deleted but cannot
    y_train = y[train_index] # Should have been deleted but cannot
    X_test = X[test_index,:]
    y_test = y[test_index]

    # Fit and evaluate KNN
    model1 = knnBest
    y_KNN = model1.predict(X_test)
    Error_KNN[k] = 100*(y_KNN != y_test).sum().astype(float)/len(y_test)
    
    # Fit and evaluate ANN
    model2 = annBest
    y_ANN = model2.predict(X_test)
    Error_ANN[k] = 100*(y_ANN != y_test).sum().astype(float)/len(y_test)
    
    # Fit and evaluate Tree
    model3 = treeBest1
    y_Tree = model3.predict(X_test)
    Error_Tree[k] = 100*(y_Tree != y_test).sum().astype(float)/len(y_test)
    

    k+=1

print("Error of ANN:")
print(Error_ANN)
print("")
print("Error of KNN:")
print(Error_KNN)
print("Error of Tree:")
print(Error_Tree)

#%%
# Statistically compare the models by computing credibility intervals
z = (Error_ANN-Error_KNN)
zb = z.mean()
nu = K-1
sig =  (z-zb).std()  / np.sqrt(K-1)
# Sets the interval to 95%
alpha = 0.05

zL = zb + sig * stats.t.ppf(alpha/2, nu);
zH = zb + sig * stats.t.ppf(1-alpha/2, nu);

if zL <= 0 and zH >= 0 :
    print('Classifiers are not significantly different')        
else:
    print('Classifiers are significantly different.')

#%%
# Boxplot to compare classifier error distributions
figure()
boxplot(np.concatenate((Error_ANN, Error_KNN, Error_Tree),axis=1))
xlabel('ANN   vs.   KNN   vs.   Tree')
ylabel('Cross-validation error [%]')

show()
