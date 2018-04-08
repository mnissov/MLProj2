#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 17:23:40 2018

@author: johandybkjaer-knudsen
"""

from matplotlib.pyplot import figure, plot, subplot, title, xlabel, ylabel, show, clim, bar
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import feature_selector_lr, bmplot
import numpy as np
import neurolab as nl
from scipy import stats

from initData import X, y, classX, classY

import pickle

#%%
# Open models
bestANN = pickle.load(open('bestModelRegrANN.sav', 'rb'))

bestLinear = pickle.load(open('bestModelRegrLinear.sav', 'rb'))

bestFeatures = pickle.load(open('bestFeatures.sav', 'rb'))

#%%
# Defines data
X = stats.zscore(X);
y = y

# The classified data
classX = classX
classY = classY


## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)
#CV = model_selection.StratifiedKFold(n_splits=K)


# Initialize variables
Error_ANN = np.empty((K,1))
Error_Linear = np.empty((K,1))
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
    y_test = classY[test_index]
    
# =============================================================================
    # Fit and evaluate Linear
    model1 = bestLinear
    y_Linear_est = model1.predict(X_test[:,bestFeatures])
    
    # Classifying the predicted score
    n = len(y_Linear_est)
    y_Linear = np.zeros(n)
    
    for i in range(0,n):
        if y_Linear_est[i] <= np.percentile(y,25):
            y_Linear[i] = 0
        elif y_Linear_est[i] <= np.percentile(y,50):
            y_Linear[i] = 1
        elif y_Linear_est[i] <= np.percentile(y,75):
            y_Linear[i] = 2
        else: 
            y_Linear[i] = 3
    
    Error_Linear[k] = 100*(y_Linear != y_test).sum().astype(float)/len(y_test)
# =============================================================================  
    # Fit and evaluate ANN
    model2 = bestANN
    y_ANN_est = model2.sim(X_test).squeeze()
    
    # Classifying the predicted score
    n = len(y_ANN_est)
    y_ANN = np.zeros(n)
    
    for i in range(0,n):
        if y_ANN_est[i] <= np.percentile(y,25):
            y_ANN[i] = 0
        elif y_ANN_est[i] <= np.percentile(y,50):
            y_ANN[i] = 1
        elif y_ANN_est[i] <= np.percentile(y,75):
            y_ANN[i] = 2
        else: 
            y_ANN[i] = 3
    
    Error_ANN[k] = 100*(y_ANN != y_test).sum().astype(float)/len(y_test)
    
    
# =============================================================================
    k+=1

print("Error of Linear:")
print(Error_Linear)
print("")
print("Error of ANN:")
print(Error_ANN)


#%%
# Statistically compare the models by computing credibility intervals
z = (Error_Linear-Error_ANN)
zb = z.mean()
nu = K-1
sig =  (z-zb).std()  / np.sqrt(K-1)
# Sets the interval to 95%
alpha = 0.05

zL = zb + sig * stats.t.ppf(alpha/2, nu);
zH = zb + sig * stats.t.ppf(1-alpha/2, nu);
# =============================================================================
# print('Credibility Interval (Lower): {:0.2f}%'.format(zL))
# print('Credibility Interval (Higher): {:0.2f}%'.format(zH))
# =============================================================================

if zL <= 0 and zH >= 0 :
    print('Classifiers are not significantly different')        
else:
    print('Classifiers are significantly different.')

#%%
# Boxplot to compare classifier error distributions
figure()
boxplot(np.concatenate((Error_Linear, Error_ANN),axis=1))
xlabel('Linear   vs.   ANN')
ylabel('Cross-validation error [%]')

show()
