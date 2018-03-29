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
## Change tcEnd to get more max depth; max_depth=tcEnd-1
tcStart = 2
tcEnd = 21
tc = np.arange(tcStart, tcEnd, 1)

K1 = 5
K2 = 10
CV1 = model_selection.KFold(n_splits=K1,shuffle=True)
CV2 = model_selection.KFold(n_splits=K2,shuffle=True)
CV3 = model_selection.LeaveOneOut()

# Initialize variable
valError = np.zeros((K2,tcEnd-tcStart))
trainError = np.zeros((K2,tcEnd-tcStart))
sGenError = np.zeros((1,tcEnd-tcStart))
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
        for s, t in enumerate(tc):
            # Fit decision tree classifier, Gini split criterion, different pruning levels
            dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
            dtc = dtc.fit(X_train,y_train)
            y_est = dtc.predict(X_val)
            trainError[j,s] = np.mean(dtc.predict(X_train)!=y_train)
            valError[j,s] = np.mean(y_est!=y_val)
        j+=1
    sGenError = (len(X_val)/len(X_par))*np.sum(valError,axis=0)   
    
    bestModel = tree.DecisionTreeClassifier(criterion='gini', max_depth = np.argmin(sGenError)+2)
    bestModel.fit(X_par,y_par)
    
    testError[i] = np.mean(bestModel.predict(X_test)!=y_test)
    
    print('\n\tBest model: {:0.0f} max depth and {:1.2f}% biased test error and {:2.2f}% unbiased'.format(bestModel.max_depth,100*np.mean(valError,axis=0)[bestModel.max_depth-2],100*testError[i]))
    
    figure()
    plot(tc, 100*np.mean(trainError,axis=0))
    plot(tc, 100*np.mean(valError,axis=0))
    xlabel('Model complexity (max tree depth)')
    ylabel('Error (CV K=)'.format(K1))
    legend(['Error_train','Error_test'])    
    show()    
    
    if(testError[i]<previous):
        absBest = bestModel
        absBestPred = absBest.predict(X_test)
        previous = testError[i]
    #genE[i] = sum(valError[i],0)/len(parX)
    i+=1
genError = (len(X_test)/N)*np.sum(testError,axis=0)
    
print('K-fold CV done')
print('The best model has {:0.0f} depth with {:1.2f}% unbiased test error'.format(absBest.max_depth,100*previous))
print('Generalized error: {:0.2f}%'.format(100*genError))

    
figure()
boxplot(valError)
xlabel('Model complexity (max tree depth)')
ylabel('Test error across CV folds, K={0})'.format(K))
show()


out = tree.export_graphviz(absBest, out_file='BestGini.gvz', feature_names=attributeNames)

graphviz.render('dot','png','BestGini.gvz')