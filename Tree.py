# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 13:18:55 2018

@author: morten
"""

import numpy as np
from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, show, boxplot, subplot, subplots
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
valError1 = np.zeros((K2,tcEnd-tcStart))
valError2 = np.zeros((K2,tcEnd-tcStart))
trainError1 = np.zeros((K2,tcEnd-tcStart))
trainError2 = np.zeros((K2,tcEnd-tcStart))
sGenError1 = np.zeros((1,tcEnd-tcStart))
sGenError2 = np.zeros((1,tcEnd-tcStart))
testError1 = np.zeros(K1)
testError2 = np.zeros(K1)
previous1=1e100
previous2=1e100

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
            dtc1 = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
            dtc2 = tree.DecisionTreeClassifier(criterion='entropy', max_depth=t)
            dtc1 = dtc1.fit(X_train,y_train)
            dtc2 = dtc2.fit(X_train,y_train)
            
            y_est1 = dtc1.predict(X_val)
            y_est2 = dtc2.predict(X_val)
            
            trainError1[j,s] = np.mean(dtc1.predict(X_train)!=y_train)
            trainError2[j,s] = np.mean(dtc2.predict(X_train)!=y_train)
            valError1[j,s] = np.mean(y_est1!=y_val)
            valError2[j,s] = np.mean(y_est2!=y_val)
        j+=1
    sGenError1 = (len(X_val)/len(X_par))*np.sum(valError1,axis=0)   
    sGenError2 = (len(X_val)/len(X_par))*np.sum(valError2,axis=0)   
    
    bestModel1 = tree.DecisionTreeClassifier(criterion='gini', max_depth = np.argmin(sGenError1)+2)
    bestModel2 = tree.DecisionTreeClassifier(criterion='entropy', max_depth = np.argmin(sGenError2)+2)
    
    bestModel1.fit(X_par,y_par)
    bestModel2.fit(X_par,y_par)
    
    testError1[i] = np.mean(bestModel1.predict(X_test)!=y_test)
    testError2[i] = np.mean(bestModel2.predict(X_test)!=y_test)
    
    print('\n\tBest model: {:0.0f} max depth and {:1.2f}% biased test error and {:2.2f}% unbiased'.format(bestModel1.max_depth,100*np.mean(valError1,axis=0)[bestModel1.max_depth-2],100*testError1[i]))
    print('\tBest model: {:0.0f} max depth and {:1.2f}% biased test error and {:2.2f}% unbiased'.format(bestModel2.max_depth,100*np.mean(valError2,axis=0)[bestModel2.max_depth-2],100*testError2[i]))
    
    fig, (plot1,plot2)=subplots(1,2,sharey=True,sharex=True)
    
    plot1.plot(tc, 100*np.mean(trainError1,axis=0))
    plot1.plot(tc, 100*np.mean(valError1,axis=0))
    plot1.set_ylabel('Error (CV K=)'.format(K1))
    plot1.set_title('Gini')
    
    plot2.plot(tc, 100*np.mean(trainError2,axis=0))
    plot2.plot(tc, 100*np.mean(valError2,axis=0))
    plot2.set_title('Entropy')
    plot2.legend(['Error_train','Error_test'], loc=3)  
    fig.text(0.5, 0.02, 'Max Tree Depth', ha='center')
    show()
    fig.savefig('fig/treeErrorFold{0}.eps'.format(i+1), format='eps', dpi=1200)   
    fig.clf
    
    if(testError1[i]<previous1):
        treeBest1 = bestModel1
        absBestPred1 = treeBest1.predict(X_test)
        previous1 = testError1[i]
        
    if(testError2[i]<previous2):
        treeBest2 = bestModel2
        absBestPred2 = treeBest2.predict(X_test)
        previous2 = testError2[i]
    #genE[i] = sum(valError[i],0)/len(parX)
    i+=1
genError1 = (len(X_test)/N)*np.sum(testError1,axis=0)
genError2 = (len(X_test)/N)*np.sum(testError2,axis=0)
    
print('K-fold CV done')
print('The best gini model has {:0.0f} depth with {:1.2f}% unbiased test error'.format(treeBest1.max_depth,100*previous1))
print('The best entropy model has {:0.0f} depth with {:1.2f}% unbiased test error'.format(treeBest2.max_depth,100*previous2))
print('Generalization error [gini:entropy]: [{:0.2f}:{:1.2f}]%'.format(100*genError1,100*genError2))

    
fig = figure()
boxplot(valError1)
xlabel('Model complexity (max tree depth)')
ylabel('Test error across CV folds, K={0})'.format(K1))
show()
fig.savefig('fig/treeGini.eps', format='eps', dpi=1200)   
fig.clf

fig=figure()
xlabel('Model complexity (max tree depth)')
ylabel('Test error across CV folds, K={0})'.format(K1))
boxplot(valError1)
show()
fig.savefig('fig/treeEntropy.eps', format='eps', dpi=1200)   
fig.clf

out1 = tree.export_graphviz(treeBest1, out_file='fig/treeBestGini.gvz', feature_names=attributeNames)
out2 = tree.export_graphviz(treeBest2, out_file='fig/treeBestEntropy.gvz', feature_names=attributeNames)

graphviz.render('dot','png','fig/treeBestGini.gvz')
graphviz.render('dot','png','fig/treeBestEntropy.gvz')