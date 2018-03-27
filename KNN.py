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

K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)
CV2 = model_selection.LeaveOneOut()
# Initialize variable
Error_train = np.empty((len(tc),K))
Error_test = np.empty((len(tc),K))

errors1 = np.zeros((N,L))
errors2 = np.zeros((N,L))
i=0
for train_index, test_index in CV.split(stdX):
    print('Crossvalidation fold: {0}/{1}'.format(i+1,K))    
    
    # extract training and test set for current CV fold
    X_train = stdX[train_index,:]
    y_train = classY[train_index]
    X_test = stdX[test_index,:]
    y_test = classY[test_index]

    # Fit classifier and classify the test points (consider 1 to 40 neighbors)
    for l in range(1,L+1):
        knclassifier = KNeighborsClassifier(n_neighbors=l);
        knclassifier.fit(X_train, y_train);
        y_est = knclassifier.predict(X_test);
        errors1[i,l-1] = np.sum(y_est[0]!=y_test[0])

    i+=1
    
print('K-fold CV done')
i=0    
for train_index, test_index in CV2.split(X):
    print('Crossvalidation fold: {0}/{1}'.format(i+1,N))    
    
    # extract training and test set for current CV fold
    X_train = stdX[train_index,:]
    y_train = classY[train_index]
    X_test = stdX[test_index,:]
    y_test = classY[test_index]

    # Fit classifier and classify the test points (consider 1 to 40 neighbors)
    for l in range(1,L+1):
        knclassifier = KNeighborsClassifier(n_neighbors=l);
        knclassifier.fit(X_train, y_train);
        y_est = knclassifier.predict(X_test);
        errors2[i,l-1] = np.sum(y_est[0]!=y_test[0])

    i+=1

# Plot the classification error rate
# K fold
figure()
plot(100*sum(errors1,0)/N)
xlabel('Number of neighbors')
ylabel('Classification error rate (%)')
show()
# Leave one out
figure()
plot(100*sum(errors2,0)/N)
xlabel('Number of neighbors')
ylabel('Classification error rate (%)')
show()

"""
figure(1)
styles = ['.b', '.r', '.g', '.y']
for c in range(C):
    class_mask = (y_train==c)
    plot(X_train[class_mask,0], X_train[class_mask,1], styles[c])
"""

# K-nearest neighbors
k=10

# Distance metric (corresponds to 2nd norm, euclidean distance).
# You can set dist=1 to obtain manhattan distance (cityblock distance).
dist=2

# Fit classifier and classify the test points
knclassifier = KNeighborsClassifier(n_neighbors=k, p=dist);
knclassifier.fit(X_train, y_train);
y_est = knclassifier.predict(stdX);


# Plot the classfication results
styles = ['ob', 'or', 'og', 'oy']
for c in range(C):
    class_mask = (y_est==c)
    plot(stdX[class_mask,0], stdX[class_mask,1], styles[c])
title('Synthetic data classification - KNN');