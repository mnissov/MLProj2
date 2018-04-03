# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 12:33:39 2018

@author: Emma
"""

from matplotlib.pyplot import figure, plot, subplot, title, xlabel, ylabel, show, clim, bar
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import feature_selector_lr, bmplot
import numpy as np
from initData import *
import neurolab as nl
from scipy import stats


# Normalize data
X = stats.zscore(X);
#%%


## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)

# Initialize variables
Features = np.zeros((M,K))
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_fs = np.empty((K,1))
Error_test_fs = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))

k=0
min_error = []

for train_index, test_index in CV.split(X):
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    internal_cross_validation = 10
    
    # Compute squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum()/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum()/y_test.shape[0]

    # Compute squared error with all features selected (no feature selection)
    m = lm.LinearRegression(fit_intercept=True).fit(X_train, y_train)
    Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    # Compute squared error with feature subset selection
    #textout = 'verbose';
    textout = '';
    selected_features, features_record, loss_record = feature_selector_lr(X_train, y_train, internal_cross_validation,display=textout)
    
    min_error.append(min(loss_record))
    
    Features[selected_features,k]=1
    # .. alternatively you could use module sklearn.feature_selection
    if len(selected_features) is 0:
        print('No features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).' )
    else:
        m = lm.LinearRegression(fit_intercept=True).fit(X_train[:,selected_features], y_train)
        Error_train_fs[k] = np.square(y_train-m.predict(X_train[:,selected_features])).sum()/y_train.shape[0]
        Error_test_fs[k] = np.square(y_test-m.predict(X_test[:,selected_features])).sum()/y_test.shape[0]
    
        figure(k)
        subplot(1,2,1)
        plot(range(1,len(loss_record)), loss_record[1:])
        xlabel('Iteration')
        ylabel('Squared error (crossvalidation)')    
        
        subplot(1,3,3)
        bmplot(attributeNames, range(1,features_record.shape[1]), -features_record[:,1:])
        clim(-1.5,0)
        xlabel('Iteration')

    print('Cross validation fold {0}/{1}'.format(k+1,K))
    print('Train indices: {0}'.format(train_index))
    print('Test indices: {0}'.format(test_index))
    print('Features no: {0}\n'.format(selected_features.size))

    k+=1


# Display results
print('\n')
print('Linear regression without feature selection:\n')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Linear regression with feature selection:\n')
print('- Training error: {0}'.format(Error_train_fs.mean()))
print('- Test error:     {0}'.format(Error_test_fs.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_fs.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test_fs.sum())/Error_test_nofeatures.sum()))

figure(k)
subplot(1,3,2)
bmplot(attributeNames, range(1,Features.shape[1]+1), -Features)
clim(-1.5,0)
xlabel('Crossvalidation fold')
ylabel('Attribute')

#%%
# Inspect selected feature coefficients effect on the entire dataset and
# plot the fitted model residual error as function of each attribute to
# inspect for systematic structure in the residual

f=min_error.index(min(min_error)) # cross-validation fold to inspect
ff=Features[:,f].nonzero()[0]

if len(ff) is 0:
    print('\nNo features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).' )
else:
    m = lm.LinearRegression(fit_intercept=True).fit(X[:,ff], y)
    
    y_est= m.predict(X[:,ff])
    residual=y-y_est
    
    figure(k+1, figsize=(12,6))
    title('Residual error vs. Attributes for features selected in cross-validation fold {0}'.format(f))
    for i in range(0,len(ff)):
       subplot(2,np.ceil(len(ff)/2.0),i+1)
       plot(X[:,ff[i]],residual,'.')
       xlabel(attributeNames[ff[i]])
       ylabel('residual error')    
show()

 #%%

# Display scatter plot of the selected features of the best model
figure()
#subplot(2,1,1)
plot(y, y_est, '.')
xlabel('Wages content (true)'); ylabel('Wages content (estimated)');
#subplot(2,1,2)
#hist(residual,40)
show()

#%%
# Fit ordinary least squares regression model

# Display scatter plot of all the features
model = lm.LinearRegression()
model.fit(X,y)

# Predict alcohol content
y_est = model.predict(X)
residual = y_est-y

# Display scatter plot
figure()
#subplot(2,1,1)
plot(y, y_est, '.')
xlabel('Wages content (true)'); ylabel('Wages content (estimated)');
#subplot(2,1,2)
#hist(residual,40)
#The histogram doesn't work !!!!!!!
show()

#%%

# Additional nonlinear attributes
iq_idx = 1
hours_idx = 0
Xiq2 = np.power(X[:,1],2).reshape(-1,1)
Xhours2 = np.power(X[:,0],2).reshape(-1,1)
Xiqhours = np.matrix(np.empty(N)).T
for i in range(0,N):
    Xiqhours[i,0] = X[i,0]*X[i,1]
#Xiqhours = (X[:,0]*X[:,1]).reshape(-1,1)
#Xiqhours = np.mat(Xiqhours)

X0 = np.asarray(np.bmat('X, Xiq2, Xhours2, Xiqhours'))

# Fit ordinary least squares regression model
model = lm.LinearRegression()
model.fit(X0,y)

# Predict alcohol content
y_est = model.predict(X0)
residual = y_est-y

# Display plots
figure(figsize=(12,8))

subplot(2,1,1)
plot(y, y_est, '.g')
xlabel('Wages content (true)'); ylabel('Wages content (estimated)')

subplot(4,3,10)
plot(Xiq2, residual, '.r')
xlabel('iq ^2'); ylabel('Residual')

subplot(4,3,11)
plot(Xhours2, residual, '.r')
xlabel('hours ^2'); ylabel('Residual')

subplot(4,3,12)
plot(Xiqhours, residual, '.r')
xlabel('iq*hours'); ylabel('Residual')

show()

#%%
# Additional nonlinear attributes
educ_idx = 2
expr_idx = 3
Xeduc2 = np.power(X[:,2],2).reshape(-1,1)
Xexpr2 = np.power(X[:,3],2).reshape(-1,1)
Xeducexpr = np.matrix(np.empty(N)).T
for i in range(0,N):
    Xeducexpr[i,0] = X[i,2]*X[i,3]


X2 = np.asarray(np.bmat('X, Xeduc2, Xexpr2, Xeducexpr'))

# Fit ordinary least squares regression model
model = lm.LinearRegression()
model.fit(X2,y)

# Predict alcohol content
y_est = model.predict(X2)
residual = y_est-y

# Display plots
figure(figsize=(12,8))

subplot(2,1,1)
plot(y, y_est, '.g')
xlabel('Wages content (true)'); ylabel('Wages content (estimated)')

subplot(4,3,10)
plot(Xeduc2, residual, '.r')
xlabel('educ ^2'); ylabel('Residual')

subplot(4,3,11)
plot(Xexpr2, residual, '.r')
xlabel('exper ^2'); ylabel('Residual')

subplot(4,3,12)
plot(Xeducexpr, residual, '.r')
xlabel('educ*exper'); ylabel('Residual')

show()

#%%Artificiual Neural Network
C = 2
mean_square_errors =[]
for h in range(1,11):
    print(h)
    # Parameters for neural network classifier
    n_hidden_units = h      # number of hidden units
    n_train = 2             # number of networks trained in each k-fold
    learning_goal = 100     # stop criterion 1 (train mse to be reached)
    max_epochs = 64         # stop criterion 2 (max epochs in training)
    show_error_freq = 5     # frequency of training status updates
    
    # K-fold crossvalidation
    K = 3                   # only three folds to speed up this example
    CV = model_selection.KFold(K,shuffle=True)
    
    # Variable for classification error
    errors = np.zeros(K)*np.nan
    error_hist = np.zeros((max_epochs,K))*np.nan
    bestnet = list()
    k=0
    for train_index, test_index in CV.split(X,y):
        print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
        
        # extract training and test set for current CV fold
        X_train = X[train_index,:]
        y_train = y[train_index]
        X_test = X[test_index,:]
        y_test = y[test_index]
        
        best_train_error = np.inf
        for i in range(n_train):
            print('Training network {0}/{1}...'.format(i+1,n_train))
            # Create randomly initialized network with 2 layers
            ann = nl.net.newff([[-3, 3]]*M, [n_hidden_units, 1], [nl.trans.TanSig(),nl.trans.PureLin()])
            if i==0:
                bestnet.append(ann)
                # train network
            train_error = ann.train(X_train, y_train.reshape(-1,1), goal=learning_goal, epochs=max_epochs, show=show_error_freq)
            if train_error[-1]<best_train_error:
                bestnet[k]=ann
                best_train_error = train_error[-1]
                error_hist[range(len(train_error)),k] = train_error

        print('Best train error: {0}...'.format(best_train_error))
        y_est = bestnet[k].sim(X_test).squeeze()
        errors[k] = np.power(y_est-y_test,2).sum()/y_test.shape[0]
        k+=1
        
    mean_square_errors.append(np.mean(errors))
        
        #break
    


# Print the average least squares error
print('Mean-square error: {0}'.format(np.mean(errors)))

figure(figsize=(6,7));
subplot(2,1,1); bar(range(0,K),errors); title('Mean-square errors');
subplot(2,1,2); plot(error_hist); title('Training error as function of BP iterations');
figure(figsize=(6,7));
subplot(2,1,1); plot(y_est); plot(y_test); title('Last CV-fold: est_y vs. test_y'); 
subplot(2,1,2); plot((y_est-y_test)); title('Last CV-fold: prediction error (est_y-test_y)'); 
show()

x_axis=np.arange(1,11,1)
line1, = plot(x_axis,lc, label="Mean sqared error")
legend = legend(handles=[line1], loc=2)
show()


#% The weights if the network can be extracted via
#bestnet[0].layers[0].np['w'] # Get the weights of the first layer
#bestnet[0].layers[0].np['b'] # Get the bias of the first layer

#%%
# Parameters for neural network classifier
n_hidden_units = 5      # number of hidden units
n_train = 2             # number of networks trained in each k-fold
learning_goal = 100     # stop criterion 1 (train mse to be reached)
max_epochs = 64         # stop criterion 2 (max epochs in training)
show_error_freq = 5     # frequency of training status updates
    
# K-fold crossvalidation
K = 3                   # only three folds to speed up this example
CV = model_selection.KFold(K,shuffle=True)
    
# Variable for classification error
errors = np.zeros(K)*np.nan
error_hist = np.zeros((max_epochs,K))*np.nan
bestnet = list()
k=0
for train_index, test_index in CV.split(X,y):
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
        
    best_train_error = np.inf
    for i in range(n_train):
        print('Training network {0}/{1}...'.format(i+1,n_train))
        # Create randomly initialized network with 2 layers
        ann = nl.net.newff([[-3, 3]]*M, [n_hidden_units, 1], [nl.trans.TanSig(),nl.trans.PureLin()])
        if i==0:
            bestnet.append(ann)
        # train network
        train_error = ann.train(X_train, y_train.reshape(-1,1), goal=learning_goal, epochs=max_epochs, show=show_error_freq)
        if train_error[-1]<best_train_error:
            bestnet[k]=ann
            best_train_error = train_error[-1]
            error_hist[range(len(train_error)),k] = train_error

    print('Best train error: {0}...'.format(best_train_error))
    y_est = bestnet[k].sim(X_test).squeeze()
    errors[k] = np.power(y_est-y_test,2).sum()/y_test.shape[0]
    k+=1
        
   #break 


# Print the average least squares error
print('Mean-square error: {0}'.format(np.mean(errors)))

figure(figsize=(6,7));
subplot(2,1,1); bar(range(0,K),errors); title('Mean-square errors');
subplot(2,1,2); plot(error_hist); title('Training error as function of BP iterations');
figure(figsize=(6,7));
subplot(2,1,1); plot(y_est); plot(y_test); title('Last CV-fold: est_y vs. test_y'); 
subplot(2,1,2); plot((y_est-y_test)); title('Last CV-fold: prediction error (est_y-test_y)'); 
show()