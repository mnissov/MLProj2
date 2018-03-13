# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 11:42:19 2018

@author: Emma
"""

import numpy as np
import xlrd
import pandas as pd
import math

from matplotlib.pyplot import figure, legend, subplot, plot, hist, title, imshow, yticks, cm, xlabel, ylabel, show, grid, boxplot
from scipy.linalg import svd
from scipy.io import loadmat
import sklearn.linear_model as lm
# Load xls sheet with data
#dataset = xlrd.open_workbook('wage2.xls').sheet_by_index(0)
#data = pd.get_dummies(dataset)


df = pd.read_excel('modified.xls', header = None)
doc = xlrd.open_workbook('modified.xls').sheet_by_index(0)

attributeNames = doc.row_values(0, 1, 8)
n = len(df.index)
df.reset_index()
df.reindex(index=range(0,n))

df.dropna(inplace=True)
dfMatrix = df.as_matrix()

y = dfMatrix[1:,0]
yMatrix = np.mat(y)

X = np.mat(np.empty((n-1,7)))

for i, col_id in enumerate(range(1,8)):
    X[:,i] = np.matrix(doc.col_values(col_id, 1, n)).T

classX = np.asarray(X)

#N = len(y)
#M = len(attributeNames)

N, M = X.shape

