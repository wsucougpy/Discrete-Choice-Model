# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 22:46:58 2018

@author: Jikhan Jeong
"""

## 2018 Fall Python Working Group Logistic Regression from Scracth and SKLearn
## Reference
## (Scratch) https://beckernick.github.io/logistic-regression-from-scratch/


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

inctr = pd.read_csv("20181112_EV.csv", header=None, index_col=False, names=['y1','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12'])
feature = inctr.loc[1:155:,"x1":"x12"].astype(int) # training features
label = inctr.loc[1:155:,"y1"].astype(int) # training label

def sigmoid(scores):
    return 1/(1+np.exp(-scores))   # P(Y=1|X)

# Section 4.4.1 of Hastie, Tibsharani, and Friedman’s Elements of Statistical Learning
    
def log_likelihood(feature, label, coefficients):
    ll = np.sum(label*np.dot(feature, coefficients) - np.log(1 + np.exp(np.dot(feature, coefficients))))
    return ll

def logistic_regression(feature, larbel, num_steps, learning_rate, add_intercept = False):
    if add_intercept:
        intercept = np.ones((feature.shape[0],1))
        feature  = np.hstack((intercept, feature))

# numpy.hstack(tup)[source]
# Stack arrays in sequence horizontally (column wise).        
        
    coefficients = np.zeros(feature.shape[1]) # [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    
    for step in range(num_steps):
        scores = np.dot(feature, coefficients)
        predctions = sigmoid(scores)
        
        output_error_signal = label - predctions
        gradient = np.dot(feature.T, output_error_signal) # T is transpose, Gradient Optimization Methods
        coefficients += learning_rate*gradient
        
        if step % 10000 ==0:  #print the values only when i is divisible by 1000:
            print (log_likelihood(feature, label,  coefficients))
            
    return coefficients

# (Learning Rate) https://www.kdnuggets.com/2018/02/understanding-learning-rates-improves-performance-deep-learning.html
#  Learning rate is a hyper-parameter that controls how much we are adjusting the weights of our network with respect the loss gradient. The lower the value, the slower we travel along the downward slope.
    

logistic_regression(feature,label,300000,5e-5)

## From SKlearn (1) Constant with L2 (2) Constant without L2 (3) No constant with L2

## Constant with automatically does L2 regularization (default for SKlearn)
logit_regression = LogisticRegression(random_state=0, solver='liblinear',max_iter=100, fit_intercept=True, multi_class='ovr')
model = logit_regression.fit(feature, label) 

model.coef_
model.intercept_

scores = cross_val_score(logit_regression, feature, label, cv=5)
scores
print("LR with constatn Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



## withou L2
logit_regression1 = LogisticRegression(random_state=0, solver='liblinear',max_iter=100, fit_intercept=True, multi_class='ovr',C = 1e15)
model1 = logit_regression1.fit(feature, label) 
model1.coef_
model1.intercept_

## No Constant
logit_regression2 = LogisticRegression(random_state=0, solver='lbfgs',fit_intercept=False, max_iter=100, multi_class='ovr')
model2 = logit_regression2.fit(feature, label) 
model2.coef_
model2.intercept_


## MLE Approach 
## Reference : https://lectures.quantecon.org/py/mle.html

