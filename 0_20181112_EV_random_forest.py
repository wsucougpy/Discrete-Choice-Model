# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 12:04:40 2018

@author: Jikhan Jeong
"""



## 2018 Fall Python Working Group Random Forest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# from scipy import stats
# stats.describe(a), when we want to know summary statistics in a arrary

# from numpy.linalg import norm # vector norm 

inctr = pd.read_csv("20181112_EV.csv", header=None, index_col=False, names=['y1','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12'])

feature = inctr.loc[1:155:,"x1":"x12"] # training features
label = inctr.loc[1:155:,"y1"] # training label

# X_train, X_test, y_train, y_test = train_test_split(
#        feature, label, test_size= 0.4, random_state=0)


random_forest = RandomForestClassifier(n_estimators=30, max_depth=2)
scores = cross_val_score(random_forest, feature, label, cv=5)
scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

## Q7 Bayesian Optimiation



## (1)__Bagging Hyper Parameter optimization
##__(1_1) Object Function to minmize
##__Accuracy is maximum problem so make it as 1-Accuracy to make it min problem

def objective_function(params):
    n_estimators, max_depth = params
    random_forest = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,n_jobs=-1)
    scores = cross_val_score(random_forest, feature, label, cv=5)
    m_scores = scores.mean()
    loss =1- m_scores
    return loss 

##__(1_2) Domain Space for depth and size of tree

space = [  
        hp.choice("n_estimators",range(1,101)),
        hp.quniform("max_depth",1,5,1)]

bag_trials = Trials()

##__(1_4) Bayesian Optimization

best = fmin(objective_function, 
            space, algo= tpe.suggest, trials= bag_trials, #quniform discrete uniform
            max_evals = 50)

best['n_estimators'] = (best.get('n_estimators')+1)*10  ### outcome is not value but position so change it

print(best)


bag_results_depths = pd.DataFrame({'loss': [max_depth['loss'] for max_depth in bag_trials.results],
                            'iteration': bag_trials.idxs_vals[0]['max_depth'], #[0] means Bayesian iteration
                            'max_depth': bag_trials.idxs_vals[1]['max_depth'],})

bag_results_depths.head()

bag_results_size = pd.DataFrame({'loss': [n_estimators['loss'] for n_estimators in bag_trials.results],
                            'iteration': bag_trials.idxs_vals[0]['n_estimators'], #[0] means Bayesian iteration
                             'max_size':  bag_trials.idxs_vals[1]['n_estimators']}) #[1] means depth values

bag_results_size    ## outcome is not value but position so change it

for i in range(0, 10):
    bag_results_size["max_size"] = bag_results_size["max_size"].replace(i,(i+1)*10)
    
bag_results_size.head()

### Final Results 

bag_results_total = pd.merge(bag_results_depths, bag_results_size) 

print(bag_results_total)
### Fianl Graph
line = bag_results_total.plot.line(x='iteration',y='loss')
