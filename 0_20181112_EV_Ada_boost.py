# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 12:04:40 2018

@author: Jikhan Jeong
"""
## 2018 Fall Python Working Group AdaBoosting

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier


# from scipy import stats
# stats.describe(a), when we want to know summary statistics in a arrary

# from numpy.linalg import norm # vector norm 

inctr = pd.read_csv("20181112_EV.csv", header=None, index_col=False, names=['y1','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12'])

feature = inctr.loc[1:155:,"x1":"x12"] # training features
label = inctr.loc[1:155:,"y1"] # training label

# X_train, X_test, y_train, y_test = train_test_split(
#        feature, label, test_size= 0.4, random_state=0)

ada_boosting = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth = 1), n_estimators= 12, algorithm="SAMME.R", learning_rate=1.0)
scores = cross_val_score(ada_boosting, feature, label, cv=5)
scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

## Q7 Bayesian Optimiation

def objective_function(params):
    n_estimators, max_depth = params    
    ada_boosting = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth = max_depth), n_estimators= n_estimators, algorithm="SAMME.R", learning_rate=1.0)
    scores = cross_val_score(ada_boosting, feature, label, cv=5)
    m_scores = scores.mean()
    loss = 1-m_scores
    return loss

##__(1_2) Domain Space for depth and size of tree

space = [  
        hp.choice("n_estimators",range(1,101)),
        hp.quniform("max_depth",1,5,1)]

boost_trials = Trials()

##__(1_4) Bayesian Optimization

best = fmin(objective_function, 
            space, algo= tpe.suggest, trials= boost_trials, #quniform discrete uniform
            max_evals = 50)

best['n_estimators'] = best.get('n_estimators')  ### outcome is not value but position so change it


print(best)



##__(1_5) Results 

boost_results_depths = pd.DataFrame({'loss': [max_depth['loss'] for max_depth in boost_trials.results],
                            'iteration': boost_trials.idxs_vals[0]['max_depth'], #[0] means Bayesian iteration
                            'max_depth': boost_trials.idxs_vals[1]['max_depth'],})


    #[1] means depth values
        
boost_results_depths.head()

boost_results_size = pd.DataFrame({'loss': [n_estimators['loss'] for n_estimators in boost_trials.results],
                            'iteration': boost_trials.idxs_vals[0]['n_estimators'], #[0] means Bayesian iteration
                             'max_size':  boost_trials.idxs_vals[1]['n_estimators']}) #[1] means depth values

boost_results_size    ## outcome is not value but position so change it

for i in range(0, 10):
    boost_results_size["max_size"] = boost_results_size["max_size"].replace(i,(i+1)*10)
    
boost_results_size.head()

### Final Results 

boost_results_total = pd.merge(boost_results_depths, boost_results_size) 
print(boost_results_total)

### Fianl Graph
line = boost_results_total.plot.line(x='iteration',y='loss')
