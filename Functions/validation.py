import pandas as pd
import numpy as np

from datetime import date, timedelta, datetime
import time
import math

from sklearn import preprocessing
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

from itertools import product

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll.base import scope
from time import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

import json
from numpyencoder import NumpyEncoder

from dynamic_input_data import *  
from HCOT import *
from CAT import *
from baseline import *
import importlib

import warnings
import random
warnings.filterwarnings("ignore")

def HCOT_optimisation():
    '''
    Optimisation of a single hierarchy with a pre-defined set of possible combinations to try. Output is a dictionary with f1 scores and optimal hyperparemtes for each combination.
    '''

    df, X_col, Y_col, historic_variable, DATE = initialiseData():

    # df_ contains a sample of training + validation data
    random.seed(100)
    df_ = df.iloc[:int(0.8*len(df))].sample(n=1100000, replace=False, random_state=1).sort_values(by = 'orderDate').reset_index(drop = True)

    Tree = ClassHierarchy('ORDERS')
    Tree.add_node(['UNKNOWN','KNOWN'], 'ORDERS')
    Tree.add_node(['HAPPY','UNHAPPY'], 'KNOWN')
    Tree.add_node(['MILDLY UNHAPPY','MEDIUM UNHAPPY','HEAVILY UNHAPPY'], 'UNHAPPY')

    combinations = [('LR','RF','RF'),('RF','LR','LR'),('LR','LR','LR'),('RF','RF','RF'),
                    ('LR','LR','RF'),('LR','RF','LR'),('RF','RF','LR'),('RF','LR','RF')]

    output = {}

    for DAY in range(0,11):
        
        X_preBurn, y_preBurn = dataX(df_, DATE, X_col, Y_col, historic_variable, DAY)
        index = range(0, X_preBurn.shape[0])

        X_train_val = X_preBurn.iloc[int(0.1*len(X_preBurn)):]
        y_train_val = y_preBurn.iloc[int(0.1*len(y_preBurn)):]

        X_train = X_train_val.iloc[0:int(0.8*len(X_train_val))]
        X_val = X_train_val.iloc[int(0.8*len(X_train_val)):]

        y_train = y_train_val.iloc[0:int(0.8*len(y_train_val))]
        y_val = y_train_val.iloc[int(0.8*len(y_train_val)):]

        output[DAY] = {}

        for combination in combinations:

            best_param, f1, accuracy = hyperopt(get_hyperspace(combination), X_train, y_train, X_val, y_val, 20)

            output[DAY][str(combination)] = (DAY, best_param, f1, accuracy)

            #with open('/path.../validation.json', 'w') as f:
            with open('/Users/thoma/Documents/seminar_data/validation.json', 'w') as f:
                json.dump(output, f, cls = NumpyEncoder)

    return output

def get_hyperspace(combination):
    # create hyperspace for optimisation (currently supports DT, RF, NN and LR).
    param_hyperopt = {}
    
    for node, clf in enumerate(combination):
        
        if clf == 'DT':
            hyper = {'DT_criterion_'+str(node)   : hp.choice('DT_criterion_'+str(node) ,['gini','entropy']),
                     'DT_max_depth_'+str(node)   : scope.int(hp.quniform('DT_max_depth_'+str(node), 5, 15, 1))}
        elif clf == 'RF':
            hyper = {'RF_max_depth_'   +str(node) : scope.int(hp.quniform('RF_max_depth_'+str(node), 5, 15, 1)),
                     'RF_n_estimators_'+str(node) : scope.int(hp.quniform('RF_n_estimators_'+str(node), 10, 50, 5))}
        elif clf == 'NN':
            hyper = {'NN_dropout_'+str(node)  : hp.uniform('NN_dropout_'+str(node), 0, 0.5),
                     'NN_nodes_'  +str(node)  : scope.int(hp.quniform('NN_nodes_'+str(node), 5, 50, 5)),
                     'NN_layers_' +str(node)  : scope.int(hp.quniform('NN_layers_'+str(node), 1, 2, 1))}
        elif clf == 'LR':
            hyper = {'LR_penalty_' + str(node) : hp.choice('LR_penalty_' + str(node), ['l1','l2'])}
            
        param_hyperopt = {**param_hyperopt, **hyper}
        
    return param_hyperopt

def clf_hypers(params):
    #  function which distributes the hyperparameters over the 3 specific parent nodes in the hierarchy.
    clf = {}
    
    for ix, node in enumerate(['ORDERS','KNOWN','UNHAPPY']):

        node_hypers = [x for x in list(params.keys()) if x[-1] == str(ix)]

        if combination[ix] == 'DT':
            clf[node] = DecisionTreeClassifier(random_state=0, class_weight='balanced', max_depth = params[node_hypers[1]], criterion = params[node_hypers[0]])
        elif combination[ix] == 'RF':
            clf[node] = RandomForestClassifier(random_state=0, class_weight='balanced', max_depth = params[node_hypers[0]], n_estimators = params[node_hypers[1]])
        elif combination[ix] == 'NN':
            if ix == 2:
                output = 3
            else:
                output = 1
            clf[node] = KerasClassifier(functions.neuralNetwork, output = output, nodes = params[node_hypers[1]], layers = params[node_hypers[2]], droprate = params[node_hypers[0]], epochs = 15, verbose = 0)
        elif combination[ix] == 'LR':
            clf[node] = LogisticRegression(penalty = params[node_hypers[0]], class_weight = 'balanced', solver = 'liblinear')
            
    return clf

def objective_function(params):
    # objective for optimising daily hierarchical classifier combination in HCOT.
    HC = HierarchicalClassifier(Tree)
    HC.fit_classifiers(clf_hypers(params))
    
    HC = HC.fit(X_train,y_train)
    pred = HC.predict(X_val)
    
    score = f1_score_ancestors(Tree, y_val['detailedMatchClassification'], pred, beta=1)
    accuracy = metrics.accuracy_score(y_val, pred)
    
    return {'loss': -score, 'status': STATUS_OK, 'accuracy': accuracy}

def hyperopt(param_space, X_train, y_train, X_val, y_val, num_eval):
    # actual optimisation function for a single hierarchy with pre-set classifier combination.
    trials = Trials()

    best_param = fmin(objective_function, 
                      param_space, 
                      algo = tpe.suggest, 
                      max_evals = num_eval, 
                      trials = trials,
                      rstate = np.random.RandomState(1))
    
    loss = [x['result']['loss'] for x in trials.trials]
    index_min_loss = loss.index(min(loss))
    accuracy_scores = [x['result']['accuracy'] for x in trials.trials]
    
    f1 = min(loss)*-1
    accuracy = accuracy_scores[index_min_loss]
    
    return best_param, f1, accuracy

def flat_HCOT_optimisation():
    '''
    Optimisation of single 'one-level hierarchy' of the flat HCOT basline (supporting LR + RF). Output is a dictionary with f1 scores and optimal hyperparemtes for each combination.
    For LR only 2 trailes are needed as there are only 2 possible hyperparemters. 
    '''
    output = {}

    combinations = ['RF','LR']

    df, X_col, Y_col, historic_variable, DATE = initialiseData():

    # df_ contains a sample of training + validation data
    random.seed(100)
    df_ = df.iloc[:int(0.8*len(df))].sample(n=1100000, replace=False, random_state=1).sort_values(by = 'orderDate').reset_index(drop = True)

    for DAY in range(0,11):
        
        X_preBurn, y_preBurn = dataX(df_, DATE, X_col, Y_col, historic_variable, DAY)
        index = range(0, X_preBurn.shape[0])

        X_train_val = X_preBurn.iloc[int(0.1*len(X_preBurn)):]
        y_train_val = y_preBurn.iloc[int(0.1*len(y_preBurn)):]

        X_train = X_train_val.iloc[0:int(0.8*len(X_train_val))]
        X_val = X_train_val.iloc[int(0.8*len(X_train_val)):]

        y_train = y_train_val.iloc[0:int(0.8*len(y_train_val))]
        y_val = y_train_val.iloc[int(0.8*len(y_train_val)):]

        output[DAY] = {}

        for combination in combinations:
            
            if combination == 'RF':
                n_trials = 20
            elif combination == 'LR':
                n_trials = 2

            best_param, f1, accuracy = flat_hyperopt(flat_get_hyperspace(combination), X_train, y_train, X_val, y_val, n_trials)

            output[DAY][str(combination)] = (DAY, best_param, f1, accuracy)
            print(output)

            #with open('path.../validation.json', 'w') as f:
            with open('/Users/thoma/Documents/seminar_data/validation.json', 'w') as f:
                json.dump(output, f, cls = NumpyEncoder)

    return output

def flat_get_hyperspace(combination):
    # reate hyperspace for optimisation.
    param_hyperopt = {}

    if combination == 'DT':
        hyper = {'DT_criterion'   : hp.choice('DT_criterion',['gini','entropy']),
                 'DT_max_depth'   : scope.int(hp.quniform('DT_max_depth', 5, 15, 1))}
    elif combination == 'RF':
        hyper = {'RF_max_depth'    : scope.int(hp.quniform('RF_max_depth', 5, 15, 1)),
                 'RF_n_estimators' : scope.int(hp.quniform('RF_n_estimators', 10, 50, 5))}
    elif combination == 'NN':
        hyper = {'NN_dropout'  : hp.uniform('NN_dropout', 0, 0.5),
                 'NN_nodes'    : scope.int(hp.quniform('NN_nodes', 5, 50, 5)),
                 'NN_layers'   : scope.int(hp.quniform('NN_layers', 1, 2, 1))}
    elif combination == 'LR':
        hyper = {'LR_penalty' : hp.choice('LR_penalty', ['l1','l2'])}

    param_hyperopt = {**param_hyperopt, **hyper}
        
    return param_hyperopt

def flat_objective_function(params):
    #  objective for optimising either LR or RF.
    if combination == 'RF':
        clf = RandomForestClassifier(random_state=0, class_weight='balanced', max_depth = params['RF_max_depth'], n_estimators = params['RF_n_estimators'])
    elif combination == 'LR':
        print(params)
        clf = LogisticRegression(penalty = params['LR_penalty'], class_weight = 'balanced', solver = 'liblinear')
    
    clf = clf.fit(X_train,y_train)
    pred = clf.predict(X_val)
    
    precision, recall, f1, support = metrics.precision_recall_fscore_support(y_val, pred, average = 'weighted', beta = 1)
    accuracy = metrics.accuracy_score(y_val, pred)
    
    return {'loss': -f1, 'status': STATUS_OK, 'accuracy': accuracy}

def flat_hyperopt(param_space, X_train, y_train, X_val, y_val, num_eval):
    #  actual optimisation function.
    trials = Trials()

    best_param = fmin(flat_objective_function, 
                      param_space, 
                      algo = tpe.suggest, 
                      max_evals = num_eval, 
                      trials = trials,
                      rstate = np.random.RandomState(1))
    
    loss = [x['result']['loss'] for x in trials.trials]
    index_min_loss = loss.index(min(loss))
    accuracy_scores = [x['result']['accuracy'] for x in trials.trials]
    
    f1 = min(loss)*-1
    accuracy = accuracy_scores[index_min_loss]
    
    return best_param, f1, accuracy

