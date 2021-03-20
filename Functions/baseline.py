import pandas as pd
import numpy as np

from datetime import date, timedelta, datetime
import time
import math

from sklearn import preprocessing
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

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
import importlib

import warnings
warnings.filterwarnings("ignore")

def staticHierarchicalClassifier(START, END):
    '''
    Static HCOT baseline (no blocking), hence predicting all instances on a single point in time (each day mandatory leaf node prediction is used)
    Optimal hyperparameters are already included. 
    '''
    
    Tree = ClassHierarchy('ORDERS')
    Tree.add_node(['UNKNOWN','KNOWN'], 'ORDERS')
    Tree.add_node(['HAPPY','UNHAPPY'], 'KNOWN')
    Tree.add_node(['MILDLY UNHAPPY','MEDIUM UNHAPPY','HEAVILY UNHAPPY'], 'UNHAPPY')
    
    hypers = pd.DataFrame({'1_penalty'     : ['l1','l1','l2','l2','l2','l2','l1','l1','l1','l1','l1'],
                           '2_max_depth'   : [ 9,10,12,12,12,12,10,10,10,10,10], 
                           '2_n_estimators': [35,45,30,30,30,30,45,45,45,45,45],
                           '3_max_depth'   : [14,14,14,14,14,14,14,14,14,14,14], 
                           '3_n_estimators': [20,45,30,30,30,30,45,45,45,45,45]})
    
    statistics = {'accuracy':{},
                  'precision':{},
                  'recall':{},
                  'f1':{}}
    for leaf in Tree._get_leaf_nodes(): 
        statistics['precision_'+leaf] = {}
        statistics['recall_'+leaf]    = {}
        statistics['f1_'+leaf]        = {}
    
    df, X_col, Y_col, historic_variable, DATE = initialiseData()
    
    for DAYS in range(START, END+1):

        X, y = dataX(df, DATE, X_col, Y_col, historic_variable, DAYS)

        X_train_preburn = X.iloc[:int(0.8*len(X))]
        y_train_preburn = y.iloc[:int(0.8*len(y))]

        X_train = X_train_preburn.iloc[int(0.1*len(X_train_preburn)):]
        y_train = y_train_preburn.iloc[int(0.1*len(y_train_preburn)):]

        X_test = X.iloc[int(0.8*len(X)):]
        y_test = y.iloc[int(0.8*len(y)):]

        HC = HierarchicalClassifier(Tree)
        HC.fit_classifiers({'ORDERS'  : LogisticRegression(random_state=0, class_weight='balanced', solver = 'liblinear', penalty = hypers.loc[DAYS, '1_penalty']),
                            'KNOWN'   : RandomForestClassifier(random_state=0, class_weight='balanced', max_depth = hypers.loc[DAYS, '2_max_depth'], n_estimators = hypers.loc[DAYS, '2_n_estimators']),
                            'UNHAPPY' : RandomForestClassifier(random_state=0, class_weight='balanced', max_depth = hypers.loc[DAYS, '3_max_depth'], n_estimators = hypers.loc[DAYS, '3_n_estimators'])})

        HC = HC.fit(X_train,y_train)
        pred = HC.predict(X_test)
        
        y_test = y_test['detailedMatchClassification']
        
        statistics['accuracy'][DAYS] = metrics.accuracy_score(y_test, pred)
        statistics['precision'][DAYS] = precision_score_ancestors(Tree, y_test, pred)
        statistics['recall'][DAYS] = recall_score_ancestors(Tree, y_test, pred)
        statistics['f1'][DAYS] = f1_score_ancestors(Tree, y_test, pred, beta = 1)
        
        for leaf in Tree._get_leaf_nodes():
            leaf_ix = pred.loc[pred == leaf].index
            statistics['precision_'+leaf][DAYS] = precision_score_ancestors(Tree, y_test.loc[leaf_ix], pred.loc[leaf_ix])
            statistics['recall_'+leaf][DAYS]    = recall_score_ancestors(Tree, y_test.loc[leaf_ix], pred.loc[leaf_ix])
            statistics['f1_'+leaf][DAYS]        = f1_score_ancestors(Tree, y_test.loc[leaf_ix], pred.loc[leaf_ix], beta = 1)
            
        print('DAY ',DAYS)
            
    return statistics

def dynamicFlatClassifier(START, END):  
    '''
    Flat CAT-HCOT baseline (one-level hierarchy) which does make use of a similar blocking approach with CAT. Only the start and end period should be given as input. 
    Optimal hyperparameters are already included. 
    '''

    hypers = pd.DataFrame({'LR_penalty'     : ['l1','l1','l1','l1','l1','l1','l2','l1','l1','l1','l1'],
                           'RF_max_depth'   : [14,14,14,14,14,14,14,14,14,14,14], 
                           'RF_n_estimators': [45,45,40,45,40,45,40,45,45,45,45]})
    
    Tree = ClassHierarchy('ORDERS')
    Tree.add_node(['UNKNOWN','KNOWN'], 'ORDERS')
    Tree.add_node(['HAPPY','UNHAPPY'], 'KNOWN')
    Tree.add_node(['MILDLY UNHAPPY','MEDIUM UNHAPPY','HEAVILY UNHAPPY'], 'UNHAPPY')

    CERTAINTY = 0.7
    
    statistics = {'accuracy'  :{},
                  'classified':{},
                  'thresholds':{},
                  'precision' :{},
                  'recall'    :{},
                  'f1'        :{}}
    for leaf in ['HAPPY','UNKNOWN','MILDLY UNHAPPY','MEDIUM UNHAPPY','HEAVILY UNHAPPY']: 
        statistics['precision_'+leaf] = {}
        statistics['recall_'+leaf]    = {}
        statistics['f1_'+leaf]        = {}
    for leaf in ['HAPPY','UNKNOWN','MILDLY UNHAPPY','MEDIUM UNHAPPY','HEAVILY UNHAPPY']: 
        statistics['2precision_'+leaf] = {}
        statistics['2recall_'+leaf]    = {}
        statistics['2f1_'+leaf]        = {}
     
    df, X_col, Y_col, historic_variable, DATE = initialiseData()
    
    for DAYS in range(START, END+1):

        X, y = dataX(df, DATE, X_col, Y_col, historic_variable, DAYS)

        X_train_preburn = X.iloc[:int(0.8*len(X))]
        y_train_preburn = y.iloc[:int(0.8*len(y))]

        X_train = X_train_preburn.iloc[int(0.1*len(X_train_preburn)):]
        y_train = y_train_preburn.iloc[int(0.1*len(y_train_preburn)):]

        X_test = X.iloc[int(0.8*len(X)):]
        y_test = y.iloc[int(0.8*len(y)):]

        if DAYS < 5:
            clf = LogisticRegression(random_state=0, class_weight='balanced', solver = 'liblinear', penalty = hypers.loc[DAYS, 'LR_penalty'])
        else:
            clf = RandomForestClassifier(random_state=0, class_weight='balanced', max_depth = hypers.loc[DAYS, 'RF_max_depth'], n_estimators = hypers.loc[DAYS, 'RF_n_estimators'])
            
        clf.fit(X_train, y_train)

        y_train_hat = clf.predict_proba(X_train) 
        y_classes = clf.classes_
        y_train_hat = pd.DataFrame(y_train_hat, index = X_train.index, columns = y_classes)
        probs = pd.concat([y_train, y_train_hat], axis=1)
        
        THRESHOLDS = {}
        for node in range(1,6):
            name, threshold = flat_thresholds(probs, node, DAYS, CERTAINTY, steps = 100)
            THRESHOLDS[name] = threshold

        if DAYS == START: #create dataframe to save predictions
            y_hat = pd.DataFrame(['ORDERS'] * len(X_test),
                                    columns=[DAYS],
                                    index=X_test.index)
            index_no_leaf = X_test.index
        else:
            y_hat[DAYS] = y_hat[DAYS - 1]

        if DAYS < END:
            X_test_ = X_test.loc[index_no_leaf]
            y_proba = clf.predict_proba(X_test_)
            y_classes = clf.classes_
        
            max_prob = np.amax(y_proba, axis=1)              #max probability of classes
            max_class = np.argmax(y_proba, axis=1)           #class number with max probability
            max_class_thresholds = np.vectorize(lambda x: THRESHOLDS[y_classes[x]])(max_class)  #get node specific threshold

            accept_index = np.where(max_prob >= max_class_thresholds)[0]
            accept_class = np.take(max_class, accept_index)  #filtered list of orders which are above threshold

            if len(accept_class) > 0: #check if samples reach threshold
                accept_label = np.vectorize(lambda x: y_classes[x])(accept_class)                             #convert class number into label
                y_hat_stage = pd.DataFrame(accept_label, index = np.take(X_test_.index.values, accept_index))  #set labels to correct position
            else:
                y_hat_stage = pd.DataFrame(columns = [0], index = X_test_.index)
                
            index_leaf = y_hat_stage.index

        else:
            pred        = clf.predict(X_test.loc[index_no_leaf]) #last day you want a label for each order
            y_hat_stage = pd.DataFrame(pred, index = index_no_leaf)
            index_leaf  = index_no_leaf

        y_hat = y_hat.assign(stage_col = y_hat_stage)
        y_hat.stage_col = y_hat.stage_col.fillna(y_hat[DAYS]) #fill previously predicted labels
        y_hat = y_hat.drop(DAYS, axis=1)
        y_hat = y_hat.rename(columns={'stage_col': DAYS})

        current_pred = y_hat.iloc[:, y_hat.shape[1] - 1]
        check_no_leaf = (current_pred == 'ORDERS')    #from current non_leaf predictions which are now leaf
        index_no_leaf = check_no_leaf[check_no_leaf].index
        
        statistics['accuracy'][DAYS]   = metrics.accuracy_score(y_test.loc[index_leaf], current_pred.loc[index_leaf])
        statistics['classified'][DAYS] = (current_pred != 'ORDERS').sum() / len(y_test)
        statistics['thresholds'][DAYS] = THRESHOLDS
        
        precision, recall, f1, support = metrics.precision_recall_fscore_support(y_test.loc[index_leaf], current_pred.loc[index_leaf], average = 'weighted', beta = 1)
        
        statistics['precision'][DAYS] = precision
        statistics['recall'][DAYS]    = recall
        statistics['f1'][DAYS]        = f1
        
        precision, recall, f1, support = metrics.precision_recall_fscore_support(y_test.loc[index_leaf], current_pred.loc[index_leaf], average = None, labels = ['HAPPY','UNKNOWN','MILDLY UNHAPPY','MEDIUM UNHAPPY','HEAVILY UNHAPPY'])
        
        for ix,leaf in enumerate(['HAPPY','UNKNOWN','MILDLY UNHAPPY','MEDIUM UNHAPPY','HEAVILY UNHAPPY']):
            statistics['precision_'+leaf][DAYS] = precision[ix]
            statistics['recall_'+leaf][DAYS]    = recall[ix]
            statistics['f1_'+leaf][DAYS]        = f1[ix]
            
        for leaf in ['HAPPY','UNKNOWN','MILDLY UNHAPPY','MEDIUM UNHAPPY','HEAVILY UNHAPPY']:
            leaf_ix = current_pred.loc[current_pred == leaf].index
            statistics['2precision_'+leaf][DAYS] = precision_score_ancestors(Tree, y_test['detailedMatchClassification'].loc[leaf_ix], current_pred.loc[leaf_ix])
            statistics['2recall_'+leaf][DAYS]    = recall_score_ancestors(Tree, y_test['detailedMatchClassification'].loc[leaf_ix], current_pred.loc[leaf_ix])
            statistics['2f1_'+leaf][DAYS]        = f1_score_ancestors(Tree, y_test['detailedMatchClassification'].loc[leaf_ix], current_pred.loc[leaf_ix], beta = 1)
        
        file_name = 'flat_statistics_'+str(CERTAINTY)+'.json'
        #path_name = 'path...' + file_name
        path_name = '/Users/thoma/Documents/seminar_data/' + file_name
        with open(path_name, 'w') as f:
            json.dump(statistics, f, cls = NumpyEncoder)

        print('DAYS: ',DAYS)
     
        final_pred = y_hat.iloc[:, y_hat.shape[1] - 1]
        accuracy = metrics.accuracy_score(y_test, final_pred)
        precision, recall, f1, support = metrics.precision_recall_fscore_support(y_test, final_pred, average = None, labels = ['HAPPY','UNKNOWN','MILDLY UNHAPPY','MEDIUM UNHAPPY','HEAVILY UNHAPPY'], beta = 1)
        print(accuracy)
        print(precision)
        print(recall)
        print(precision_score_ancestors(Tree, y_test['detailedMatchClassification'], final_pred))
        print(recall_score_ancestors(Tree, y_test['detailedMatchClassification'], final_pred))
        for leaf in ['HAPPY','UNKNOWN','MILDLY UNHAPPY','MEDIUM UNHAPPY','HEAVILY UNHAPPY']:
            leaf_ix = final_pred.loc[final_pred == leaf].index
            print(leaf,precision_score_ancestors(Tree, y_test['detailedMatchClassification'].loc[leaf_ix], final_pred.loc[leaf_ix]),
                  recall_score_ancestors(Tree, y_test['detailedMatchClassification'].loc[leaf_ix], final_pred.loc[leaf_ix]))
        
    return final_pred, statistics




