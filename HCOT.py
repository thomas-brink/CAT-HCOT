#IMPORT FUCNTIONS functions.dataX, CAT.opt_threshold

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
from CAT import *
from baseline import * 
import importlib

import warnings
import random
warnings.filterwarnings("ignore")

class ClassHierarchy:
    
    def __init__(self, root):
        self.root = root
        self.nodes = {}
        
    def add_node(self, children, parent):
        for child in children:
            self.nodes[child] = parent
            
    def _get_leaf_nodes(self):
        leaf_nodes = []
        for child in self.nodes.keys():
            if self._get_children(child) == []:
                leaf_nodes.append(child)
        return leaf_nodes
    
    def _get_internal_nodes(self):
        internal_nodes = []
        leaves = self._get_leaf_nodes()
        for child in self.nodes.keys():
            if (child != self.root) and (child not in leaves):
                internal_nodes.append(child)
        return internal_nodes

    def _get_children(self, parent):
        return sorted([child for child, childs_parent in
                       self.nodes.items() if childs_parent == parent])
    
    def _get_parent(self, child):
        return self.nodes[child] if (child in self.nodes and child != self.root) else self.root
    
    def _get_ancestors(self, child):
        # Not including root, not including the child
        ancestors = []
        while True:
            child = self._get_parent(child)
            if child == self.root:
                break
            ancestors.append(child)
        return ancestors
    
    def _get_descendants(self, parent):
        # Return a list of the descendants of this node, not including the parent
        descendants = []
        self._depth_first(parent, descendants)
        descendants.remove(parent)
        return descendants
    
    def _depth_first(self, parent, classes):
        classes.append(parent)
        for node in self._get_children(parent):
            self._depth_first(node, classes)
            
    def _tree_distance(self, y_test, pred):
        
        y_test_path = [y_test] + self._get_ancestors(y_test) + [self.root] if y_test != self.root else [y_test] + self._get_ancestors(y_test)
        pred_path   = [pred] + self._get_ancestors(pred) + [self.root] if pred != self.root else [pred] + self._get_ancestors(pred)
        
        y_test_edges = []
        for ix, node in enumerate(y_test_path):
            length = len(y_test_path)
            if ix < length - 1:
                y_test_edges.append((node, y_test_path[ix+1]))
                
        pred_edges = []
        for ix, node in enumerate(pred_path):
            length = len(pred_path)
            if ix < length - 1:
                pred_edges.append((node, pred_path[ix+1]))        
        
        tree_distance = len([edge for edge in y_test_edges + pred_edges if edge not in pred_edges or edge not in y_test_edges])
        
        return tree_distance

class HierarchicalClassifier:
    '''
    This class contains all functions to train and test a hierarchical classifier
    '''

    def __init__(self, class_hierarchy):
        self.stages = {}
        self.class_hierarchy = class_hierarchy
        self._create_stages(self.stages, self.class_hierarchy.root, 0)

    def _create_stages(self, stages, parent, depth):
        # Get the children of this parent
        children = self.class_hierarchy._get_children(parent)
        
        if len(children) > 0:
            stage = {}
            stage['depth'] = depth
            stage['labels'] = children
            stage['classes'] = stage['labels'] + [parent]
            stage['target'] = 'target_stage_' + parent
            stages[parent] = stage

            for node in children:
                self._create_stages(stages, node, depth + 1)
                
    def _recode_label(self, classes, label):

        while label != self.class_hierarchy.root and label not in classes:
            label = self.class_hierarchy._get_parent(label)
        return label
                
    def _prep_data(self, X, y):
        
        Xcols = range(0, X.shape[1])
        Ycol = X.shape[1]
        
        df = pd.concat([X, y], axis=1, ignore_index=True)
        # Create a target column for each stage with the recoded labels
        for stage_name, stage_info in self.stages.items():
            df[stage_info['target']] = pd.DataFrame.apply(df[[Ycol]],
                                    lambda row: self._recode_label(stage_info['classes'], row[Ycol]),
                                    axis=1)
        return df, Xcols
    
    def _label_mapping(self, y_train, stage_name):
        labels = np.unique(y_train)
        int_label_mapping = dict(enumerate(labels))
        label_int_mapping = {y:x for x,y in int_label_mapping.items()}
        self.stages[stage_name]['mapping'] = {'int_label':int_label_mapping,
                                              'label_int':label_int_mapping}
        
    def _class_weights(self, y_train, stage_name):
        class_weights = class_weight.compute_class_weight('balanced',classes = np.unique(y_train),y = y_train)
        class_weights = dict(enumerate(class_weights))
        self.stages[stage_name]['classifier'].set_params(class_weight = class_weights)
    
    def fit_classifiers(self, classifiers):
        """
        Fit a classifier to each stage
        """
        if classifiers.keys() != self.stages.keys():
             raise ValueError('Your assigned classifiers do not match the stages of the hierarchy, fit a classifier to each of: '+self.stages.keys())
        else:
            for stage, classifier in classifiers.items():
                self.stages[stage]['classifier'] = classifier
    
    def fit(self, X, y):
        """
        Build a multi-classifier from training data (X, y).
        """
        df, Xcols = self._prep_data(X, y)
        self.scaler = preprocessing.MinMaxScaler().fit(X)
        
        for stage_name, stage_info in self.stages.items():
            
            dfFilter = df[df[stage_info['target']].isin(stage_info['classes'])]
            
            X_train = dfFilter[Xcols]
            y_train = dfFilter[[stage_info['target']]]
                        
            #warning - no samples to fit for stage
            if isinstance(stage_info['classifier'], KerasClassifier):
                y_train_col = pd.Series(np.ravel(y_train))
                
                self._class_weights(y_train_col, stage_name)
                self._label_mapping(y_train_col, stage_name)

                y_encoded = y_train_col.map(stage_info['mapping']['label_int'])

                if len(stage_info['labels']) > 2:
                    y_dummy = pd.DataFrame(np_utils.to_categorical(y_encoded))
                    y_train_NN = y_dummy
                else:
                    y_train_NN = np.asarray(y_encoded).reshape((-1,1))

                X_scaled = pd.DataFrame(self.scaler.transform(X_train))
                stage_info['classifier'].fit(X_scaled, y_train_NN)
            else:
                stage_info['classifier'] = stage_info['classifier'].fit(X_train, y_train)
            #print('Stage '+stage_name+' succesfully fitted')

        return self
    
    def predict(self, X):
        
        stage_number = 0
        for stage_name, stage_info in self.stages.items():
            
            if stage_name == self.class_hierarchy.root:
                y_hat = pd.DataFrame([self.class_hierarchy.root] * len(X),
                                        columns=[self.class_hierarchy.root],
                                        index=X.index)
            else:
                y_hat[stage_name] = y_hat[list(self.stages.keys())[stage_number - 1]]
            stage_number += 1             
                
            X_test = X[y_hat[stage_name].isin([stage_name])]  #warning - no samples to fit for stage
            
            if X_test.empty:
                continue
            
            if isinstance(stage_info['classifier'], KerasClassifier):
                X_scaled = pd.DataFrame(self.scaler.transform(X_test))
                if len(stage_info['labels']) == 2:
                    y_pred = pd.Series(stage_info['classifier'].predict(X_scaled).flatten()).map(stage_info['mapping']['int_label'])
                else:
                    y_pred = pd.Series(stage_info['classifier'].predict(X_scaled)).map(stage_info['mapping']['int_label'])
                y_hat_stage = pd.DataFrame(y_pred.values, index = X_test.index)
            else:
                y_hat_stage = pd.DataFrame(stage_info['classifier'].predict(X_test), index = X_test.index)
                
            y_hat = y_hat.assign(stage_col = y_hat_stage)
            y_hat.stage_col = y_hat.stage_col.fillna(y_hat[stage_name]) #fill previously predicted labels
            y_hat = y_hat.drop(stage_name, axis=1)
            y_hat = y_hat.rename(columns={'stage_col': stage_name})
            
        return y_hat.iloc[:, y_hat.shape[1] - 1]     
    
    def predict_proba(self, X, threshold = 0.5):
        
        self.blocking = {}
        stage_number = 0
        for stage_name, stage_info in self.stages.items():
            
            if stage_name == self.class_hierarchy.root:
                y_hat = pd.DataFrame([self.class_hierarchy.root] * len(X),
                                        columns=[self.class_hierarchy.root],
                                        index=X.index)
            else:
                y_hat[stage_name] = y_hat[list(self.stages.keys())[stage_number - 1]]
            stage_number += 1             
                
            X_test = X[y_hat[stage_name].isin([stage_name])]  #warning - no samples to fit for stage
            
            if isinstance(stage_info['classifier'], KerasClassifier):
                X_scaled = pd.DataFrame(self.scaler.transform(X_test))
                y_proba = stage_info['classifier'].predict_proba(X_scaled)
                y_classes = list(stage_info['mapping']['int_label'].values())
            else:
                y_proba = stage_info['classifier'].predict_proba(X_test)
                y_classes = stage_info['classifier'].classes_
            
            max_prob = np.amax(y_proba, axis=1)              #max probability of classes
            max_class = np.argmax(y_proba, axis=1)           #class number with max probability
            accept_index = np.where(max_prob >= threshold)[0]#indexes which are above threshold
            accept_class = np.take(max_class, accept_index)  #filtered list of orders which are above threshold
            
            if len(accept_class) > 0: #check if samples reach threshold
                accept_label = np.vectorize(lambda x: y_classes[x])(accept_class)                             #convert class number into label
                y_hat_stage = pd.DataFrame(accept_label, index = np.take(X_test.index.values, accept_index))  #set labels to correct position
                self.blocking[stage_name] = 1 - (len(accept_class) / len(max_class)) #blocking factor
            else:
                y_hat_stage = pd.DataFrame(columns = [0], index = X_test.index)
                self.blocking[stage_name] = 1
                
            y_hat = y_hat.assign(stage_col = y_hat_stage)
            y_hat.stage_col = y_hat.stage_col.fillna(y_hat[stage_name]) #fill previously predicted labels
            y_hat = y_hat.drop(stage_name, axis=1)
            y_hat = y_hat.rename(columns={'stage_col': stage_name})
            
        return y_hat.iloc[:, y_hat.shape[1] - 1]
    
    def predict_proba2(self, X, THRESHOLDS):
        
        self.blocking = {}
        self.Tblocking = {}
        stage_number = 0
        for stage_name, stage_info in self.stages.items():
            
            if stage_name == self.class_hierarchy.root:
                y_hat = pd.DataFrame([self.class_hierarchy.root] * len(X),
                                        columns=[self.class_hierarchy.root],
                                        index=X.index)
            else:
                y_hat[stage_name] = y_hat[list(self.stages.keys())[stage_number - 1]]
            stage_number += 1             
                
            X_test = X[y_hat[stage_name].isin([stage_name])]  #warning - no samples to fit for stage
            
            if X_test.empty:
                self.blocking[stage_name] = None
                self.Tblocking[stage_name] = None
                continue
            
            if isinstance(stage_info['classifier'], KerasClassifier):
                X_scaled = pd.DataFrame(self.scaler.transform(X_test))
                y_proba = stage_info['classifier'].predict_proba(X_scaled)
                y_classes = list(stage_info['mapping']['int_label'].values())
            else:
                y_proba = stage_info['classifier'].predict_proba(X_test)
                y_classes = stage_info['classifier'].classes_
            
            max_prob = np.amax(y_proba, axis=1)              #max probability of classes
            max_class = np.argmax(y_proba, axis=1)           #class number with max probability
            max_class_thresholds = np.vectorize(lambda x: THRESHOLDS[y_classes[x]])(max_class)  #get node specific threshold
            
            accept_index = np.where(max_prob >= max_class_thresholds)[0]

            accept_class = np.take(max_class, accept_index)  #filtered list of orders which are above threshold
            
            if len(accept_class) > 0: #check if samples reach threshold
                accept_label = np.vectorize(lambda x: y_classes[x])(accept_class)                             #convert class number into label
                y_hat_stage = pd.DataFrame(accept_label, index = np.take(X_test.index.values, accept_index))  #set labels to correct position
                
                self.blocking[stage_name] = 1 - (len(accept_class) / len(max_class)) #blocking factor
                self.Tblocking[stage_name] = len(max_class) - len(accept_class)
            else:
                y_hat_stage = pd.DataFrame(columns = [0], index = X_test.index)
                self.blocking[stage_name] = 1
                self.Tblocking[stage_name] = len(max_class)
                
            y_hat = y_hat.assign(stage_col = y_hat_stage)
            y_hat.stage_col = y_hat.stage_col.fillna(y_hat[stage_name]) #fill previously predicted labels
            y_hat = y_hat.drop(stage_name, axis=1)
            y_hat = y_hat.rename(columns={'stage_col': stage_name})
            
        return y_hat.iloc[:, y_hat.shape[1] - 1]
    
    def get_probabilities(self, X, y):
        
        df, Xcols = self._prep_data(X, y)
        
        stage_number = 0
        
        y_hat = pd.DataFrame(columns = [self.class_hierarchy.root], index = X.index)
        
        for stage_name, stage_info in self.stages.items():
                
            stage_number += 1             
            
            dfFilter = df[df[stage_info['target']].isin(stage_info['classes'])]
            
            X_test = dfFilter[Xcols]
            y_test = dfFilter[[stage_info['target']]]
            
            if isinstance(stage_info['classifier'], KerasClassifier):
                X_scaled = pd.DataFrame(self.scaler.transform(X_test))
                y_proba = stage_info['classifier'].predict_proba(X_scaled)
                y_classes = list(stage_info['mapping']['int_label'].values())
            else:
                y_proba = stage_info['classifier'].predict_proba(X_test)
                y_classes = stage_info['classifier'].classes_
            
            y_hat_stage = pd.DataFrame(y_proba, index = X_test.index)

            for col, label in enumerate(y_classes):
                y_hat[label] = y_hat_stage[col]
               
        return y_hat

def dynamicHierarchicalClassifier(START, END):  
    '''
    CAT-HCOT algorithm
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
    
    CERTAINTY = 0.7
    
    statistics, previous_pred_block, feature_importances = None, None, None

    df, X_col, Y_col, historic_variable, DATE = initialiseData()
    
    for DAYS in range(START, END+1):

        X, y = functions.dataX(df, DATE, X_col, Y_col, historic_variable, DAYS)

        X_train_preburn = X.iloc[:int(0.8*len(X))]
        y_train_preburn = y.iloc[:int(0.8*len(y))]

        X_train = X_train_preburn.iloc[int(0.1*len(X_train_preburn)):]
        y_train = y_train_preburn.iloc[int(0.1*len(y_train_preburn)):]

        X_test = X.iloc[int(0.8*len(X)):]
        y_test = y.iloc[int(0.8*len(y)):]

        N_test = len(y_test)

        HC = HierarchicalClassifier(Tree)
        HC.fit_classifiers({'ORDERS'  : LogisticRegression(random_state=0, class_weight='balanced', solver = 'liblinear', penalty = hypers.loc[DAYS, '1_penalty']),
                            'KNOWN'   : RandomForestClassifier(random_state=0, class_weight='balanced', max_depth = hypers.loc[DAYS, '2_max_depth'], n_estimators = hypers.loc[DAYS, '2_n_estimators']),
                            'UNHAPPY' : RandomForestClassifier(random_state=0, class_weight='balanced', max_depth = hypers.loc[DAYS, '3_max_depth'], n_estimators = hypers.loc[DAYS, '3_n_estimators'])})

        HC = HC.fit(X_train,y_train)

        y_train_hat = HC.get_probabilities(X_train, y_train)
        probs = pd.concat([y_train, y_train_hat], axis=1)

        THRESHOLDS = {}
        for node in range(1,8):
            name, threshold = opt_threshold(probs, node, DAYS, CERTAINTY, 2)
            THRESHOLDS[name] = threshold

        if DAYS == START: #create dataframe to save predictions
            y_hat = pd.DataFrame([Tree.root] * len(X_test),
                                    columns=[DAYS],
                                    index=X_test.index)
            index_no_leaf = X_test.index
        else:
            y_hat[DAYS] = y_hat[DAYS - 1]

        if DAYS < END:
            pred = HC.predict_proba2(X_test.loc[index_no_leaf], THRESHOLDS = THRESHOLDS)

            check_no_leaf = ~pred.isin(Tree._get_leaf_nodes())
            index_no_leaf = check_no_leaf[check_no_leaf].index
            check_leaf    = pred.isin(Tree._get_leaf_nodes())      #from current non_leaf predictions which are now leaf
            index_leaf    = check_leaf[check_leaf].index
            y_hat_stage   = pd.DataFrame(pred, index = index_leaf)
        else:
            pred        = HC.predict(X_test.loc[index_no_leaf]) #last day you want a label for each order
            y_hat_stage = pd.DataFrame(pred, index = index_no_leaf)
            index_leaf  = index_no_leaf

        y_hat = y_hat.assign(stage_col = y_hat_stage)
        y_hat.stage_col = y_hat.stage_col.fillna(y_hat[DAYS]) #fill previously predicted labels
        y_hat = y_hat.drop(DAYS, axis=1)
        y_hat = y_hat.rename(columns={'stage_col': DAYS})

        current_pred = y_hat.iloc[:, y_hat.shape[1] - 1]

        statistics, feature_importances, previous_pred_block = get_performance(DAYS, END, pred, current_pred, index_leaf, index_no_leaf, 
                                                                               previous_pred_block, THRESHOLDS, OPTION, CERTAINTY, y_test, Tree, HC, feature_importances, statistics)
        
        file_name = 'statistics_optimal_'+str(CERTAINTY)+'.json'
        path_name = '/Users/LV/Desktop/' + file_name
        with open(path_name, 'w') as f:
            json.dump(statistics, f, cls = NumpyEncoder)

        print('DAYS: ',DAYS)
     
    final_pred = y_hat.iloc[:, y_hat.shape[1] - 1]
        
    return final_pred, statistics, feature_importances

def get_performance(DAYS, END, pred, current_pred, index_leaf, index_no_leaf, previous_pred_block, THRESHOLDS, OPTION, CERTAINTY, y_test, Tree, HC, feature_importances, statistics):
    
    #Initialize Dictionary at Day 0
    
    if DAYS == 0:
        statistics = {'%classified'     :{}, 'N_classified'         :{},  'N_predicted' : {},
                      'leaf_accuracy'   :{}, 'total_leaf_accuracy'  :{},
                      'leaf_precision'  :{}, 'total_leaf_precision' :{},
                      'leaf_recall'     :{}, 'total_leaf_recall'    :{},
                      'label_precision' :{}, 
                      'label_recall'    :{}, 
                      'block_precision' :{},
                      'block_recall'    :{},
                      'block_Nchange'   :{}, 'block_Pchange'        :{},
                      '%blocking'       :{}, '%Tblocking'           :{},
                      'tree_error'      :{},
                      'thresholds'      :{},
                      'option'          :{},
                      'certainty'       :{}}

        for leaf in Tree._get_leaf_nodes()+Tree._get_internal_nodes(): 
            statistics['precision_'+leaf] = {}
            statistics['recall_'+leaf]    = {}
            statistics['f1_'+leaf]        = {}
            
        feature_importances = pd.DataFrame(index = X_col)
        decision_trees = {}
    
    #Get Daily information
    
    check_block = pred.isin(Tree._get_internal_nodes())
    index_block = check_block[check_block].index        
        
    total_check_leaf = current_pred.isin(Tree._get_leaf_nodes())   #of all predictions which are now leaf
    total_index_leaf = total_check_leaf[total_check_leaf].index
        
    if DAYS > 0:
        block = pd.concat([previous_pred_block, pred.loc[previous_pred_block.index]], axis=1, keys = [0,1])
        block['Nchange'] = block.apply(lambda row: 0 if row[1] in Tree._get_descendants(row[0])+[row[0]] else 1, axis = 1)
        block['Pchange'] = block.apply(lambda row: 1 if row[1] in Tree._get_descendants(row[0]) else 0, axis = 1)
    previous_pred_block = pred.loc[index_block]
    previous_index_block = index_block #was commented?

    y_test = y_test['detailedMatchClassification']
    test_pred = pd.concat([y_test.loc[index_leaf], current_pred[index_leaf]], axis=1, keys = [0,1])
    test_pred['TE'] = test_pred.apply(lambda row: Tree._tree_distance(row[0], row[1]), axis = 1)
    
    #Update Dictionary
    
    statistics['option'][DAYS]          = OPTION
    statistics['certainty'][DAYS]       = CERTAINTY
    statistics['thresholds'][DAYS]      = THRESHOLDS
    statistics['%classified'][DAYS]     = current_pred.isin(Tree._get_leaf_nodes()).sum() / len(y_test)
    statistics['N_classified'][DAYS]    = int(len(index_leaf))
    statistics['N_predicted'][DAYS]     = int(len(pred))

    statistics['leaf_accuracy'][DAYS]   = metrics.accuracy_score(y_test.loc[index_leaf], pred.loc[index_leaf])
    statistics['leaf_precision'][DAYS]  = precision_score_ancestors(Tree, y_test.loc[index_leaf], pred.loc[index_leaf])
    statistics['leaf_recall'][DAYS]     = recall_score_ancestors(Tree, y_test.loc[index_leaf], pred.loc[index_leaf])

    for leaf in Tree._get_leaf_nodes()+Tree._get_internal_nodes():
        leaf_ix = pred.loc[pred == leaf].index
        statistics['precision_'+leaf][DAYS] = precision_score_ancestors(Tree, y_test.loc[leaf_ix], pred.loc[leaf_ix])
        statistics['recall_'+leaf][DAYS]    = recall_score_ancestors(Tree, y_test.loc[leaf_ix], pred.loc[leaf_ix])
        statistics['f1_'+leaf][DAYS]        = f1_score_ancestors(Tree, y_test.loc[leaf_ix], pred.loc[leaf_ix], beta = 1)
        
    for clf in list(HC.stages.keys()):
        if isinstance(HC.stages[clf]['classifier'],RandomForestClassifier):
            feature_importances[clf+'_'+str(DAYS)] = HC.stages[clf]['classifier'].feature_importances_ 
        elif isinstance(HC.stages[clf]['classifier'],LogisticRegression):
            feature_importances[clf+'_'+str(DAYS)] = HC.stages[clf]['classifier'].coef_[0] 

    statistics['total_leaf_accuracy'][DAYS]  = metrics.accuracy_score(y_test.loc[total_index_leaf], current_pred.loc[total_index_leaf])
    statistics['total_leaf_precision'][DAYS] = precision_score_ancestors(Tree, y_test.loc[total_index_leaf], current_pred.loc[total_index_leaf])
    statistics['total_leaf_recall'][DAYS]    = recall_score_ancestors(Tree, y_test.loc[total_index_leaf], current_pred.loc[total_index_leaf])

    statistics['label_precision'][DAYS]  = precision_score_ancestors(Tree, y_test.loc[index_leaf.union(index_block)], pred.loc[index_leaf.union(index_block)]) 
    statistics['label_recall'][DAYS]     = recall_score_ancestors(Tree, y_test.loc[index_leaf.union(index_block)], pred.loc[index_leaf.union(index_block)])  

    statistics['block_precision'][DAYS] = precision_score_ancestors(Tree, y_test.loc[index_block], pred.loc[index_block]) if DAYS < END else None
    statistics['block_recall'][DAYS]    = recall_score_ancestors(Tree, y_test.loc[index_block], pred.loc[index_block]) if DAYS < END else None
    statistics['block_Nchange'][DAYS]   = block['Nchange'].sum() / block['Nchange'].count() if DAYS > 0 else None
    statistics['block_Pchange'][DAYS]   = block['Pchange'].sum() / block['Pchange'].count() if DAYS > 0 else None
    statistics['%blocking'][DAYS]       = HC.blocking  if len(total_index_leaf) != len(y_test) else {'ORDERS':None,'KNOWN':None,'UNHAPPY':None}
    statistics['%Tblocking'][DAYS]      = HC.Tblocking if len(total_index_leaf) != len(y_test) else {'ORDERS':None,'KNOWN':None,'UNHAPPY':None}

    statistics['tree_error'][DAYS]      = np.mean(test_pred['TE'])
        
    return statistics, feature_importances, previous_pred_block

def _aggregate_class_sets(set_function, y_true, y_pred):
    intersection_sum = 0
    true_sum = 0
    predicted_sum = 0
    for true, pred in zip(list(y_true), list(y_pred)):
        true_set = set([true] + set_function(true))
        pred_set = set([pred] + set_function(pred))
        intersection_sum += len(true_set.intersection(pred_set))
        true_sum += len(true_set)
        predicted_sum += len(pred_set)
    return (true_sum, predicted_sum, intersection_sum)

def precision_score_ancestors(class_hierarchy, y_true, y_pred):
    true_sum, predicted_sum, intersection_sum = _aggregate_class_sets(
        class_hierarchy._get_ancestors, y_true, y_pred)
    if predicted_sum == 0:
        return None
    else:
        return intersection_sum / predicted_sum

def recall_score_ancestors(class_hierarchy, y_true, y_pred):
    true_sum, predicted_sum, intersection_sum = _aggregate_class_sets(
        class_hierarchy._get_ancestors, y_true, y_pred)
    if true_sum == 0:
        return None
    else:
        return intersection_sum / true_sum

def f1_score_ancestors(class_hierarchy, y_true, y_pred, beta):
    precision = precision_score_ancestors(class_hierarchy, y_true, y_pred)
    recall = recall_score_ancestors(class_hierarchy, y_true, y_pred)
    if (precision == None) or (recall == None):
        return None
    elif (precision == 0) or (recall == 0):
        return 0
    else:
        return ((beta ** 2 + 1) * precision * recall) / ((beta ** 2 * precision) + recall)