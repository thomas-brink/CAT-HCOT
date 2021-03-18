# Seminar-QM-BA
Code for Seminar in QM &amp; BA

The files in this directory belong to the paper "A Certainty-Based Approach for Dynamic Hierarchical Classification of Product Order Satisfaction".

## PaperCAT-HCOT.pdf

Research paper

## Data_Cleaning_Preparation.ipynb

This file includes the code which was used to clean the data

## functions.py

This file includes functions which are used to create the input data for the HCOT algorithm. 
- dataX(df, DATE, X_col, y_col, historic_variable, days):
- addKnownColumns(df,X):
- addProductColumns(df,X):
- addSellerColumns(df,X):
- addHistoricPerformance(df, variable = 'transporterCode', X = 0): 

## HCOT.py

This file includes the complete HCOT algoritm together with function to test and evaluate the performance
- ClassHierarchy(root): Class with all functions to construct a hierarchy which is thereafter used for HierarchicalClassifier, main function:
  - add_node(children, parent):
- HierarchicalClassifier(class_hierarchy): Class with all functions to train and test a hierarchical classifier, the main functions are:
  - fit_classifiers(classifiers):
  - fit(X, y):
  - predict(X):
  - predict_proba(X, threshold):
  - predict_proba2(X, THRESHOLDS):
  - get_probabilities(X, y)
- initialiseData(): function to load data and retrieve features for prediction (included in dynamicHierarchicalClassifier)
- dynamicHierarchicalClassifier(START, END): CAT-HCOT algorithm
- get_performance(...): function included in dynamicHierarchicalClassifier to compute daily/global performance measures while running HCOT
- precision_score_ancestors(class_hierarchy, y_true, y_pred):
- recall_score_ancestors(class_hierarchy, y_true, y_pred):
- f1_score_ancestors(class_hierarchy, y_true, y_pred, beta):

## CAT.py

This file includes the CAT algorithm for both CAT-HCOT and flat CAT-HCOT
- opt_threshold(probs, node, day, certainty, option = 2, steps = 100): CAT for HCOT
- flat_thresholds(probs, node, day, certainty, steps = 100): CAT for flat HCOT

## validation.py

This file includes the validation procedure for HCOT and flat CAT-HCOT (baseline)
- HCOT_optimisation(): optimisation of a single hierarchy with pre-defined set of possible combination which includes the functions:
  - get_hyperspace(combination): create hyperspace for optimisation (currently supports DT, RF, NN and LR)
  - clf_hypers(params): function which distributes the hyperparameters over the 3 specific parent nodes in the hierarchy
  - objective_function(params): objective for optimising daily hierarchical classifier combination in HCOT
  - hyperopt(param_space, X_train, y_train, X_val, y_val, num_eval): actual optimisation function for a single hierarchy with pre-set classifier combination
- flat_HCOT_optimisation(): optimisation of single 'one-level hierarchy' of the flat HCOT basline which includes similar functions (supporting LR + RF):
  - flat_get_hyperspace(combination):
  - flat_objective_function(params):
  - flat_hyperopt(param_space, X_train, y_train, X_val, y_val, num_eval):

## baseline.py

This file includes the baseline methods: static HCOT and flat CAT-HCOT
- staticHierarchicalClassifier(START, END): static HCOT
- dynamicFlatClassifier(START, END): flat CAT-HCOT
