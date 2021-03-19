# Seminar-QM-BA
Code for Seminar in QM &amp; BA

The files in this directory belong to the paper "A Certainty-Based Approach for Dynamic Hierarchical Classification of Product Order Satisfaction".

## PaperCAT-HCOT.pdf

Research paper

## Data_Cleaning_Preparation.ipynb

This file includes the code which was used to clean the data

## functions.py

This file includes functions which are used to create the input data for the HCOT algorithm. 
- dataX(df, DATE, X_col, y_col, historic_variable, days): main function for creating input data which computes the features based on X, which is the point in time when we make the predictions.
- addKnownColumns(df,X): add the dynamic variables accociated with the match determinants
- addProductColumns(df,X): add dynamic variables accociated with product groups
- addSellerColumns(df,X): add dynamic variables accociated with sellerId's
- addHistoricPerformance(df, variable = 'transporterCode', X = 0): add dynamic variables with historic performance. This creates 3 columns with the historic value for Happy, Unhappy and Unknown matches.

## HCOT.py

This file includes the complete HCOT algoritm together with function to test and evaluate the performance
- ClassHierarchy(root): Class with all functions to construct a hierarchy which is thereafter used for HierarchicalClassifier, main function:
  - add_node(children, parent): add a node with corresponding childnodes to the hierachy
- HierarchicalClassifier(class_hierarchy): Class with all functions to train and test a hierarchical classifier, the main functions are:
  - fit_classifiers(classifiers): function to fit the classifier per parent node, the input is a dictionary with the parent nodes as keys and values the classifiers
  - fit(X, y): train the hierarchy with X and Y
  - predict(X): predict the values of X with the fitted hierarchy
  - predict_proba(X, threshold): predict the values of X by making use of a pre-defined threshold for each node, if threshold not reached then blocked
  - predict_proba2(X, THRESHOLDS): predict the values of X by making use of node-specific thresholds (THRESHOLDS), if threshold not reached then blocked
  - get_probabilities(X, y): get the class probabilities for each instance, this is used for training the thresholds with CAT
- initialiseData(): function to load data and retrieve features for prediction (included in dynamicHierarchicalClassifier)
- dynamicHierarchicalClassifier(START, END): CAT-HCOT algorithm to test instances over a period from START till END
- get_performance(...): function included in dynamicHierarchicalClassifier to compute daily/global performance measures while running HCOT
- precision_score_ancestors(class_hierarchy, y_true, y_pred): hierarchical precision score, input is the class_hierarchy from ClassHierarchy
- recall_score_ancestors(class_hierarchy, y_true, y_pred): hierarchical recall score, input is the class_hierarchy from ClassHierarchy
- f1_score_ancestors(class_hierarchy, y_true, y_pred, beta): hierarchical f1 score, input is the class_hierarchy from ClassHierarchy

## CAT.py

This file includes the CAT algorithm for both CAT-HCOT and flat CAT-HCOT
- opt_threshold(probs, node, day, certainty, option = 2, steps = 100): CAT for HCOT, using the hierarchy from our report
- flat_thresholds(probs, node, day, certainty, steps = 100): CAT for flat HCOT, using the flat hierarchy from our report

## validation.py

This file includes the validation procedure for HCOT and flat CAT-HCOT (baseline)
- HCOT_optimisation(): optimisation of a single hierarchy with pre-defined set of possible combination which includes the functions:
  - get_hyperspace(combination): create hyperspace for optimisation (currently supports DT, RF, NN and LR)
  - clf_hypers(params): function which distributes the hyperparameters over the 3 specific parent nodes in the hierarchy
  - objective_function(params): objective for optimising daily hierarchical classifier combination in HCOT
  - hyperopt(param_space, X_train, y_train, X_val, y_val, num_eval): actual optimisation function for a single hierarchy with pre-set classifier combination
- flat_HCOT_optimisation(): optimisation of single 'one-level hierarchy' of the flat HCOT basline which includes similar functions (supporting LR + RF):
  - flat_get_hyperspace(combination): create hyperspace for optimisation 
  - flat_objective_function(params): objective for optimising either LR or RF
  - flat_hyperopt(param_space, X_train, y_train, X_val, y_val, num_eval): actual optimisation function

## baseline.py

This file includes the baseline methods: static HCOT and flat CAT-HCOT
- staticHierarchicalClassifier(START, END): static HCOT (no blocking), hence predicting all instances on a single point in time
- dynamicFlatClassifier(START, END): flat CAT-HCOT (one-level hierarchy), using similar blokking approach as in CAT-HCOT
