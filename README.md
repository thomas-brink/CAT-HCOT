# Seminar-QM-BA
Code for Seminar in QM &amp; BA.

The files in this repository belong to the paper "A Certainty-Based Approach for Dynamic Hierarchical Classification of Product Order Satisfaction". This paper models the classification of customer satisfaction belonging to product orders at an online retailer in the Netherlands using hierarchical classification over time with a certainty-based blocking approach. The main components of this paper are the Hierarchical Classification Over Time (HCOT) and Certainty-based Automated Thresholds (CAT) procedures. These two components are outlined in corresponding code files below. In addition, we provide code for data cleaning and preparation, dynamic data generation, training and validation, and baseline comparison. The code for implementing hierarchical classifiers is based on https://github.com/davidwarshaw/hmc.

The steps for running our code are as follows.
1. Run the `Data_Cleaning_Preparation' file. This file loads in the data received from the retailer and creates a .csv file that contains the cleaned and prepared data.
2. Load in the 'dynamic_input_data' file. This file creates dynamic data variables and the input data to be used for the CAT and HCOT procedures.
3. Run the code as provided in the 'Seminar_Hierarchical_Classifiers' file. This file uses the cleaned and prepared data from step 1 as well as the functions provided in step 2 to perform both the validation and testing process for CAT-HCOT and the two baselines. More specific information on this code is provided in the specific section below. 

Before elaborating on these three steps in detail, we provide a short note on the code and briefly mention the paper that this code is applied to.

Please note that some of the code requires the user to select a path from which to read input or to which to write output. Currently, examples for such paths are provided and a general format is provided as well. However, users should fill in these paths themselves when applying the code. Furthermore, we note that the data used for this research paper is not publicly available due to confidentiality agreements between the researchers and the online retailer. 

## PaperCAT-HCOT.pdf

File with our respective research paper. 

**Abstract**: Online retailers collaborate more and more with partners to sell products via their platform, making it increasingly important to preserve platform quality. In this paper, we predict the quality of matches between sellers and customers associated with product orders, where we face a trade-off between the accuracy and timeliness of our predictions. To deal with this trade-off, we introduce the Hierarchical Classification Over Time (HCOT) algorithm, which dynamically classifies product orders using top-down, non-mandatory leaf-node prediction. To enforce a blocking approach with respect to the tree-based class hierarchy, we introduce the Certainty-based Automated Thresholds (CAT) algorithm, which automatically computes optimal thresholds at each node in the hierarchy. The resulting CAT-HCOT algorithm has the ability to provide both accurate and timely predictions while specifically tailoring the hierarchical classification approach to the domain of customer satisfaction in online retailing. CAT-HCOT obtains a predictive accuracy of 94%. In terms of timeliness, CAT-HCOT classifies 40% of product orders on the order date itself, 80% of product orders within five days after the order date, and 100% of product orders after 10 days. In contrast, a static classification approach is unable to capture the accuracy-timeliness trade-off. Also, CAT-HCOT outperforms a flat, one-level hierarchy baseline method in terms of hierarchical precision and recall scores.

## 1. Data_Cleaning_Preparation.ipynb

This file includes the code which is used to clean and prepare the data sets received from the online retailer. Firstly, we rename some of the columns in this data set. Secondly, we remove noisy observations from the data (as is explained in Appendix B in the research paper). Thirdly, we provide code for variables that we created ourselves. Fourth and last, code for creating Figure 1 in the research paper (determinant availability) is provided. Note that the paths where the data files from the retailer originate and to which the cleaned and prepared data should be written should be specified by the user.

## 2. dynamic_input_data.py

This file includes functions which are used to create the input data (belonging to dynamic variables) for the HCOT algorithm. 

- addKnownColumns(df,X): add the dynamic variables accociated with the match determinants.
- addProductColumns(df,X): add dynamic variables accociated with products. This function makes use of two supporting functions: 'addProductColumns0' and 'addProductColumnsX' based on the number of days after the order date that the dynamic variables should be computed for.
- addSellerColumns(df,X): add dynamic variables accociated with sellers.  This function makes use of two supporting functions: 'addSellerColumns0' and 'addSellerColumnsX' based on the number of days after the order date that the dynamic variables should be computed for.
- addHistoricPerformance(df, variable = 'transporterCode', X = 0): add dynamic variables associated with historic performance. This function creates 3 columns with the historic fraction of orders belonging to Happy, Unhappy and Unknown matches.
- dataX(df, DATE, X_col, y_col, historic_variable, days): main function for creating input data _X_ (explanatory) and _y_ (dependent), computed _days_ days after the date of ordering.
- initialiseData(): return a data frame with data as well as feature names, historic variables and date information. This function is used to initialise the data and create lists of variable names that will be used in subsequent code files.

## 3. Seminar_Hierarchical_Classifiers.ipynb

This file includes the main code that we use to define CAT and HCOT, perform validation to obtain optimal hierarchies, implement CAT-HCOT, and apply the two baseline methods. Thus, running all parts in this single file allows the user to perform validation and testing for both CAT-HCOT and the two baselines. To do so, the following steps should be followed.

1. Import packages: import the necessary packages.
2. Load Data: load the data as created in the 'data_cleaning_preparation' code by running the initialiseData() function
3. Functions: run the functions provided at the bottom of the code.
4. Validation (flat): run the code in this section to perform validation for the flat CAT-HCOT baseline.
5. Validation (HCOT): run the code in this section to perform validation for CAT-HCOT and the static HCOT baseline. Make sure to first run the functions contained in this folder before running the output.
6. CAT-HCOT: run the code in this folder to perform testing following CAT-HCOT.
7. Base Case 1 (Static): run the code in this folder to perform testing following the static HCOT baseline.
8. Base Case 2 (Flat): run the code in this folder to perform testing following the flat CAT-HCOT baseline.

Note that the functions relevant in this file can be divided into different groups, which are specifically presented in the 'functions' folder. These functions are all included in the 'Seminar_Hierarchical_Classifiers' file, so that they do not contain extra/new information. The division we provide is purely for understanding purposes. To provide a little bit more explanation on these separate functions, we dive into this division below.

### validation.py

This file includes the code for applying the validation procedure that we use for CAT-HCOT and the two baseline approaches. Note that the validation procedure for CAT-HCOT is the same as that for the static baseline, since our hierarchies are optimised on a daily basis. The validation procedure evaluates all possible classifier combinations at parent nodes in the respective daily hierarchy, optimises the hyperparameters belonging to each combination, and outputs the respective performance. From the results, the best classifier combinations (per day) and their respective hyperparameters are obtained. These can subsequently be used for testing purposes. The validation code consists of the following functions:
- HCOT_optimisation(): optimisation of a single hierarchy with pre-defined set of possible combination which includes the functions:
  - get_hyperspace(combination): create hyperspace for optimisation (currently supports DT, RF, NN and LR).
  - clf_hypers(params): function which distributes the hyperparameters over the 3 specific parent nodes in the hierarchy.
  - objective_function(params): objective for optimising daily hierarchical classifier combination in HCOT.
  - hyperopt(param_space, X_train, y_train, X_val, y_val, num_eval): actual optimisation function for a single hierarchy with pre-set classifier combination.
- flat_HCOT_optimisation(): optimisation of single 'one-level hierarchy' of the flat HCOT basline which includes similar functions (supporting LR + RF):
  - flat_get_hyperspace(combination): create hyperspace for optimisation.
  - flat_objective_function(params): objective for optimising either LR or RF.
  - flat_hyperopt(param_space, X_train, y_train, X_val, y_val, num_eval): actual optimisation function.

### CAT.py

This file includes the CAT algorithm that we apply to CAT-HCOT and the flat CAT-HCOT baseline. Note that a single run of the CAT algorithm computes a node- and day-specific threshold. Implementing CAT in our framework thus requires multiple runs of the CAT algorithm (for each node on each day with NMLNP). The code file consists of two main functions: 
- opt_threshold(probs, node, day, certainty, steps = 100): CAT for HCOT, using the hierarchy from the research paper.
- flat_thresholds(probs, node, day, certainty, steps = 100): CAT for flat CAT-HCOT, using the flat hierarchy from the research paper.

### HCOT.py

This file includes the HCOT algoritm together with functions to test and evaluate the performance of this algorithm. The code in this file applies thresholds obtained via the CAT algorithm and works with the three-level tree-based class hierarchy for product orders as defined in the research paper and as optimised in the validation phase. Therefore, when implementing this code, we apply CAT-HCOT as defined in the research paper. The code in this file consists of the following functions: 
- ClassHierarchy(root): Class with all functions to construct a hierarchy which is thereafter used for HierarchicalClassifier, main function:
  - add_node(children, parent): function to construct the hiearchy. By calling this function, you can create a children-parent relationship in the tree.
- HierarchicalClassifier(class_hierarchy): Class with all functions to train and test a hierarchical classifier, the main functions are:
  - fit_classifiers(classifiers): function to fit the classifier per parent node, the input is a dictionary with the parent nodes as keys and values the classifiers.
  - fit(X, y): Train the hierarchy with X and Y. The classifiers at each parent node are trained with the acutal training labels.
  - predict(X): predict the values of X with the fitted hierarchy. The test instances go through the hierarchy in a top-down fashion with mandatory-leaf-node prediction
  - predict_proba(X, threshold): predict the values of X by making use of a pre-defined threshold for each node, if threshold not reached then blocked.
  - predict_proba2(X, THRESHOLDS): predict the values of X by making use of node-specific thresholds (THRESHOLDS), if threshold not reached then blocked.
  - get_probabilities(X, y): get the class probabilities for each instance, this is used for training the thresholds with CAT.
- dynamicHierarchicalClassifier(START, END): CAT-HCOT algorithm to test instances over a period from START till END.
- get_performance(...): function included in dynamicHierarchicalClassifier to compute daily/global performance measures while running HCOT.
- precision_score_ancestors(class_hierarchy, y_true, y_pred): hierarchical precision score, input is the class_hierarchy from ClassHierarchy.
- recall_score_ancestors(class_hierarchy, y_true, y_pred): hierarchical recall score, input is the class_hierarchy from ClassHierarchy.
- f1_score_ancestors(class_hierarchy, y_true, y_pred, beta): hierarchical f1 score, input is the class_hierarchy from ClassHierarchy.

### baseline.py

This file includes code for the two baseline methods: static HCOT and flat CAT-HCOT. For the static baseline, we predict all test instances on a daily basis, whereas the flat baseline performs CAT-HCOT in a flat (one-level) hierarchy. The static baseline applies the optimal hierarchies obtained via the validation procedure for CAT-HCOT, while the flat baseline uses the optimised hierarchies from the flat validation procedure (both are defined in 'validation.py'). The code in this file consists of two functions:
- staticHierarchicalClassifier(START, END): static HCOT (no blocking), hence predicting all instances on a single point in time.
- dynamicFlatClassifier(START, END): flat CAT-HCOT (one-level hierarchy), using similar blokking approach as in CAT-HCOT.

Again, note that all functions from the 'validation.py', 'CAT.py', 'HCOT.py', and 'baseline.py' files are directly included in the 'Seminar_Hierarchical_Classifiers.ipynb' file, such that simply running the latter file allows us to perform all required validation and testing processes.

If any questions or comments occur, please ask them via this GitHub page (https://github.com/thomas-brink/Seminar-QM-BA).
