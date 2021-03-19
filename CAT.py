import pandas as pd
import numpy as np
from HCOT import *

def opt_threshold(probs, node, day, certainty, option = 2, steps = 100):
    '''
    CAT for CAT-HCOT
    '''
    
    if node == 1:
        probabilities_for = 'UNKNOWN'
        y_pos_filter_list = ['UNKNOWN']
        y_neg_filter_list = ['HAPPY', 'MILDLY UNHAPPY', 'MEDIUM UNHAPPY', 'HEAVILY UNHAPPY']
        level             = probs[['UNKNOWN', 'KNOWN']]
        majority_vote     = level[level['UNKNOWN'] > level['KNOWN']]['UNKNOWN']
    elif node == 2:
        probabilities_for = 'KNOWN'
        y_pos_filter_list = ['HAPPY', 'MILDLY UNHAPPY', 'MEDIUM UNHAPPY', 'HEAVILY UNHAPPY']
        y_neg_filter_list = ['UNKNOWN']
        level             = probs[['UNKNOWN', 'KNOWN']]
        majority_vote     = level[level['KNOWN'] > level['UNKNOWN']]['KNOWN']
    elif node == 3:
        probabilities_for = 'HAPPY'
        y_pos_filter_list = ['HAPPY']
        y_neg_filter_list = ['MILDLY UNHAPPY', 'MEDIUM UNHAPPY', 'HEAVILY UNHAPPY']
        level             = probs[probs.detailedMatchClassification.isin(['MILDLY UNHAPPY', 'MEDIUM UNHAPPY', 'HEAVILY UNHAPPY', 'HAPPY'])][['UNHAPPY', 'HAPPY']]
        majority_vote     = level[level['HAPPY'] > level['UNHAPPY']]['HAPPY']
    elif node == 4:
        probabilities_for = 'UNHAPPY'
        y_pos_filter_list = ['MILDLY UNHAPPY', 'MEDIUM UNHAPPY', 'HEAVILY UNHAPPY']
        y_neg_filter_list = ['HAPPY']
        level             = probs[probs.detailedMatchClassification.isin(['MILDLY UNHAPPY', 'MEDIUM UNHAPPY', 'HEAVILY UNHAPPY', 'HAPPY'])][['UNHAPPY', 'HAPPY']]
        majority_vote     = level[level['UNHAPPY'] > level['HAPPY']]['UNHAPPY']
    elif node == 5:
        probabilities_for = 'MILDLY UNHAPPY'
        y_pos_filter_list = ['MILDLY UNHAPPY']
        y_neg_filter_list = ['MEDIUM UNHAPPY', 'HEAVILY UNHAPPY']
        level             = probs[probs.detailedMatchClassification.isin(['MILDLY UNHAPPY', 'MEDIUM UNHAPPY', 'HEAVILY UNHAPPY'])][['MILDLY UNHAPPY', 'MEDIUM UNHAPPY', 'HEAVILY UNHAPPY']]
        majority_vote     = level[ (level['MILDLY UNHAPPY'] > level['MEDIUM UNHAPPY']) & (level['MILDLY UNHAPPY'] > level['HEAVILY UNHAPPY']) ]['MILDLY UNHAPPY']
    elif node == 6:
        probabilities_for = 'MEDIUM UNHAPPY'
        y_pos_filter_list = ['MEDIUM UNHAPPY']
        y_neg_filter_list = ['MILDLY UNHAPPY', 'HEAVILY UNHAPPY']
        level             = probs[probs.detailedMatchClassification.isin(['MILDLY UNHAPPY', 'MEDIUM UNHAPPY', 'HEAVILY UNHAPPY'])][['MILDLY UNHAPPY', 'MEDIUM UNHAPPY', 'HEAVILY UNHAPPY']]
        majority_vote     = level[ (level['MEDIUM UNHAPPY'] > level['MILDLY UNHAPPY']) & (level['MEDIUM UNHAPPY'] > level['HEAVILY UNHAPPY']) ]['MEDIUM UNHAPPY']
    elif node == 7:
        probabilities_for = 'HEAVILY UNHAPPY'
        y_pos_filter_list = ['HEAVILY UNHAPPY']
        y_neg_filter_list = ['MILDLY UNHAPPY', 'MEDIUM UNHAPPY']
        level             = probs[probs.detailedMatchClassification.isin(['MILDLY UNHAPPY', 'MEDIUM UNHAPPY', 'HEAVILY UNHAPPY'])][['MILDLY UNHAPPY', 'MEDIUM UNHAPPY', 'HEAVILY UNHAPPY']]
        majority_vote     = level[ (level['HEAVILY UNHAPPY'] > level['MEDIUM UNHAPPY']) & (level['HEAVILY UNHAPPY'] > level['MILDLY UNHAPPY']) ]['HEAVILY UNHAPPY']
    else:
        raise Exception('''Error: undefined node has been passed. Node options (integer input):
                           1: Unknown
                           2: Known
                           3: Happy
                           4: Unhappy
                           5: Mildly Unhappy
                           6: Medium Unhappy
                           7: Heavily Unhappy''')
    
    y_pos = probs[probs.detailedMatchClassification.isin(y_pos_filter_list)][probabilities_for] 
    y_neg = probs[probs.detailedMatchClassification.isin(y_neg_filter_list)][probabilities_for]
    
    if option == 1:
        y_pos = y_pos[y_pos > min(majority_vote)]
        y_neg = y_neg[y_neg > min(majority_vote)]
    elif option == 2:
        y_pos = y_pos[y_pos.index.isin(majority_vote.index)]
        y_neg = y_neg[y_neg.index.isin(majority_vote.index)]
    else:
        raise Exception('''Error: undefined threshold option has been passed. Threshold options (integer input):
                           1: Consider all probabilities >= min(majority vote)
                           2: Only consider probabilities that are the majority vote''')
    
    # Potential thresholds
    V = np.concatenate((y_pos, y_neg))
    V = np.unique(V) # np.unique() also sorts
    
    if len(y_neg) > 0:
        lowerbound = np.percentile(y_neg, (certainty*100))
    else:
        lowerbound = V.min()
    
    V = V[V >= lowerbound] #define allowed search space
    
    S = np.linspace(V.min(), V.max(), steps)
    
    thresholds = pd.DataFrame({'threshold'     : [0]*steps,
                               'F_score'       : [0]*steps})
    
    for i in range(steps):        
        threshold = S[i]       
        beta      = 1
        positives = len(y_pos[y_pos >= threshold])  
        negatives = len(y_neg[y_neg >= threshold])  
        recall    = positives / len(y_pos)
        precision = positives / (positives + negatives)

        thresholds.loc[i, 'threshold']       = threshold
        thresholds.loc[i, 'F_score']         = ((beta ** 2 + 1) * precision * recall) / ((beta ** 2 * precision) + recall) if ((beta ** 2 * precision) + recall) != 0 else 0
        
    F_score         = thresholds['F_score'].max()
    opt_index       = thresholds['F_score'].argmax()
    threshold       = thresholds.loc[opt_index, 'threshold']
    
    return(probabilities_for, threshold)

def flat_thresholds(probs, node, day, certainty, steps = 100):
    '''
    CAT for flat CAT-HCOT
    '''
    
    NODES = set(['UNKNOWN', 'HAPPY', 'MILDLY UNHAPPY', 'MEDIUM UNHAPPY', 'HEAVILY UNHAPPY'])
    
    if node == 1:   NODE = 'UNKNOWN'
    elif node == 2: NODE = 'HAPPY'
    elif node == 3: NODE = 'MILDLY UNHAPPY'
    elif node == 4: NODE = 'MEDIUM UNHAPPY'
    elif node == 5: NODE = 'HEAVILY UNHAPPY'
    else:
        raise Exception('''Error: undefined node has been passed. Node options (integer input):
                           1: Unknown
                           2: Happy
                           3: Mildly Unhappy
                           4: Medium Unhappy
                           5: Heavily Unhappy''')
        
    probabilities_for = NODE
    y_pos_filter_list = [NODE]
    y_neg_filter_list = list(NODES - {NODE})
    majority_vote     = probs[probs[NODE] > probs[list(NODES - {NODE})].max(axis=1)][NODE]
        
    y_pos = probs[probs.detailedMatchClassification.isin(y_pos_filter_list)][probabilities_for] 
    y_neg = probs[probs.detailedMatchClassification.isin(y_neg_filter_list)][probabilities_for]
    
    y_pos = y_pos[y_pos.index.isin(majority_vote.index)]
    y_neg = y_neg[y_neg.index.isin(majority_vote.index)]
 
    if len(y_neg) > 0:
        lowerbound = np.percentile(y_neg, (certainty*100))
    else:
        lowerbound = V.min()

    V = np.concatenate((y_pos, y_neg))
    V = np.unique(V) # np.unique() also sorts    
    V = V[V >= lowerbound] #define allowed search space
    S = np.linspace(V.min(), V.max(), steps)
    
    thresholds = pd.DataFrame({'threshold'     : [0]*steps,
                               'F_score'       : [0]*steps})
    
    for i in range(steps):        
        threshold = S[i] 
        beta      = 1
        positives = len(y_pos[y_pos >= threshold])
        negatives = len(y_neg[y_neg >= threshold])
        recall    = positives / len(y_pos)
        precision = positives / (positives + negatives)

        thresholds.loc[i, 'threshold']       = threshold
        thresholds.loc[i, 'F_score']         = ( (1 + (beta**2)) * precision * recall ) / ( (beta**2) * precision + recall ) if ( (beta**2) * precision + recall ) != 0 else 0
        
    
    F_score         = thresholds['F_score'].max()
    opt_index       = thresholds['F_score'].argmax()
    threshold       = thresholds.loc[opt_index, 'threshold']                           

    return(probabilities_for, threshold)