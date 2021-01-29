#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from datetime import timedelta

def addKnownColumns(df,X):
    """
    Function to create columns which indicate whether determinants are known after X days.
    Input: X = number of days after order date at which the prediction is made
           df = dataFrame
    """
    df_ = df[['actualDeliveryDays','onTimeDelivery','shipmentDays','transporterCodeGeneral']].values
    
    df['caseKnownX']           = df['caseDays'].apply(lambda x: True if x <= X else False)
    df['returnKnownX']         = df['returnDays'].apply(lambda x: True if x <= X else False)
    df['cancellationKnownX']   = df['cancellationDays'].apply(lambda x: True if x <= X else False)
    
    df['onTimeDeliveryKnownX'] = ((df_[:,0] <= X) & (df_[:,1] == True))
    df['lateDeliveryKnownX'] = ((df_[:,0] <= X) & (df_[:,1] == False))
    
    # df['onTimeDeliveryKnownX'] = df.apply(lambda row: True if ((row.actualDeliveryDays <= X) and (row.onTimeDelivery == True)) else False, axis = 1)
    # df['lateDeliveryKnownX']   = df.apply(lambda row: True if ((row.actualDeliveryDays <= X) and (row.onTimeDelivery == False)) else False, axis = 1)
    
    for transporter in df['transporterCodeGeneral'].unique():
        dummyColumn = 'transporter' + transporter +'/X'
        df[dummyColumn] = ((df_[:,2] <= X) & (df_[:,3] == transporter))
        # df[dummyColumn] = df.apply(lambda row: True if ((row.shipmentDays <= X) and (row.transporterCodeGeneral == transporter)) else False, axis = 1)

    return df


def addProductColumns(df,X):
    
    if ['productOrderCount0','productTotalCount0','productTotalReturned0','productReturnFraction0'] not in list(df.columns):
    
        df = addProductColumns0(df)
    
    if X > 0:
        
        df = addProductColumnsX(df,X)
        
    else:
        
        df['productOrderCountX'] = df['productOrderCount0']
        df['productTotalCountX'] = df['productTotalCount0']
        df['productTotalReturnedX'] = df['productTotalReturned0']
        df['productReturnFractionX'] = df['productReturnFraction0']
        
    return df


def addSellerColumns(df,X):
    
    if 'sellerDailyOrders0' not in list(df.columns):
    
        df = addSellerColumns0(df)
    
    if X > 0:
        
        df = addSellerColumnsX(df,X)
    
    else:
        
        df['sellerDailyOrdersX'] = df['sellerDailyOrders0']
        
    return df


def addProductColumns0(df): 
    """
    Function to add 4 columns: productOrderCount, productTotalCount, productTotalReturned and productReturnFraction.
    Input: dataFrame with columns: 'productId','orderDate','quantityOrdered','quantityReturned','returnDateTime'.
    """
    df = df.sort_values(by = ['productId','orderDate'])
    df = df.reset_index(drop = True)
    
    df_ = df[['productId','orderDate','quantityOrdered','quantityReturned','returnDateTime']]
    
    #ProductTotalCount
    pivot = df_.groupby(['productId','orderDate']).quantityOrdered.sum().groupby('productId').cumsum()
    productTotalCount = df_.merge(pivot, 
                                left_on=['productId','orderDate'], 
                                right_index=True, 
                                how = 'left').quantityOrdered_y
    
    #ProductOrderCount
    pivot = df_.groupby(['productId','orderDate']).quantityOrdered.count().groupby('productId').cumsum()
    productOrderCount = df_.merge(pivot, 
                                left_on=['productId','orderDate'], 
                                right_index=True, 
                                how = 'left').quantityOrdered_y
    
    #ProductTotalReturned
    productTotalReturned = np.zeros(df_.shape[0])
    
    previousID = None
    
    returnDic = {}
    
    for row in df_.itertuples(): #iterate through dataFrame: row[0] = index, row[1] = productId, row[2] = orderDate
                                                           # row[3] = quantityOrdered, row[4] = quantityReturned
        if row[0] == 0:                                    # row[5] = returnDateTime
            
            #update return dictionary if this product is returned
            if np.isnan(row[4]) == True:
                if row[5] in returnDic:
                    returnDic[row[5]] += row[4]
                else:
                    returnDic[row[5]] = row[4]

            previousID = row[1]
            
        elif (previousID == row[1]):
            
            #update return dictionary if this product is returned
            if np.isnan(row[4]) == True:
                if row[5] in returnDic:
                    returnDic[row[5]] += row[4]
                else:
                    returnDic[row[5]] = row[4]
            
            #add returned products to new dictionary if known
            known = {k: v for k, v in returnDic.items() if k <= row[2]}
            productTotalReturned[row[0]] = sum(known.values())
            
            #update the dictionary by removing the returns which are now known
            returnDic = {k: v for k, v in returnDic.items() if k > row[2]}
                        
            previousID = row[1]
            
        else:
            returnDic = {} #new productId, hence empty the return dictionary
            
            #update return dictionary if this product is returned
            if np.isnan(row[4]) == True:
                if row[5] in returnDic:
                    returnDic[row[5]] += row[4]
                else:
                    returnDic[row[5]] = row[4]
                    
            previousID = row[1]
    
    df_['productTotalReturned'] = productTotalReturned
    pivot = df_.groupby(by = ['productId','orderDate']).productTotalReturned.sum().groupby('productId').cumsum()
    productTotalReturned = df_.merge(pivot, 
                                left_on=['productId','orderDate'], 
                                right_index=True, 
                                how = 'left').productTotalReturned_y
     
    #Add new columns to dataFrame    
    df['productOrderCount0'] = productOrderCount
    df['productTotalCount0'] = productTotalCount
    df['productTotalReturned0'] = productTotalReturned
    df['productReturnFraction0'] = productTotalReturned / productTotalCount
    
    return(df)


def addProductColumnsX(df,X):
    """
    Function to add 4 columns: productOrderCountX, productTotalCountX, productTotalReturnedX and productReturnFractionX.
    Input: dataFrame with columns: 'productId','orderDate','productOrderCount','productTotalCount','productTotalReturned'
    """
    df = df.sort_values(by = ['productId','orderDate'], ascending = [True, False]) #reverse ordering on Orderdate!
    df = df.reset_index(drop = True)
    
    df_ = df[['productId','orderDate','productOrderCount0','productTotalCount0','productTotalReturned0']]
    #            row[1]       row[2]        row[3]               row[4]                 row[5]    
    
    df_['orderDateX'] = df_['orderDate'] + timedelta(X)
    #      row[6]

    knownProductInfo = np.zeros((df_.shape[0],3))
    
    previousID = None
    previousMaxDate = None
    
    dic = {}
    
    for row in df_.itertuples(): #iterate  
                                                                  
        if row[0] == 0:                                          
            
            knownProductInfo[[row[0]]] = (row[3],row[4],row[5]) 
            
            dic[row[2]] = (row[3],row[4],row[5])

            previousMaxDate = row[2]
            previousID = row[1]
            
        elif (previousID == row[1]):
            
            if row[6] >= previousMaxDate:
                dic[row[2]] = (row[3],row[4],row[5])
                knownProductInfo[[row[0]]] = dic[max(dic)]
            else:
                dic[row[2]] = (row[3],row[4],row[5])
                dic = {k: v for k, v in dic.items() if k <= row[6]}
                
                knownProductInfo[[row[0]]] = dic[max(dic)]
                previousMaxDate = max(dic)
                 
            previousID = row[1]
            
        else:
            dic = {} #new productId -> empty the dictionary
            
            knownProductInfo[[row[0]]] = (row[3],row[4],row[5])
            dic[row[2]] = (row[3],row[4],row[5])
                    
            previousMaxDate = row[2]
            previousID = row[1]

    df['productOrderCountX'] = knownProductInfo[:,0]
    df['productTotalCountX'] = knownProductInfo[:,1]
    df['productTotalReturnedX'] = knownProductInfo[:,2]
    df['productReturnFractionX'] = knownProductInfo[:,2] / knownProductInfo[:,1]
    
    #Reverse to natural order
    df = df.sort_values(by = ['productId','orderDate'], ascending = [True, True])
    df = df.reset_index(drop = True)
    
    return(df)


def addSellerColumns0(df):
    """
    Function to add 4 columns: 
    Input: dataFrame with columns: 'sellerId','orderDate','quantityOrdered','partnerSellingDays'
    """
    df = df.sort_values(by = ['sellerId','orderDate'])
    df = df.reset_index(drop = True)
    
    df_ = df[['sellerId','orderDate','quantityOrdered','partnerSellingDays']]
    
    firstOrder = df_.groupby('sellerId').orderDate.min()
    df_['firstOrder'] = df_.merge(firstOrder,
                                  left_on = 'sellerId',
                                  right_index = True,
                                  how = 'left').orderDate_y
    df_['daysFirstOrder'] = (df_['orderDate'] - df_['firstOrder']).dt.days + 1
    
    pivot = df_.groupby(['sellerId','orderDate']).quantityOrdered.count().groupby('sellerId').cumsum()
    sellerTotalCount = df_.merge(pivot, 
                                left_on=['sellerId','orderDate'], 
                                right_index=True, 
                                how = 'left').quantityOrdered_y
    
    df['sellerDailyOrders0'] = np.log(sellerTotalCount / df_['daysFirstOrder'])
    
    return df


def addSellerColumnsX(df,X):
    """
    Function to add 4 columns: 
    Input: dataFrame with columns: 'sellerId','orderDate','quantityOrdered','partnerSellingMonths'
    """
        
    df = df.sort_values(by = ['sellerId','orderDate'], ascending = [True, False]) #reverse ordering orderdate!
    df = df.reset_index(drop = True)

    df_ = df[['sellerId','orderDate','sellerDailyOrders0']]
    #            row[1]       row[2]        row[3]        

    df_['orderDateX'] = df_['orderDate'] + timedelta(X)
    #      row[4]

    knownSellerInfo = np.zeros(df_.shape[0])

    previousID = None
    previousMaxDate = None

    dic = {}

    for row in df_.itertuples(): #iterate  

        if row[0] == 0:                                          

            knownSellerInfo[[row[0]]] = row[3]

            dic[row[2]] = row[3]

            previousMaxDate = row[2]
            previousID = row[1]

        elif (previousID == row[1]):

            if row[4] >= previousMaxDate:
                dic[row[2]] = row[3]
                knownSellerInfo[[row[0]]] = dic[max(dic)]
            else:
                dic[row[2]] = row[3]
                dic = {k: v for k, v in dic.items() if k <= row[4]}

                knownSellerInfo[[row[0]]] = dic[max(dic)]
                previousMaxDate = max(dic)

            previousID = row[1]

        else:
            dic = {} #new productId -> empty the dictionary

            knownSellerInfo[[row[0]]] = row[3]
            dic[row[2]] = row[3]

            previousMaxDate = row[2]
            previousID = row[1]

    df['sellerDailyOrdersX'] = knownSellerInfo

    #Reverse to natural order
    df = df.sort_values(by = ['sellerId','orderDate'], ascending = [True, True])
    df = df.reset_index(drop = True)

    return df


def dataX(df, DATE, X_col, y_col, days):
    """
    Function to return features and output labels for a pre-specified number of days after orderDate. 
    
    Input:
    - df: dataframe containing all features available at the time of ordering.
    - DATE: column list of date variables that should be ordered on.
    - X_col: list of column names that are to be used as features.
    - y_col: list containing the output variable (e.g. binaryMatchClassification)
    - days: integer number of days after orderDate that should be considered.
    
    Output: 
    - X: dataframe output of features that can be used the number of days after orderDate. E.g. information on cases and deliveries are added.
    - y: dataframe output of output labels that can be used the number of days after orderDate.
    """    
    
    df = functions.addKnownColumns(df,days)
    df = functions.addProductColumns(df,days)
    df = functions.addSellerColumns(df,days)
    
    df = df[DATE + X_col + Y_col]
    
    df = df.dropna()
    df = df.sort_values(by = 'orderDate')
    df = df.reset_index(drop = True)
    
    X = df[X_col]
    y = df[Y_col[0]]
    
    return(X,y)


def neuralNetwork():
    
    model = Sequential()

    model.add(Dense(units=25,activation='relu'))
    model.add(Dense(units=3,activation='softmax')) #units should equal number of labels
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

#estimator = KerasClassifier(build_fn = neuralNetwork, epochs = 20, class_weight = class_weights, verbose = 1)
#history = estimator.fit(X_train, y_train)
#pd.DataFrame(history.history).plot()


def classifyLabels(classifier, X, y, n, split = 'TimeSeries', smote = False, scale = None, days = 0):
    """
    Function to classify match labels using a pre-specified classifier with X and y variables. 
    
    Input:
    - classifier: can be any supported classifier. E.g. DecisionTreeClassifier(random_state=0, class_weight='balanced', max_depth=10). Necessary!
    - X: dataframe input on explanatory features. Necessary!
    - y: dataframe input on labels. Necessary!
    - n: number of folds to be evaluated.
    - split: object that can take value 'Random' to make K-fold random train/test split. Default is to apply time series split.
    - smote: boolean, if true Synthetic Minority Oversampling will be applied. Default = False.
    - scale: object that can take values 'MinMax' or 'Standard' to scale X correspondingly. Any other input will not scale X. Default = None.
    - days: integer number of days after orderDate that should be considered. Default = 0.
    
    Output: 
    - accuracy: list of accuracies for the n evaluated classifiers.
    - class_report: report of performance measures for the n evaluated classifiers.
    """
    
    accuracy = {}
    class_report = {}
    count = 1
    
    if split == 'Random':
        
        kf = StratifiedKFold(n_splits = n, random_state = 0, shuffle = True)
        for train_index, test_index in kf.split(X, y):

            if scale == 'MinMax':
                scaler = preprocessing.MinMaxScaler()
                X_scaled = pd.DataFrame(scaler.fit_transform(X))
                X_train, X_test = X_scaled.iloc[train_index], X_scaled.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            elif scale == 'Standard':
                scaler = preprocessing.StandardScaler()
                X_scaled = pd.DataFrame(scaler.fit_transform(X))
                X_train, X_test = X_scaled.iloc[train_index], X_scaled.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            else:
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            if smote == True:
                smote = SMOTE('not majority')
                X_train, y_train = smote.fit_sample(X_train,y_train)
            else:
                X_train, y_train = X_train, y_train
            
            clf = classifier
            clf = clf.fit(X_train,y_train)
            prediction = clf.predict(X_test)
            accuracy[count] = metrics.accuracy_score(y_test, prediction)
            class_report[count] = metrics.classification_report(y_test, prediction)
    
            print(count)
            count +=1
    
    else:
        
        tscv = TimeSeriesSplit(n_splits = n)
        
        for train_index, test_index in tscv.split(X):
        
            if scale == 'MinMax':
                scaler = preprocessing.MinMaxScaler()
                X_scaled = pd.DataFrame(scaler.fit_transform(X))
                X_train, X_test = X_scaled.iloc[train_index], X_scaled.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            elif scale == 'Standard':
                scaler = preprocessing.StandardScaler()
                X_scaled = pd.DataFrame(scaler.fit_transform(X))
                X_train, X_test = X_scaled.iloc[train_index], X_scaled.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            else:
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            if smote == True:
                smote = SMOTE('not majority')
                X_train, y_train = smote.fit_sample(X_train,y_train)
            else:
                X_train, y_train = X_train, y_train
            
            clf = classifier
            clf = clf.fit(X_train,y_train)
            prediction = clf.predict(X_test)
            accuracy[count] = metrics.accuracy_score(y_test, prediction)
            class_report[count] = metrics.classification_report(y_test, prediction)
    
            print(count)
            count +=1

    return(accuracy, class_report)


def classifyLabelsNew(classifier, X, y, n, split = 'TimeSeries', smote = False, scale = None, NN = False):

    labels = np.unique(y)
    int_label_mapping = dict(enumerate(labels))
    label_int_mapping = {y:x for x,y in int_label_mapping.items()}
    
    acc,pre,rec,f1,results = {},{},{},{},{}
        
    if split == 'Random':
        cv = StratifiedKFold(n_splits = n, random_state = 0, shuffle = True)   
    else:
        cv = TimeSeriesSplit(n_splits = n)
        
    if NN == True:
        y_encoded = y.map(label_int_mapping)
        y_dummy = pd.DataFrame(np_utils.to_categorical(y_encoded))
    
    count = 1

    for train_index, test_index in cv.split(X):
        
        if scale != None:
            X_scaled = pd.DataFrame(scaler.fit_transform(X))
            X_train, X_test = X_scaled.iloc[train_index], X_scaled.iloc[test_index]
        else:   
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        
        clf = classifier
        
        if NN == True:
            class_weights = class_weight.compute_class_weight('balanced',labels,y.iloc[train_index])
            class_weights = dict(enumerate(class_weights))
            clf.set_params(class_weight = class_weights)
            y_train, y_test = y_dummy.iloc[train_index], y.iloc[test_index]
        else:
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        
        if NN == True:
            y_pred = pd.Series(y_pred).map(int_label_mapping)
        
        accuracy = accuracy_score(y_test, y_pred)
        scores = precision_recall_fscore_support(y_test, y_pred, average = None, labels = labels, beta = 1)
        
        acc[count] = accuracy
        pre[count] = scores[0]
        rec[count] = scores[1]
        f1[count] = scores[2]
        
        count += 1

    results['accuracy'] = sum(acc.values()) / n
    
    for ix,label in enumerate(labels):
        results[('precision_'+label)] = (sum(pre.values()) / n)[ix]
        results[('recall_'+label)] = (sum(rec.values()) / n)[ix]
        results[('f1_'+label)] = (sum(f1.values()) / n)[ix]
     
    return results


def classifyLabelsQuick(classifier, X, y, n, split = 'TimeSeries', smote = False, scale = None, days = 0):
    
    accuracy = {}
    class_report = {}
    
    if split == 'Random':
        split_type = StratifiedKFold(n_splits = n, random_state = 0, shuffle = True)
        
    else:
        split_type = TimeSeriesSplit(n_splits = n)
    
    #Create pipeline -> everything in the pipeline is executed after eachother
    pipe = Pipeline([('scaler', preprocessing.MinMaxScaler()), ('classifier', classifier)])
    
    #Cross validation function which outputs the accuracy and average (unweighted) precision & recall of labels
    metrics = ('accuracy','precision_macro','recall_macro','f1_macro','f1_weighted')
    y_pred = cross_validate(pipe, X, y, cv = split_type, scoring = metrics, return_train_score = True)
        
    return y_pred