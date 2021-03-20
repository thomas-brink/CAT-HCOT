import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from datetime import date, timedelta, datetime
import time

from sklearn import preprocessing
from sklearn.utils import class_weight
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

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
    df['lateDeliveryKnownX']   = ((df_[:,0] <= X) & (df_[:,1] == False))
    
    for transporter in df['transporterCodeGeneral'].unique():
        dummyColumn = 'transporter' + transporter +'/X'
        df[dummyColumn] = ((df_[:,2] <= X) & (df_[:,3] == transporter))

    return df


def addProductColumns(df,X):
    '''
    Add dynamic variables accociated with products. This function makes use of two supporting functions: 'addProductColumns0' and 'addProductColumnsX' based on the number of days after the order date that the dynamic variables should be computed for.
    '''
    
    if ['productOrderCount0','productTotalCount0','productTotalReturned0','productReturnFraction0'] not in list(df.columns):
    
        df = addProductColumns0(df)
    
    if X > 0:
        
        df = addProductColumnsX(df,X)
        
    else:
        
        df['productOrderCountX']     = df['productOrderCount0']
        df['productTotalCountX']     = df['productTotalCount0']
        df['productTotalReturnedX']  = df['productTotalReturned0']
        df['productReturnFractionX'] = df['productReturnFraction0']
        
    return df


def addSellerColumns(df,X):
    '''
    Add dynamic variables accociated with sellers.  This function makes use of two supporting functions: 'addSellerColumns0' and 'addSellerColumnsX' based on the number of days after the order date that the dynamic variables should be computed for.
    '''
    
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
            if np.isnan(row[4]) == False:
                if row[5] in returnDic:
                    returnDic[row[5]] += row[4]
                else:
                    returnDic[row[5]] = row[4]

            previousID = row[1]
            
        elif (previousID == row[1]):
            
            #update return dictionary if this product is returned
            if np.isnan(row[4]) == False:
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
            if np.isnan(row[4]) == False:
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
    df['productOrderCount0']     = productOrderCount
    df['productTotalCount0']     = productTotalCount
    df['productTotalReturned0']  = productTotalReturned
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

    df['productOrderCountX']     = knownProductInfo[:,0]
    df['productTotalCountX']     = knownProductInfo[:,1]
    df['productTotalReturnedX']  = knownProductInfo[:,2]
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


def addHistoricPerformance(df, variable = 'transporterCode', X = 0): 
    """
    Function to add 3 columns: '[variable]HistoricHappyX', '[variable]HistoricUnhappyX', '[variable]HistoricUnknownX'.
    Input: dataFrame with columns: variable*,'orderDate','generalMatchClassification'.
    
    Input for X is how many days after the order date that the prediction is made. Default is immediately after the order, i.e., X = 0.
    
    * Variable can be any descriptive variable, e.g., 'sellerId', 'transporterCode', 'productGroup'. Default is 'transporterCode'.
    """
    # Check if the variables already exist. If so, drop them.
    if variable+'HistoricHappyX'   in list(df.columns): df = df.drop([variable+'HistoricHappyX'],   axis=1)
    if variable+'HistoricUnhappyX' in list(df.columns): df = df.drop([variable+'HistoricUnhappyX'], axis=1)
    if variable+'HistoricUnknownX' in list(df.columns): df = df.drop([variable+'HistoricUnknownX'], axis=1)   
    
    # Correct sorting
    df = df.sort_values(by = [variable,'orderDate'])
    df = df.reset_index(drop = True)
    
    # Prep the needed dataset
    one_hot      = pd.get_dummies(df['generalMatchClassification'])
    df           = df.join(one_hot)
    df[variable] = df[variable].fillna('UNKNOWN') #So far only for transporterCode, hence the datatype string. If that changes, this needs to be updated as well
    
    # STEP 1: Add the prediction and finalized dates
    df['dateFinal']      = df['orderDate'] + timedelta(days = 30)
    df['predictionDate'] = df['orderDate'] + timedelta(days = X)

    # STEP 2: Gather all dates for which you know (dateFinal) and need to know (predictionDate) something, sort and drop duplicates
    dates = pd.concat([df[[variable, 'predictionDate']].rename(columns={'predictionDate': 'date'}), df[[variable, 'dateFinal']].rename(columns={'dateFinal': 'date'})])
    dates = dates.sort_values([variable, 'date'])
    dates = dates.drop_duplicates(keep = 'first')
    dates = dates.reset_index(drop = True)

    # STEP 3: Join the table with orders (df) left on the table with dates (dates) (match dateFinal (df) to date (dates))
    dic = pd.merge(left = dates, right = df[[variable, 'dateFinal', 'HAPPY', 'UNHAPPY', 'UNKNOWN']], how = 'left', left_on = [variable, 'date'], right_on = [variable, 'dateFinal'])
    dic = dic.drop('dateFinal', axis=1)
    dic = dic.rename(columns={'date': 'dateFinal'})
    dic = dic.fillna(0)

    # STEP 4: Per sellerId find the cumulative sum of happy, unhappy, unknown, and total orders over time
    NO_happy_orders   = dic.groupby([variable,'dateFinal']).HAPPY.sum().groupby(variable).cumsum().reset_index(drop = True)
    NO_unknown_orders = dic.groupby([variable,'dateFinal']).UNKNOWN.sum().groupby(variable).cumsum().reset_index(drop = True)
    NO_unhappy_orders = dic.groupby([variable,'dateFinal']).UNHAPPY.sum().groupby(variable).cumsum().reset_index(drop = True)
    NO_orders         = NO_happy_orders + NO_unknown_orders + NO_unhappy_orders

    # STEP 5: The NO_[type of order] (e.g., number of happy orders) variables are without duplicates; put into dictionary
    dic = dic[[variable, 'dateFinal']]
    dic = dic.drop_duplicates(keep = 'first')
    dic = dic.reset_index(drop = True)
    dic = dic.rename(columns={variable: variable+'_', 'dateFinal': 'dateFinal_'})

    dic[variable+'HistoricHappyX']   = round((NO_happy_orders  /NO_orders)*100, 2)
    dic[variable+'HistoricUnhappyX'] = round((NO_unhappy_orders/NO_orders)*100, 2)
    dic[variable+'HistoricUnknownX'] = round((NO_unknown_orders/NO_orders)*100, 2)

    dic = dic.fillna(0)

    # STEP 6: Join the dictionary (dic; on dateFinal) with the orders (df; on predictionDate)
    df = df.drop(['dateFinal', 'HAPPY', 'UNHAPPY', 'UNKNOWN'], axis = 1)

    df = pd.merge(left = df, right = dic, how = 'left', left_on = [variable, 'predictionDate'], right_on = [variable+'_', 'dateFinal_'])
    df = df.drop([variable+'_', 'dateFinal_', 'predictionDate'], axis = 1)
     
    return(df)


def dataX(df, DATE, X_col, y_col, historic_variable, days):
    """
    Function to return features and output labels for a pre-specified number of days after orderDate. 
    
    Input:
    - df: dataframe containing all features available at the time of ordering.
    - DATE: column list of date variables that should be ordered on.
    - X_col: list of column names that are to be used as features.
    - y_col: list containing the output variable (e.g. binaryMatchClassification)
    - historic_variable: list containing the variables for which historic performance should be used
    - days: integer number of days after orderDate that should be considered.
    
    Output: 
    - X: dataframe output of features that can be used the number of days after orderDate. E.g. information on cases and deliveries are added.
    - y: dataframe output of output labels that can be used the number of days after orderDate.
    """    
    
    df = addKnownColumns(df,days)
    df = addProductColumns(df,days)
    df = addSellerColumns(df,days)
    for i in range(len(historic_variable)):
        df = addHistoricPerformance(df,historic_variable[i],days)
    
    df = df[DATE + X_col + y_col]
    
    df = df.dropna()
    df = df.sort_values(by = 'orderDate')
    df = df.reset_index(drop = True)
    
    X = df[X_col]
    y = df[y_col]
    
    return(X,y)


def initialiseData():
    '''
    Return a data frame with data as well as feature names, historic variables and date information. This function is used to initialise the data and create lists of variable names that will be used in subsequent code files.
    '''

    # Read in cleaned and prepared data file (.csv) that is created in the data_cleaning_preparation code
    #df = pd.read_csv('path...', low_memory = True)
    df = pd.read_csv('/Users/thoma/Documents/seminar_data/cleaned_prepared_data.csv')
    
    df['orderDate']                   = pd.to_datetime(df['orderDate'])
    df['cancellationDate']            = pd.to_datetime(df['cancellationDate'])
    df['promisedDeliveryDate']        = pd.to_datetime(df['promisedDeliveryDate'])
    df['shipmentDate']                = pd.to_datetime(df['shipmentDate'])
    df['dateTimeFirstDeliveryMoment'] = pd.to_datetime(df['dateTimeFirstDeliveryMoment'])
    df['startDateCase']               = pd.to_datetime(df['startDateCase'])
    df['returnDateTime']              = pd.to_datetime(df['returnDateTime'])
    df['registrationDateSeller']      = pd.to_datetime(df['registrationDateSeller'])

    # Fixed Columns:
    DATE = ['orderDate']
    BASIC = ['totalPrice','quantityOrdered','fulfilmentByPlatform','countryCodeNL','countryOriginNL','countryOriginBE',
            'countryOriginDE','productTitleLength','promisedDeliveryDays','partnerSellingDays', 'orderCorona']
    WEEK = ['orderMonday','orderTuesday','orderWednesday','orderThursday','orderFriday','orderSaturday','orderSunday']
    MONTH = ['orderJanuary','orderFebruary','orderMarch','orderApril','orderMay','orderJune',
             'orderJuly','orderAugust','orderSeptember','orderOctober','orderNovember','orderDecember']
    YEAR = ['orderYear2020']
    GROUP = ['groupHealth','groupHome','groupSports','groupComputer','groupPets','groupToys','groupBooks', 
             'groupBaby', 'groupMusic', 'groupFood','groupOffice','groupFashion','groupOther','groupCar']

    # Dynamic Columns:
    TRANSPORTERX = ['transporterPOSTNL/X','transporterDHL/X','transporterDPD/X','transporterBRIEF/X','transporterOTHER/X']
    KNOWNX = ['caseKnownX','returnKnownX','cancellationKnownX','onTimeDeliveryKnownX','lateDeliveryKnownX']
    PRODUCTX = ['productOrderCountX','productTotalCountX','productTotalReturnedX','productReturnFractionX']
    SELLERX = ['sellerDailyOrdersX']
    HISTORICX = []
    historic_variable = ['transporterCode','sellerId','productGroup']
    for x in range(len(historic_variable)):
        HISTORICX = HISTORICX + [historic_variable[x]+'HistoricHappyX',historic_variable[x]+'HistoricUnhappyX',historic_variable[x]+'HistoricUnknownX']

    # Determinants:
    DETERMINANT = ['noReturn', 'noCase', 'noCancellation', 'onTimeDelivery']

    # Classifications
    CLASSIFICATION = ['generalMatchClassification','detailedMatchClassification','binaryMatchClassification','determinantClassification']

    X_col = BASIC + WEEK + MONTH + YEAR + GROUP + TRANSPORTERX + KNOWNX + PRODUCTX + SELLERX + HISTORICX
    Y_col = ['detailedMatchClassification']
    
    return df, X_col, Y_col, historic_variable, DATE