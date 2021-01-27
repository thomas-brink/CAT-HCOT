#!/usr/bin/env python
# coding: utf-8

import numpy as np


def addKnownColumns(df,X):
    """
    Function to create columns which indicate whether determinants are known after X days.
    Input: X = number of days after order date at which the prediction is made
           df = dataFrame
    """
#     df_ = df[['actualDeliveryDays','onTimeDelivery','shipmentDays','transporterCodeGeneral']]
    
    df['caseKnownX']           = df['caseDays'].apply(lambda x: True if x <= X else False)
    df['returnKnownX']         = df['returnDays'].apply(lambda x: True if x <= X else False)
    df['cancellationKnownX']   = df['cancellationDays'].apply(lambda x: True if x <= X else False)
    
#     df_['actualDeliveryKnown'] = df['actualDeliveryDays'].apply(lambda x: True if x <= X else False)
#     df_['shipmentDaysKnown']   = df['shipmentDays'].apply(lambda x: True if x <= X else False)
    
    df['onTimeDeliveryKnownX'] = df.apply(lambda row: True if ((row.actualDeliveryDays <= X) and (row.onTimeDelivery == True)) else False, axis = 1)
    df['lateDeliveryKnownX']   = df.apply(lambda row: True if ((row.actualDeliveryDays <= X) and (row.onTimeDelivery == False)) else False, axis = 1)
    
    for transporter in df['transporterCodeGeneral'].unique():
        dummyColumn = 'transporter' + transporter +'/X'
        df[dummyColumn] = df.apply(lambda row: True if ((row.shipmentDays <= X) and (row.transporterCodeGeneral == transporter)) else False, axis = 1)

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
            if row[4] != None:
                if row[5] in returnDic:
                    returnDic[row[5]] += row[4]
                else:
                    returnDic[row[5]] = row[4]

            previousID = row[1]
            
        elif (previousID == row[1]):
            
            #update return dictionary if this product is returned
            if row[4] != None:
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
            if row[4] != None:
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
    Input: dataFrame with columns: 'sellerId','orderDate','quantityOrdered','partnerSellingMonths'
    """
    df = df.sort_values(by = ['sellerId','orderDate'])
    df = df.reset_index(drop = True)
    
    df_ = df[['sellerId','orderDate','quantityOrdered','partnerSellingMonths']]
    
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