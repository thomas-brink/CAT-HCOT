# # Seminar BA & QM
# ### Code - Data Cleaning and Preparation
# 
# Group 8

import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
import time
import functions
import importlib

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')

### Data Preparation
## Load data and set data-types 

# Import data: fill in path to where files are located
df_2019 = pd.read_csv('/Users/thoma/Documents/seminar_data/data_2019.csv', low_memory = True)
df_2020 = pd.read_csv('/Users/thoma/Documents/seminar_data/data_2020.csv', low_memory = True)

# Concatenate files and create new index
df_full = pd.concat([df_2019, df_2020])
df_full = df_full.reset_index(drop = True)

print('Total # records: ',df_full.shape[0])

#Rename columns:
rename_columns = {'datetTimeFirstDeliveryMoment': 'dateTimeFirstDeliveryMoment',
                  'generalMatchClassification'  : 'detailedMatchClassification',
                  'detailedMatchClassification' : 'generalMatchClassification',
                  'quanityReturned'             : 'quantityReturned'}

df_full = df_full.rename(columns = rename_columns)


df_full['generalMatchClassification'] = df_full['generalMatchClassification'].replace('KNOWN HAPPY', 'HAPPY')

df_full['detailedMatchClassification'] = df_full['detailedMatchClassification'].replace({'KNOWN HAPPY':'HAPPY',

# Transform date variables to date-type
df_full['orderDate']                   = pd.to_datetime(df_full['orderDate'])
df_full['cancellationDate']            = pd.to_datetime(df_full['cancellationDate'])
df_full['promisedDeliveryDate']        = pd.to_datetime(df_full['promisedDeliveryDate'])
df_full['shipmentDate']                = pd.to_datetime(df_full['shipmentDate'])
df_full['dateTimeFirstDeliveryMoment'] = pd.to_datetime(df_full['dateTimeFirstDeliveryMoment'])
df_full['startDateCase']               = pd.to_datetime(df_full['startDateCase'])
df_full['returnDateTime']              = pd.to_datetime(df_full['returnDateTime'])
df_full['registrationDateSeller']      = pd.to_datetime(df_full['registrationDateSeller'])

# Change type of columns
dtype = {'calculationDefinitive': bool,
         'noCancellation'       : bool,
         'noCase'               : bool,
         'hasOneCase'           : bool,
         'hasMoreCases'         : bool,
         'noReturn'             : bool}

df_full = df_full.astype(dtype)


## Remove Noise

# Remove nonsensical rows
noise = df_full.loc[(df_full['startDateCase']        < df_full['orderDate']) | 
                   (df_full['cancellationDate']      < df_full['orderDate']) |
                   (df_full['promisedDeliveryDate']  < df_full['orderDate']) |
                   (df_full['shipmentDate']          < df_full['orderDate']) |
                   (df_full['returnDateTime']        < df_full['orderDate']) |
                   (df_full['cancellationDate']      > df_full['returnDateTime']) |
                   (df_full['orderDate']             < df_full['registrationDateSeller']) |
                   (df_full['orderDate']             > df_full['dateTimeFirstDeliveryMoment'].dt.normalize()) |
                   (df_full['cancellationDate']      > df_full['dateTimeFirstDeliveryMoment'].dt.normalize()) |
                   ((df_full['returnDateTime']       < df_full['dateTimeFirstDeliveryMoment'].dt.normalize()) & 
                                                       (df_full['dateTimeFirstDeliveryMoment'].notnull()) &
                                                       (df_full['returnDateTime'].notnull())) |
                   (df_full['registrationDateSeller'].isnull()) |
                   (df_full['promisedDeliveryDate'].isnull())].index

# Drop noisy data
df = df_full.drop(index = noise)
print(len(noise), 'complete records removed from the data')
print('Cleaned # records: ',df.shape[0],'\n')

# Sort rows by orderDate and create new index
df = df.sort_values(by = 'orderDate')
df = df.reset_index(drop = True)

# Remove inconsistent values
cancellationNoise = df.loc[(df['noCancellation'] == True) & (df['cancellationDate'].notnull())].index
returnNoise       = df.loc[(df['noReturn'] == True) & (df['returnDateTime'].notnull())].index
caseNoise         = df.loc[(df['noCase'] == True) & (df['startDateCase'].notnull())].index
quantityNoise     = df.loc[df['quantityReturned'] > df['quantityOrdered']].index
deliveryNoise     = df.loc[(df['dateTimeFirstDeliveryMoment'].notnull()) & (df['onTimeDelivery'].isnull())].index

df.loc[cancellationNoise, ['cancellationDate','cancellationReasonCode']] = None
df.loc[returnNoise,       ['returnDateTime','quantityReturned','returnCode']] = None
df.loc[caseNoise,         ['startDateCase','cntDistinctCaseIds','hasOneCase','hasMoreCases']] = None
df.loc[quantityNoise,     ['quantityReturned']] = df.loc[quantityNoise, ['quantityOrdered']]
df.loc[deliveryNoise,     ['dateTimeFirstDeliveryMoment']] = None

print('# Records where cancellation values are emptied:',len(cancellationNoise))
print('# Records where return values are emptied: \t',len(returnNoise))
print('# Records where case values are emptied: \t',len(caseNoise))
print('# Records where quantity values are equalized: \t',len(quantityNoise))
print('# Records where delivery values are emptied: \t',len(deliveryNoise))

# Transform totalPrice variable
df['totalPrice'] = np.log(df['totalPrice'])


### Create New Variables

## Time Related

# Create new variables related to periods (days difference)
df['caseDays']             = (df['startDateCase'] - df['orderDate']).dt.days
df['returnDays']           = (df['returnDateTime'] - df['orderDate']).dt.days
df['cancellationDays']     = (df['cancellationDate'] - df['orderDate']).dt.days
df['actualDeliveryDays']   = (df['dateTimeFirstDeliveryMoment'].dt.normalize() - df['orderDate']).dt.days
df['shipmentDays']         = (df['shipmentDate'] - df['orderDate']).dt.days
df['partnerSellingDays']   = (df['orderDate'] - df['registrationDateSeller']).dt.days
df['promisedDeliveryDays'] = (df['promisedDeliveryDate'] - df['orderDate']).dt.days

# Time-related variables
df['orderYear']    = df['orderDate'].dt.year
df['orderMonth']   = df['orderDate'].dt.month
df['orderWeekday'] = df['orderDate'].dt.weekday
df['orderCorona']  = df['orderDate'].apply(lambda x: True if x > datetime.strptime('2020-03-20','%Y-%m-%d') else False)

# Create dummy variables for weekdays, months and years
df['orderMonday']    = df['orderWeekday'].apply(lambda x: True if x == 0 else False)
df['orderTuesday']   = df['orderWeekday'].apply(lambda x: True if x == 1 else False)
df['orderWednesday'] = df['orderWeekday'].apply(lambda x: True if x == 2 else False)
df['orderThursday']  = df['orderWeekday'].apply(lambda x: True if x == 3 else False)
df['orderFriday']    = df['orderWeekday'].apply(lambda x: True if x == 4 else False)
df['orderSaturday']  = df['orderWeekday'].apply(lambda x: True if x == 5 else False)
df['orderSunday']    = df['orderWeekday'].apply(lambda x: True if x == 6 else False)

df['orderJanuary']   = df['orderMonth'].apply(lambda x: True if x == 1 else False)
df['orderFebruary']  = df['orderMonth'].apply(lambda x: True if x == 2 else False)
df['orderMarch']     = df['orderMonth'].apply(lambda x: True if x == 3 else False)
df['orderApril']     = df['orderMonth'].apply(lambda x: True if x == 4 else False)
df['orderMay']       = df['orderMonth'].apply(lambda x: True if x == 5 else False)
df['orderJune']      = df['orderMonth'].apply(lambda x: True if x == 6 else False)
df['orderJuly']      = df['orderMonth'].apply(lambda x: True if x == 7 else False)
df['orderAugust']    = df['orderMonth'].apply(lambda x: True if x == 8 else False)
df['orderSeptember'] = df['orderMonth'].apply(lambda x: True if x == 9 else False)
df['orderOctober']   = df['orderMonth'].apply(lambda x: True if x == 10 else False)
df['orderNovember']  = df['orderMonth'].apply(lambda x: True if x == 11 else False)
df['orderDecember']  = df['orderMonth'].apply(lambda x: True if x == 12 else False)

df['orderYear2020'] = df['orderYear'].apply(lambda x: True if x == 2020 else False)


## Other

df['productTitleLength'] = df['productTitle'].str.len()
df['fulfilmentByPlatform'] = df['fulfilmentType'].apply(lambda x: True if x == 'FBB' else False)

df['countryCodeNL']   = df['countryCode'].apply(lambda x: True if x == 'NL' else False)
df['countryOriginNL'] = df['countryOriginSeller'].apply(lambda x: True if x == 'NL' else False)
df['countryOriginBE'] = df['countryOriginSeller'].apply(lambda x: True if x == 'BE' else False)
df['countryOriginDE'] = df['countryOriginSeller'].apply(lambda x: True if x == 'DE' else False)


## Determinant Classification (for own insight)
df_values = df[['noCancellation','noReturn','noCase','onTimeDelivery']].values

determinantClassification = np.empty(df_values.shape[0], dtype='object')

for ix,df_ in enumerate(df_values):
    if ((df_[0] == 1) & (df_[1] == 1) & (df_[2] == 1) & (df_[3] == True)): 
        determinantClassification[ix] = 'All good'
    elif ((df_[0] == 1) & (df_[1] == 1) & (df_[2] == 1) & (np.isnan(df_[3]) == True)):
        determinantClassification[ix] = 'Unknown delivery'
    elif ((df_[0] == 1) & (df_[1] == 1) & (df_[2] == 1) & (df_[3] == False)):
        determinantClassification[ix] = 'Late delivery'
    elif ((df_[0] == 1) & (df_[1] == 1) & (df_[2] == 0) & (df_[3] == True)):
        determinantClassification[ix] = 'Case'
    elif ((df_[0] == 1) & (df_[1] == 1) & (df_[2] == 0) & (np.isnan(df_[3]) == True)):
        determinantClassification[ix] = 'Case + Unknown delivery'
    elif ((df_[0] == 1) & (df_[1] == 1) & (df_[2] == 0) & (df_[3] == False)):
        determinantClassification[ix] = 'Case + Late delivery'
    elif ((df_[0] == 1) & (df_[1] == 0) & (df_[2] == 1) & (df_[3] == True)):
        determinantClassification[ix] = 'Return'
    elif ((df_[0] == 1) & (df_[1] == 0) & (df_[2] == 1) & (np.isnan(df_[3]) == True)):
        determinantClassification[ix] = 'Return + Unknown delivery'
    elif ((df_[0] == 1) & (df_[1] == 0) & (df_[2] == 1) & (df_[3] == False)):
        determinantClassification[ix] = 'Return + Late delivery'
    elif ((df_[0] == 1) & (df_[1] == 0) & (df_[2] == 0) & (df_[3] == True)):
        determinantClassification[ix] = 'Return + Case'
    elif ((df_[0] == 1) & (df_[1] == 0) & (df_[2] == 0) & (np.isnan(df_[3]) == True)):
        determinantClassification[ix] = 'Return + Case + Unknown delivery'
    elif ((df_[0] == 1) & (df_[1] == 0) & (df_[2] == 0) & (df_[3] == False)):
        determinantClassification[ix] = 'Return + Case + Late delivery'
    elif (df_[0] == 0):
        determinantClassification[ix] = 'Cancellation'
        
df['determinantClassification'] = determinantClassification
df['determinantClassification'].value_counts(dropna = False)


# ##### Binary Match Classification (for own insight)

df['binaryMatchClassification'] = df['generalMatchClassification'].apply(lambda x: 'UNKNOWN' if x == 'UNKNOWN' else 'KNOWN')


# ##### Transporter


def transporterCluster(transporterCode):
    """
    Function to create a new manually clustered transporter variable: 28 -> 5 categories
    """
    if transporterCode in ['AH-NL','TNT','TNT-EXPRESS','TNT-EXTRA']:
        return 'POSTNL'
    elif transporterCode in ['DHL','DHL_DE','DHLFORYOU']:
        return 'DHL'
    elif transporterCode in ['DPD-NL','DPD-BE']:
        return 'DPD'
    elif transporterCode in ['BRIEFPOST','BPOST_BRIEF','DHL-GLOBAL-MAIL','TNT_BRIEF']:
        return 'BRIEF'
    else:
        return 'OTHER'


df['transporterCodeGeneral'] = df['transporterCode'].apply(transporterCluster)
df['transporterCodeGeneral'].value_counts(dropna = False)


# ##### Product Group


def productGroupCluster(productGroup):
    """
    Function to create a new manually clustered product group variable based on categories of retailer
    60 -> 14 groups.
    """
    if productGroup in ['Dutch Books PG','Ebooks and Audiobooks','International Books PG']:
        return 'Books'
    elif productGroup in ['Games Accessories','Games Consoles','Games Software Physical',
                          'Movies','Music']:
        return 'Music, Film & Games'
    elif productGroup in ['Camera','Desktop Monitor and Beamer','Ereaders and Accessories',
                          'Laptop Computers','PC Accessories','Personal Audio',
                          'Sound and Vision Accessories','Storage and Network',
                          'Telephone and Tablet Accessories','Telephones and Tablets','Television']:
        return 'Computer & Electronics'
    elif productGroup in ['General Toys','Recreational and Outdoor Toys']:
        return 'Toys & Hobby'
    elif productGroup in ['Baby and Kids Fashion','Baby PG']:
        return 'Baby & Kids'
    elif productGroup in ['Daily Care PG','Health PG','Perfumery PG','Personal Care']:
        return 'Health & Care'
    elif productGroup in ['Footwear','Jewelry and Watches','Mens and Womens Fashion','Wearables']:
        return 'Fashion, Shoes & Accessories'
    elif productGroup in ['Bodyfashion and Beachwear','Camping and Outdoor','Cycling',
                          'Sporting Equipment','Sportswear','Travel Bags and Accessories']:
        return 'Sports, Outdoor & Travel'
    elif productGroup in ['Educational Dutch','Educational International','Printing and Ink']:
        return 'Office & School'
    elif productGroup in ['Supermarket PG'] :
        return 'Food & Beverage'
    elif productGroup in ['Furniture','Heating and Air','Home Decoration','Home Entertainment',
                          'Household','Household Appliances','Kitchen','Kitchen Machines',
                          'Lighting','Major Domestic Appliances PG','Plumbing and Safety']:
        return 'Home, Cooking & Household'
    elif productGroup in ['Garden','Pet PG','Textiles','Tools and Paint']:
        return 'Pets, Garden & Jobs'
    elif productGroup in ['Car and Motorcycle'] :
        return 'Car & Motor'
    else:
        return 'Other'


df['productGroupGeneral'] = df['productGroup'].apply(productGroupCluster)
df['productGroupGeneral'].value_counts(dropna = False)

# Create dummies of new product grouping
for group in df['productGroupGeneral'].unique():
    columnName = 'group' + group.split(' ')[0].replace(',','')
    df[columnName] = df['productGroupGeneral'].apply(lambda x: True if x == group else False)


# ##### Total Columns

print(df.columns)
print('Total: ',len(df.columns),' columns')


# Convert cleaned and prepared data to CSV File
df.to_csv('/Users/thoma/Documents/seminar_data/cleaned_prepared_data.csv')


# #### Determinant availability (for own insight)


#Calculate totals per Match Determinant
totalCase = df['caseDays'].count()
totalReturn = df['returnDays'].count()
totalCancel = df['cancellationDays'].count()
totalPromisedDelivery = df['promisedDeliveryDays'].count()
totalDelivery = df['actualDeliveryDays'].count()


#Create table for plot
periodTable = pd.concat([df['caseDays'].value_counts().sort_index(),
                         df['returnDays'].value_counts().sort_index(),
                         df['cancellationDays'].value_counts().sort_index(),
                         df['promisedDeliveryDays'].value_counts().sort_index(),
                         df['actualDeliveryDays'].value_counts().sort_index()],axis=1)


#Create percantages per day and calculate running sum
periodTable['caseDays%'] = (periodTable['caseDays'] / totalCase).cumsum()
periodTable['returnDays%'] = (periodTable['returnDays'] / totalReturn).cumsum()
periodTable['cancellationDays%'] = (periodTable['cancellationDays'] / totalCancel).cumsum()
periodTable['promisedDeliveryDays%'] = (periodTable['promisedDeliveryDays'] / totalPromisedDelivery).cumsum()
periodTable['actualDeliveryDays%'] = (periodTable['actualDeliveryDays'] / df.shape[0]).cumsum()




