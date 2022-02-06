import pandas as pd
import numpy as np
import statistics
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import math
import requests
from pickle import load
from datetime import datetime, timedelta
from alpha_vantage.timeseries import TimeSeries
from time import sleep
from tqdm import tqdm
from pymongo import MongoClient,InsertOne
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

# Connect  to MongoDB
client = MongoClient()
client_name = "MSDS_696_Slim"
db = client[client_name]

# List of Stocks
CoList = pd.read_excel('Input Files/List of ETFs.xlsx',sheet_name='Low_Missing')
CoList = CoList['Symbol']

# List of Proxies
proxies = pd.read_excel('Input Files/List of ETFs.xlsx',sheet_name='Proxies')
proxies = proxies['Symbol']

# List of Market Holidays
holiday_list = list(pd.read_excel("Input Files/Stock Market Holidays.xlsx")['Date'])
holiday_list = [holiday.strftime('%Y-%m-%d') for holiday in holiday_list]

################## Set Interval #######################
# 1min, 5min, 15min, 30min, 60min
minute_interval = '1min'
delay = 30

######### Alphavantage key
key = 'xxxxxxxxxxxxxxxxxxxxxx'
# Special Key
key = 'xxxxxxxxxxxxxxxxxxxx'

def get_stocks_periods():
    file_name = 'Result Log by Window - ' +str(delay)+' min.csv'
    file = pathlib.Path.cwd() / 'Logs'/ 'Result Logs' / 'By Window' / file_name
    result_by_window_old = pd.read_csv(file)
    old_test = result_by_window_old.loc[(result_by_window_old['ROC Score'] > 0.80)]

    tups = []
    for group in old_test.groupby(['Weekday','Hour']):
        for x in range(len(group[1])):
            new_group = group[1].reset_index(drop=True)
            ticker = new_group.loc[x,"Ticker"]
            day = new_group.loc[x,"Weekday"]
            hour = new_group.loc[x,"Hour"]
            minute = new_group.loc[x,"Minute"]
            tups.append((ticker, day, hour, minute))
    
    return tups


'''
##################################################################################
################# Update MongoDB Functions
##################################################################################
'''

def mongo_as_df(collection_suffix,co):
    # Connect  to MongoDB
    client = MongoClient()
    client_name = "MSDS_696_Slim"
    db = client[client_name]
    collection = db[co + collection_suffix]
    
    df = pd.DataFrame(list(collection.find({})))
    df.sort_values("time", inplace = True, ignore_index=True) 
                        
    return df

def data_downloader(minute_interval, co, month_ago):
    #################### Set Time Periods to DL ##########################
    years = [1]
    months = [month_ago]

    ################# Header ###################################

    ts = TimeSeries(key, output_format='csv')
        
    limit_reached = False
    
    df = pd.DataFrame()

    # For each month, download data if needed
    for year in years:
        for month in months:
            
            # If API limit is not reached
            if limit_reached == False:

                data, meta_data = ts.get_intraday_extended(symbol=co,interval=minute_interval, slice='year'+str(year)+'month'+ str(month))

                df_list = []
                for row in data:
                    df_list.append(row)
                df = pd. DataFrame(df_list)
                df.columns = df.iloc[0]
                df = df.drop(0)
                df = df.reset_index(drop=True)

                # If df has content and not the error message, save it
                if len(df) > 2:
                    # Save the DF
                    sleep(12)
                    return df

                else:
                    print("API Limit Reached")
                    limit_reached = True
                    break  

            else:
                continue

    return df

def data_cleaner(df, minute_interval, co):
    
    #print("++++++++++++++++++++++++++++")
    #print(f"Cleaning for: {co}")

    df.sort_values("time", inplace = True, ignore_index=True) 
    df.drop_duplicates(subset ="time", keep = 'first', inplace = True, ignore_index=True)
    df = df.drop(['open','high','low','volume'], axis=1)


    ################################ Add columns for date data ##########################################################

    # Create datetime from given string date

    df['datetime'] = pd.to_datetime(df['time'])

    # Create short date
    df['short_date'] = df['datetime'].dt.strftime('%Y-%m-%d')
    # Create hour/minute feature
    df['hour_minute'] = df['datetime'].dt.strftime('%H:%M:%S')
    # Create Hour Feature
    df['hour'] = df['datetime'].dt.hour
    # Create minute feature
    df['minute'] = df['datetime'].dt.minute
    # Day of the week integer
    df['weekday'] = df['datetime'].dt.weekday 

    ############################### Data Cleaning ####################################################################

    missing_percents_list = []
    dates_list = []
    clean_combined = pd.DataFrame(columns=df.columns)

    #print("Walking through the days")
    # Group the whole combined df by date
    for doc in df.groupby(df['short_date']):

        # For each day, if it's not a holiday, clean the day, log the missing values, impute missing values, and combine back into the main DF    
        if doc[0] not in holiday_list:
            sample = doc[1].copy()

            # Filter out pre-market and after market
            sample1 = sample.loc[(sample['hour_minute'] < "16:00:00") & (sample['hour_minute'] >= "09:30:00")]

            # Get List of times that should be in the period
            given_time = datetime.strptime(doc[0], '%Y-%m-%d')
            start_time = given_time + timedelta(hours=9)
            start_time = start_time + timedelta(minutes=30)

            required_times = []
            time = start_time
            required_times.append(time)

            for x in range(389):
                time = time + timedelta(minutes=1)
                required_times.append(time)

            # Make List of times not in period that should be
            datetimes_to_add = []

            for required_time in required_times:
                if required_time not in list(sample1['datetime']):
                    datetimes_to_add.append(required_time)

            # Creat variables for the row that should be there
            for date in datetimes_to_add:
                time = date.strftime('%Y-%m-%d %H:%M:%S')
                close = np.nan
                d_time = date
                short_date = date.strftime('%Y-%m-%d')
                hour_min = date.strftime('%H:%M:%S')
                hr = date.hour
                minute = date.minute
                weekday = date.weekday()

                # Create new df (1 row) out of variables
                df_new_line = pd.DataFrame([[time,close,d_time,short_date,hour_min,hr,minute,weekday]], columns=sample1.columns )

                # add the row to the existing df
                sample1 = pd.concat([sample1,df_new_line], ignore_index=True)

            # Sort the DF by time
            df_to_save = sample1.sort_values(by=['datetime'], ignore_index=True)

            # missing data stats
            number_missing = sum(df_to_save.close.isna())
            number_total = df_to_save.shape[0]
            missing_pct = round(number_missing/number_total*100,2)
            missing_percents_list.append(missing_pct)
            dates_list.append(doc[0])

            # Interpolate to fill in missing data.  Try 'forward first' to get previous minute's data, then use future data if that fails.  Shouldn't be too much data
            df_to_save = df_to_save.interpolate(method='linear',limit_direction ='forward')
            df_to_save = df_to_save.interpolate(method='linear',limit_direction ='backward')
            df_to_save = df_to_save.fillna(method='bfill')

            # Add day df into new_combined df
            clean_combined = pd.concat([clean_combined,df_to_save],ignore_index=True)     

    # Sort the new df by time and delete duplicates
    clean_combined.sort_values("time", inplace = True, ignore_index=True) 
    clean_combined.drop_duplicates(subset ="time", keep = 'first', inplace = True, ignore_index=True)

    ################################ Save log of missing percents ########################################
    #print("Logging Missing Data")

    missing_data_df = pd.DataFrame()
    missing_data_df['Day'] = dates_list
    missing_data_df['Day'] = pd.to_datetime(missing_data_df['Day'])
    missing_data_df['%Missing'] = missing_percents_list

    # See if folder exists for missing data log, if not, create it
    destination = pathlib.Path.cwd() / 'Logs'/ 'Missing Values Logs' / minute_interval
    if not destination.exists():
        destination.mkdir(parents=True, exist_ok=True)

    file_name = minute_interval + " " + co + ' Missing Values Log.csv'
    file = pathlib.Path.cwd() / 'Logs' / 'Missing Values Logs' / minute_interval / file_name
    
    if file.is_file():
        existing_log = pd.read_csv(file)
        existing_log['Day'] = pd.to_datetime(existing_log['Day'])
        existing_log = pd.concat([existing_log,missing_data_df],ignore_index=True)
        
        existing_log.sort_values("Day", inplace = True, ignore_index=True) 
        existing_log.drop_duplicates(subset ="Day", keep = 'last', inplace = True, ignore_index=True)
        existing_log.to_csv(file, index=False)

    else:
        missing_data_df.to_csv(file, index=False)


    return clean_combined

def append_to_3(co, cleaned_df):
    #print('+++++++++++++++++')
    #print(co)
    collection_suffix = ' Cleaned Combined'
    
    big_clean_combined_df = mongo_as_df(collection_suffix,co)
    last_date = list(big_clean_combined_df['time'])[-1]
    df_to_add = cleaned_df.loc[(cleaned_df['time'] > last_date)]
    
    if len(df_to_add) > 0:
        #print('Adding')
        collection = db[co + collection_suffix]
        my_dict = df_to_add.to_dict('records')
        collection.insert_many(my_dict)
    #else:
        #print("Nothing to add")
    
    return

def calc_bollinger(df,feature,window=20*60,st=2):
    """
    Calculates bollinger bands for a price time-series.  
    Input: 
    df     : A dataframe of time-series prices
    feature: The name of the feature in the df to calculate the bands for
    window : The size of the rolling window.  Defaults to 20 days with is standard
    st     : The number of standard deviations to use in the calculation. 2 is standard 
    Output: 
    Returns the df with the bollinger band columns added
    """

    # rolling mean and stdev
    rolling_m  = df[feature].rolling(window).mean()
    rolling_st = df[feature].rolling(window).std()

    # add the upper/lower and middle bollinger bands
    df['b-upper']  = rolling_m + (rolling_st * st)
    df['b-middle'] = rolling_m 
    df['b-lower']  = rolling_m - (rolling_st * st)
    
def calc_rsi(df,feature='close',window=14*60):
    """
    Calculates the RSI for the input feature
    Input:
    df      : A dataframe with a time-series of prices
    feature : The name of the feature in the df to calculate the bands for
    window  : The size of the rolling window.  Defaults to 14 days which is standard
    Output: 
    Returns the df with the rsi band column added
    """
    # RSI
    # calc the diff in daily prices, exclude nan
    diff =df[feature].diff()
    diff.dropna(how='any',inplace=True)

    # separate positive and negitive changes
    pos_m, neg_m = diff.copy(),diff.copy()
    pos_m[pos_m<0]=0
    neg_m[neg_m>0]=0

    # positive/negative rolling means
    prm = pos_m.rolling(window).mean()
    nrm = neg_m.abs().rolling(window).mean()

    # calc the rsi and add to the df
    ratio = prm /nrm
    rsi = 100.0 - (100.0 / (1.0 + ratio))
    df['rsi']=rsi

def calc_macd(df,feature='close'):
    """3.
    Calculates the MACD and signial for the input feature
    Input:
    df      : A dataframe with a time-series of prices
    feature : The name of the feature in the df to calculate the bands for
    Output: 
    Returns the df with the macd columns added
    """
    ema12 = df[feature].ewm(span=12*60,adjust=False).mean()
    ema26 = df[feature].ewm(span=26*60,adjust=False).mean()
    df['macd']=ema12-ema26
    df['macd_signal'] = df['macd'].ewm(span=9*60,adjust=False).mean()
    
def add_tech_indicators(df):
    
    df.sort_values("time", inplace = True, ignore_index=True) 
    df.drop_duplicates(subset ="time", keep = 'first', inplace = True, ignore_index=True)

    #print('Calculating Technical Indicators')
    calc_bollinger(df,'close',window=20,st=2)
    calc_rsi(df,feature='close',window=14)
    calc_macd(df,feature='close')

    ######################################### Seasonality Features ###########################################

    date_time = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
    timestamp_s = date_time.map(pd.Timestamp.timestamp)

    day = 24*60*60
    year = (365.2425)*day

    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
    
    # Interpolate Missing Values                 
    df = df.interpolate(method='linear',limit_direction ='forward')
    df = df.interpolate(method='linear',limit_direction ='backward',limit=5)
    df = df.fillna(method='bfill')    

    return df

def calc_techs_append4(co):
    #print("+++++++++++++++++++++")
    #print(co)
    
    # Load in 3, get length
    collection_suffix = ' Cleaned Combined'
    mongo_df_clean_combined = mongo_as_df(collection_suffix,co)
    
    # Convert Strings to floats
    #columns_to_float = ['open','high','low','close','volume']
    columns_to_float = ['close']
    for col in columns_to_float:
        mongo_df_clean_combined[col] = mongo_df_clean_combined[col].astype(float)

    len3 = len(mongo_df_clean_combined)

    # Get length of 4 (doc count)
    collection_suffix = ' First Features'
    collection = db[co + collection_suffix]
    len4 = collection.count_documents({})

    # If length 3 - length 4 - 390 > 0
    if len3 - len4 - 390 > 0:
        # Subsection 3 for len of 3 - len of 4 + a buffer for tech indicators (26*60)    
        subsect1 = mongo_df_clean_combined[len(mongo_df_clean_combined) - (len3-len4) - (30*60):]
        subsect1 = subsect1.drop(labels=['_id'], axis=1)

        # Engineer first features subsection
        df_with_indicators = add_tech_indicators(subsect1)

        # load in 4
        collection_suffix = ' First Features'
        mongo_df_first_features = mongo_as_df(collection_suffix,co)

        # Get subsection of subsection 3 for dates not already in 4
        last_date = list(mongo_df_first_features['time'])[-1]
        df_to_add = subsect1.loc[(subsect1['time'] > last_date)]

        # if len of new subsection > 0 
        if len(df_to_add) > 0:
            #print('Adding')
            collection = db[co + collection_suffix]
            # convert to dict
            my_dict = df_to_add.to_dict('records')
            # add to 4. MongoDB
            collection.insert_many(my_dict)
        #else:
            #print("Nothing to add")
            
    #else:
        #print("Nothing to add")
    
    return

def get_proxies():

    collection_suffix = ' First Features'

    SPY = mongo_as_df(collection_suffix,'SPY')
    TLT = mongo_as_df(collection_suffix,'TLT')
    VXX = mongo_as_df(collection_suffix,'VXX')
    XLY = mongo_as_df(collection_suffix,'XLY')
    VNQ = mongo_as_df(collection_suffix,'VNQ')

    # Trims Proxies and Renames Columns.  Prepares for merging them with ETF Df
    proxy_names = ['SPY', 'TLT', 'VXX', 'XLY', 'VNQ']
    proxies_ = [SPY, TLT, VXX, XLY, VNQ]
    trimmed_proxies = []

    for x in range(len(proxies_)):
        # Drop redundant columns
        proxy = proxies_[x].drop(['_id','datetime','short_date',
                                 'hour_minute','hour','minute','weekday',
                                 'Day sin', 'Day cos', 'Year sin', 'Year cos'
                                ], axis=1)

        cols = proxy.columns
        # Rename unique columns
        new_cols = [proxy_names[x] + col if col != 'time' else col for col in cols]
        proxy.columns = new_cols
        # Add proxy to new list
        trimmed_proxies.append(proxy)

    return trimmed_proxies

def add_proxies(trimmed_proxies, co):

    # Load in 4, get length
    collection_suffix = ' First Features'
    mongo_df_first_features = mongo_as_df(collection_suffix,co)

    len4 = len(mongo_df_first_features)

    # Get length of 5 (doc count)
    collection_suffix = ' Proxy Features'
    collection = db[co + collection_suffix]
    len5 = collection.count_documents({})

    # If length 4 - length 5 > 0
    if len4 - len5 > 0:

        # Subsection 4 for len of 4 - len of 5 + a buffer for safety    
        subsect1 = mongo_df_first_features[len(mongo_df_first_features) - (len4-len5) - (13*60):]
        subsect1 = subsect1.drop(labels=['_id'], axis=1)

        ##################### Get relevant proxy sections. ###############
        subsect_last_date = list(subsect1['time'])[-1]
        subsect_first_date = list(subsect1['time'])[0]

        # Subset each proxy in proxies
        SPY_sub = trimmed_proxies[0].loc[(trimmed_proxies[0]['time'] >= subsect_first_date)]
        SPY_sub = SPY_sub.loc[(SPY_sub['time'] <= subsect_last_date)]

        subsect2 = pd.merge(subsect1, SPY_sub,  how='left', on = ['time'])

        TLT_sub = trimmed_proxies[1].loc[(trimmed_proxies[1]['time'] >= subsect_first_date)]
        TLT_sub = TLT_sub.loc[(TLT_sub['time'] <= subsect_last_date)]

        subsect2 = pd.merge(subsect2, TLT_sub,  how='left', on = ['time'])

        VXX_sub = trimmed_proxies[2].loc[(trimmed_proxies[2]['time'] >= subsect_first_date)]
        VXX_sub = VXX_sub.loc[(VXX_sub['time'] <= subsect_last_date)]

        subsect2 = pd.merge(subsect2, VXX_sub,  how='left', on = ['time'])

        XLY_sub = trimmed_proxies[3].loc[(trimmed_proxies[3]['time'] >= subsect_first_date)]
        XLY_sub = XLY_sub.loc[(XLY_sub['time'] <= subsect_last_date)]

        subsect2 = pd.merge(subsect2, XLY_sub,  how='left', on = ['time'])

        VNQ_sub = trimmed_proxies[4].loc[(trimmed_proxies[4]['time'] >= subsect_first_date)]
        VNQ_sub = VNQ_sub.loc[(VNQ_sub['time'] <= subsect_last_date)] 

        subsect2 = pd.merge(subsect2, VNQ_sub,  how='left', on = ['time'])

        # Interpolate Missing Values                 
        subsect2 = subsect2.interpolate(method='linear',limit_direction ='forward')
        subsect2 = subsect2.interpolate(method='linear',limit_direction ='backward',limit=5)
        subsect2 = subsect2.fillna(method='bfill')

        #################################################################

        # load in 5
        collection_suffix = ' Proxy Features'
        mongo_df_proxies = mongo_as_df(collection_suffix,co)

        # Get subsection of subsection 4 for dates not already in 5
        last_date = list(mongo_df_proxies['time'])[-1]
        df_to_add = subsect2.loc[(subsect2['time'] > last_date)]

        # if len of new subsection > 0 
        if len(df_to_add) > 0:
            #print('Adding')
            collection = db[co + collection_suffix]
            # convert to dict
            my_dict = df_to_add.to_dict('records')
            # add to 5. MongoDB
            collection.insert_many(my_dict)
        #else:
            #print("Nothing to add")
    #else:
        #print("Nothing to add")
            
    return

features_to_transform = ['close','b-upper','b-middle', 'b-lower',
    'SPYclose','SPYb-upper', 'SPYb-middle', 'SPYb-lower', 
    'TLTclose', 'TLTb-upper', 'TLTb-middle', 'TLTb-lower',
    'VXXclose', 'VXXb-upper', 'VXXb-middle', 'VXXb-lower', 
    'XLYclose', 'XLYb-upper', 'XLYb-middle', 'XLYb-lower', 
    'VNQclose', 'VNQb-upper', 'VNQb-middle','VNQb-lower',]

def transform_stationary(df,features_to_transform,transform='log'):
    """
    Transform time-series data using a log or boxcox transform.  Calculate the augmented
    dickey-fuller (ADF) test for stationarity after the transform
    Inputs:
    df: a dataframe of features
    features_to_transform: A list of features to apply the transform
    transform: The transform to apply (log, boxbox)
    Output
    Applies the transforms inplace in df
    """
        
    # transform each column in the features_to_transform list
    for feature in df.columns:
        if feature in features_to_transform:
            # log transform
            if transform=='log':
                df[feature] = df[feature].apply(np.log)

            # boxcox transform  
            elif transform=='boxcox':
                bc,_ = stats.boxcox(df[feature])
                df[feature] = bc

            else:
                print("Transformation not recognized")
    return df
                
def make_df_stationary(df, features_to_transform):

    df_copy = df.copy()
    close_prices = df_copy.loc[:,'close']
    transform_stationary(df,features_to_transform,'log')

    df['close_prices'] = close_prices

    return df

def add_stationary_data(co,features_to_transform):

    # Load 5
    collection_suffix = ' Proxy Features'
    mongo_df_proxies = mongo_as_df(collection_suffix,co)
    
    # Make 5 Stationary
    mongo_df_proxies = make_df_stationary(mongo_df_proxies, features_to_transform)

    # Get len of 5
    len5 = len(mongo_df_proxies)

    # Get length of 6 (doc count)
    collection_suffix = ' Stationary'
    collection = db[co + collection_suffix]
    len6 = collection.count_documents({})

    # If length 5 - length 6 > 0
    if len5 - len6 > 0:

        mongo_df_proxies1 = mongo_df_proxies.drop(labels=['_id'], axis=1)

        # Load 6
        collection_suffix = ' Stationary'
        mongo_df_stationary = mongo_as_df(collection_suffix,co)

        # Get portion of stationary 5 not in 6
        last_date = list(mongo_df_stationary['time'])[-1]
        df_to_add = mongo_df_proxies1.loc[(mongo_df_proxies1['time'] > last_date)]

        # if len of new subsection > 0 
        if len(df_to_add) > 0:
            #print('Adding')
            collection = db[co + collection_suffix]
            # convert to dict
            my_dict = df_to_add.to_dict('records')
            # add to 6. MongoDB
            collection.insert_many(my_dict)
    return

# Download data, cleand data, and add to 3.

collection_suffix = ' Cleaned Combined'
client = MongoClient()
client_name = "MSDS_696_Slim"
db = client[client_name]
#months_to_dl = [2,1]
months_to_dl = [1]


for month_to_dl in months_to_dl:
    for co in tqdm(CoList):
        # Doanload Data
        new_data = data_downloader(minute_interval, co, month_to_dl)

        # Clean: Trim and Impute.  
        cleaned_new_data = data_cleaner(new_data, minute_interval, co)

        # Append to 3.
        append_to_3(co, cleaned_new_data)

# Add First Features to Data
for co in tqdm(CoList):
    calc_techs_append4(co)

# Add proxies to Data
trimmed_proxies = get_proxies()

for co in tqdm(CoList):
    add_proxies(trimmed_proxies, co)   

# Add stationarity to Data
for co in tqdm(CoList):
    add_stationary_data(co, features_to_transform)