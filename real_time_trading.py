import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import statistics
import pathlib
import csv
import math
import requests
import pathlib
from pickle import load
from yahoo_fin import stock_info as si
import yfinance as yf
from twelvedata import TDClient
import robin_stocks.robinhood as r
import schedule
import threading
import time
import queue
import sys
import pytz
from pytz import timezone
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from time import sleep
from tqdm import tqdm
from pymongo import MongoClient,InsertOne
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

# Initialize Order Logging
buy_orders_temp = []
sell_orders_temp = []

# Time Zone Stuff
timeZ_Ny = pytz.timezone('America/New_York')
dt_Ny = datetime.now(timeZ_Ny)
ny_hour = dt_Ny.hour

local_time = datetime.now()
local_hour = local_time.hour

tz_offset = local_hour - ny_hour

first_hour_ny = 9
last_hour_ny = 15

first_hour_local = first_hour_ny + tz_offset
last_hour_local = last_hour_ny + tz_offset

######## Twelve Data API Key ###########
td_key = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'

# Initialize Twelve Data client - apikey parameter is required
td = TDClient(apikey=td_key)
today = datetime.today()
year = today.year
month = today.month
day = today.day
market_open = datetime(year, month, day, 9, 30, 0)

# RH login
file_name = 'RH.xlsx'
file = pathlib.Path.cwd() / 'Input Files' / file_name
try: 
    rh_info = pd.read_excel(file)
    login = r.login(rh_info['User'][0],rh_info['Pass'][0])
    
except:
    print("Stored Credentials not found, enter login info here:")
    username = input("RH Username:")
    password = input("RH Password:")
    login = r.login(username,password)
    
max_capital = 5000
max_capital_per_trade = 5000

'''
##################################################################################
################# Setup Functions
##################################################################################
'''

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

def filter_tup_by_weekday(tups):
    weekday = datetime.today().weekday()
    filtered_tups = [x for x in tups if x[1] == weekday]    
    return filtered_tups

def load_a_model(delay, co):
    str_delay = str(delay) + ' min'
    path = 'Models/'+ str_delay +'/'
    file_name = path + co + ' ' + str_delay + ".h5"

    loaded_model = load_model(file_name)
    
    return loaded_model

def load_in_models(tups):
    models = []
    for tup in tups:
        model = load_a_model(30, tup[0])
        models.append(model)
    return models

def pickle_load_scaler(delay, co):
    '''
    ###################################################################
    Loads in the model from local storage
    ###################################################################
    
    '''
    str_delay = str(delay) + ' min'
    path = 'Models/'+ str_delay +'/'
    #file_name = path + co + ' ' + str_delay + ".h5"
    file_name = path + co + ' ' + str_delay + " scaler.pkl"

    #loaded_model = load_model(file_name)
    loaded_scaler = load(open(file_name, 'rb'))
    
    return loaded_scaler

def mongo_as_df(collection_suffix,co):
    # Connect  to MongoDB
    client = MongoClient()
    client_name = "MSDS_696_Slim"
    db = client[client_name]
    collection = db[co + collection_suffix]
    
    df = pd.DataFrame(list(collection.find({})))
    df.sort_values("time", inplace = True, ignore_index=True) 
                        
    return df

'''
##################################################################################
################# Helper Functions
##################################################################################
'''
def load_last_3_days_of_stationary_data(co):
    collection_suffix = ' Stationary'
    collection = db[co + collection_suffix]
    mongo_df_stationary = mongo_as_df(collection_suffix,co)
    mongo_df_stationary = mongo_df_stationary.drop(columns=['_id', 'time','datetime','short_date', 'hour_minute'])
    mongo_df_stationary = mongo_df_stationary[len(mongo_df_stationary) - (390*3) : ]
    return mongo_df_stationary

def load_last_5_days_of_combined_data(co):
    
    collection_suffix = ' Cleaned Combined'
    collection = db[co + collection_suffix]
    mongo_df_combined = mongo_as_df(collection_suffix,co)
    mongo_df_combined = mongo_df_combined.drop(columns=['_id'], axis=1)
    mongo_df_combined = mongo_df_combined[len(mongo_df_combined) - (390*5) : ]

    return mongo_df_combined

def make_early_prediction(co, delay, test_data):
    test_data_list = []
    scaler = pickle_load_scaler(delay, co)
    model = load_a_model(delay, co)
    
    test_data = scaler.transform(test_data)
    test_data_list.append(test_data)
    test_array = np.asarray(test_data_list)

    try:
        y_pred = model.predict(test_array)
        pred = y_pred[0][0]
    except:
        print("Error for:",co)
        pred = 0
        
    return pred

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

def get_proxies():

    SPY = proxy_dict['SPY']['TechIndicators']
    TLT = proxy_dict['TLT']['TechIndicators']
    VXX = proxy_dict['VXX']['TechIndicators']
    XLY = proxy_dict['XLY']['TechIndicators']
    VNQ = proxy_dict['VNQ']['TechIndicators']

    # Trims Proxies and Renames Columns.  Prepares for merging them with ETF Df
    proxy_names = ['SPY', 'TLT', 'VXX', 'XLY', 'VNQ']
    proxies_ = [SPY, TLT, VXX, XLY, VNQ]
    trimmed_proxies = []

    for x in range(len(proxies_)):
        # Drop redundant columns
        proxy = proxies_[x].drop(['datetime','short_date',
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

def add_proxies(trimmed_proxies, df):

    ##################### Get relevant proxy sections. ###############
    subsect_last_date = list(df['time'])[-1]
    subsect_first_date = list(df['time'])[0]

    # Subset each proxy in proxies
    SPY_sub = trimmed_proxies[0].loc[(trimmed_proxies[0]['time'] >= subsect_first_date)]
    SPY_sub = SPY_sub.loc[(SPY_sub['time'] <= subsect_last_date)]

    subsect2 = pd.merge(df, SPY_sub,  how='left', on = ['time'])

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

    return subsect2

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

def make_prediction(co):
    
    # Load in stationary data
    df = main_dict[co]['Stationary']
    
    # Trim to be 3 days
    df = df[len(df)- (390*3):]
    
    # load in scaler from dict and scale
    scaler = main_dict[co]['Scaler']
    test_data = scaler.transform(df)
    
    # Make numpy array
    test_data = np.asarray(test_data)
    
    # add to list 
    test_data_list = []
    test_data_list.append(test_data)
    test_array = np.asarray(test_data_list)
    
    # load in model from dict
    model = main_dict[co]['Model']
    
    # Make Prediction
    try:
        y_pred = model.predict(test_array)
        pred = y_pred[0][0]
    except:
        print("Error for:",co)
        pred = 0
        
    return pred
    
def get_current_price(co):
    try:
        price = float(r.get_latest_price(co)[0])
    except:
        try:
            price = float(si.get_live_price(co))
        except:
            try:
                url = f"https://finance.yahoo.com/quote/{co}/"
                response = requests.get(url)
                soup = BeautifulSoup(response.text, "html.parser")
                class_ = "My(6px) Pos(r) smartphone_Mt(6px) W(100%)"
                price = float(soup.find("div", class_=class_).find("fin-streamer").text)
            except:
                price = float('NaN')
        
    time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    return price, time      

'''
##################################################################################
################# Proxy Streaming
##################################################################################
''' 
def update_proxy(co):
    print('Updating Proxy: ',co)
    
    # Update if time zone changes
    if datetime.now().hour < 7:
        print('Too Early')
        
    else:
    
        # Get Existing DF
        df = proxy_dict[co]['Cleaned_Combined']
        
        data = td_time_series(co)
    
        if len(data) == 0:
            # Download Most Recent Day
            retries = 0
            while retries <= 3:
                try:
                    #code with possible error
                    data = yf.download(tickers=co, period='1d', interval='1m',threads = False)
                except:
                    print('error...trying again')
                    retries += 1
                    sleep(2)
                    #print('trying again')
                    continue
                else:
                    #the rest of the code
                    break       

        # Make First Features
        data['time'] = data.index
        #data.sort_values("time", inplace = True, ignore_index=True) 
        
        try:
            data['close'] = data['Close']
        except:
            pass
        data = data.reset_index()
        try:
            data = data.drop(columns=['Open','High','Low','Adj Close','Volume','Datetime','Close'])
        except:
            pass
        try:
            data = data.drop(columns=['open','high','low','volume','datetime'])
        except:
            pass
        
        data['datetime'] = data['time']
        data['short_date'] = data['datetime'].dt.strftime('%Y-%m-%d')
        data['hour_minute'] = data['datetime'].dt.strftime('%H:%M:%S')
        data['hour'] = data['datetime'].dt.hour
        data['minute'] = data['datetime'].dt.minute
        data['weekday'] = data['datetime'].dt.weekday
        data['time'] = data['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        data['datetime'] = data['time']

        # Combine into one
        combined_df = pd.concat([df,data], ignore_index=True)
        
        # Sort the new df by time and delete duplicates
        combined_df.sort_values("time", inplace = True, ignore_index=True) 
        combined_df.drop_duplicates(subset ="time", keep = 'first', inplace = True, ignore_index=True)

        # Interpolate to fill in missing data.  Try 'forward first' to get previous minute's data, then use future data if that fails.  Shouldn't be too much data
        combined_df = combined_df.interpolate(method='linear',limit_direction ='forward')
        combined_df = combined_df.interpolate(method='linear',limit_direction ='backward')
        combined_df = combined_df.fillna(method='bfill')

        # Convert Strings to floats
        columns_to_float = ['close']
        for col in columns_to_float:
            combined_df[col] = combined_df[col].astype(float)

        # Save to Dict
        proxy_dict[co]['Cleaned_Combined'] = combined_df

        df2 = combined_df.copy()

        # Make Tech Indicators
        add_tech_indicators(df2)

        # Save to Dict
        proxy_dict[co]['TechIndicators'] = df2

        print(co," Updated")
    
    return

'''
##################################################################################
################# Streaming Other Cos
##################################################################################
''' 

def update_co(co):
    print('Updating Scheduled Co: ',co)
    # Get Existing DF
    df = main_dict[co]['Cleaned_Combined']

    data = td_time_series(co)
    
    if len(data) == 0:
        # Download Most Recent Day
        retries = 0
        while retries <= 3:
            try:
                #code with possible error
                data = yf.download(tickers=co, period='1d', interval='1m',threads = False)
            except:
                print('error...trying again')
                retries += 1
                sleep(2)
                #print('trying again')
                continue
            else:
                #the rest of the code
                break 
    
    # Make First Features
    data['time'] = data.index
    #data.sort_values("time", inplace = True, ignore_index=True) 

    try:
        data['close'] = data['Close']
    except:
        pass
    data = data.reset_index()
    try:
        data = data.drop(columns=['Open','High','Low','Adj Close','Volume','Datetime','Close'])
    except:
        pass
    try:
        data = data.drop(columns=['open','high','low','volume','datetime'])
    except:
        pass

    data['datetime'] = data['time']
    data['short_date'] = data['datetime'].dt.strftime('%Y-%m-%d')
    data['hour_minute'] = data['datetime'].dt.strftime('%H:%M:%S')
    data['hour'] = data['datetime'].dt.hour
    data['minute'] = data['datetime'].dt.minute
    data['weekday'] = data['datetime'].dt.weekday
    data['time'] = data['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    data['datetime'] = data['time']

    # Combine into one
    combined_df = pd.concat([df,data], ignore_index=True)
    
    # Sort the new df by time and delete duplicates
    combined_df.sort_values("time", inplace = True, ignore_index=True) 
    combined_df.drop_duplicates(subset ="time", keep = 'first', inplace = True, ignore_index=True)

    # Interpolate to fill in missing data.  Try 'forward first' to get previous minute's data, then use future data if that fails.  Shouldn't be too much data
    combined_df = combined_df.interpolate(method='linear',limit_direction ='forward')
    combined_df = combined_df.interpolate(method='linear',limit_direction ='backward')
    combined_df = combined_df.fillna(method='bfill')

    # Convert Strings to floats
    columns_to_float = ['close']
    for col in columns_to_float:
        combined_df[col] = combined_df[col].astype(float)

    # Save to Dict
    main_dict[co]['Cleaned_Combined'] = combined_df

    df2 = combined_df.copy()

    # Make Tech Indicators
    add_tech_indicators(df2)

    # Save to Dict
    main_dict[co]['TechIndicators'] = df2
    
    # Add Proxy Data
    trimmed_proxies = get_proxies()
    df3 = add_proxies(trimmed_proxies, df2)
    
    # Save to Dict
    main_dict[co]['WithProxies'] = df3
    
    # Make Stationary
    df4 = make_df_stationary(df3, features_to_transform)
    
    df4 = df4.drop(columns=['time', 'datetime', 'short_date', 'hour_minute'],axis=1)
    
    # Save to Dict
    main_dict[co]['Stationary'] = df4
    print(co,'Updated')

    return

'''
##################################################################################
################# Time Series Data
##################################################################################
''' 
def td_time_series(co):
    # Construct the necessary time series
    ts = td.time_series(
        symbol=co,
        interval="1min",
        outputsize=400,
        timezone="America/New_York",
        start_date = market_open
    )

    # Returns pandas.DataFrame
    try:
        df = ts.as_pandas()
    except:
        df = pd.DataFrame()
    
    return df

'''
##################################################################################
################# Job Functions
##################################################################################
''' 
def buy_job(co,hour,minute,amount):    
    # Future Robinhood functionality
    buy_trade = r.order_buy_fractional_by_price(co, amount, timeInForce='gfd', extendedHours=False,jsonify=True)
    sleep(1)

    try:
        trade_id = buy_trade['id']
    except:
        print('Order Rejected.  Trying Again...')
        sleep(1)
        buy_trade = r.order_buy_fractional_by_price(co, amount, timeInForce='gfd', extendedHours=False,jsonify=True)
        sleep(1)
        try:
            trade_id = buy_trade['id']
        except:
            print('Order Rejected. Skip This Time')
            shares = 0
            price, time_= get_current_price(co)
            paper_trade_logger(co, 'Buy',hour,minute,price,time_,shares)
            return shares

    order_info = r.get_stock_order_info(trade_id)
    print(order_info['state'])

    if order_info['state'] == 'cancelled':
        print('Order Cancelled.  Trying Again...')
        buy_trade = r.order_buy_fractional_by_price(co, amount, timeInForce='gfd', extendedHours=False,jsonify=True)
        sleep(1)

        try:
            trade_id = buy_trade['id']
            order_info = r.get_stock_order_info(trade_id)

            if order_info['state'] == 'cancelled':
                print('Order Cancelled. Skip This Time')
                shares = 0
                price, time_= get_current_price(co)
                paper_trade_logger(co, 'Buy',hour,minute,price,time_,shares)
                return shares

            else:
                print("RH Order Successful for:",co)
                price, time_= get_current_price(co)
                try:
                    shares = round(amount/price,4)
                except:
                    shares = 0
                print("Buy",amount," of",co,' For:',price,' At:', time_)
                paper_trade_logger(co, 'Buy',hour,minute,price,time_,shares)
                rh_trade_id_logger('Buy', trade_id)
                
                # Append order id to list
                try:
                    buy_orders_temp.append(buy_trade['id'])
                except:
                    pass
                return shares
               

        except:
            print('Order Rejected. Skip This Time')
            shares = 0
            price, time_= get_current_price(co)
            paper_trade_logger(co, 'Buy',hour,minute,price,time_,shares)
            return shares

    else:
        print("RH Order Successful for:",co)
        price, time_= get_current_price(co)
        try:
            shares = round(amount/price,4)
        except:
            shares = 0
        print("Buy",amount," of",co,' For:',price,' At:', time_)
        paper_trade_logger(co, 'Buy',hour,minute,price,time_,shares)
        rh_trade_id_logger('Buy', trade_id)
        
        # Append order id to list
        try:
            buy_orders_temp.append(buy_trade['id'])
        except:
            pass
        return shares
    
    return shares

def sell_job(co,hour,minute,shares):
    
    # Robinhood functionality
    try:
        shares_held = float(current_positions[co])
    except:
        shares_held = 0
        
    # Sell Stock
    if shares_held > 0:
        try:
            sale_trade = r.order_sell_fractional_by_quantity(co,shares_held, timeInForce='gfd', priceType='bid_price', extendedHours=False,jsonify=True)
    
            try:
                sell_orders_temp.append(sale_trade['id'])
            except:
                pass
            
            price, time_= get_current_price(co)
            print("Selling",shares," of:",co,' For:',price,' At:', time_)
            paper_trade_logger(co, 'Sell',hour,minute,price,time_,shares)
            trade_id = sale_trade['id']
            rh_trade_id_logger('Sell', trade_id)
            
        except:
            print('No Sale made')
    else:
        print('No Shares to Sell')
    
    return

def proxy_job(co):
    update_proxy(co)
    
    return

def later_period_job(tup):
    
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    
    co = tup[0]
    print('Starting Job for: ',co)
    day = tup[1]
    hour = tup[2]
    minute = tup[3]    
    
    update_co(co)
    
    pred = make_prediction(co)

    if pred > .50:
        print('Adding:', co,' to order book')
        order_book.loc[(order_book.Ticker == co) & (order_book.Hour == hour) & (order_book.Minute == minute), 'Action'] = 1

    else:
        print("Don't buy", co)
        
    print('Finished Job for: ',co)

    return

'''
##################################################################################
################# Scheduling Prep Functions
##################################################################################
''' 
def run_threaded(job_func,argument):
    job_thread = threading.Thread(target=job_func(argument))
    job_thread.start()
    
def worker_main():
    while True:
        try:
            job_func, job_args = jobqueue.get()
            job_func(*job_args)
        except Exception as e:
            print(e)
            
'''
##################################################################################
################# Log Trades and Run Backtests
##################################################################################
''' 
def paper_trade_logger(co, BuyOrSell,hour,minute,price,time, shares):
    if minute == 0:
        minute = '00'
    
    # Initiate Dataframe
    cols = ['Buy/Sell', 'Ticker','Hour','Minute','Price','Time','Shares']
    df_new_line = pd.DataFrame([[BuyOrSell,co,hour,minute,price,time,shares]], columns=cols)
    
    # Save Dataframe
    #destination = pathlib.Path.cwd() / 'Logs' / 'Daily Trade Logs' / str(today.year) / today.strftime('%Y-%m-%d')
    destination = pathlib.Path.cwd() / 'Logs' / 'Daily Trade Logs' / str(today.year) / 'Individual Trades' / today.strftime('%Y-%m-%d')

    if not destination.exists():
        destination.mkdir(parents=True, exist_ok=True)

    # Set Filename
    file_name = BuyOrSell + ' ' + co + ' ' + str(hour) + ' ' + str(minute) +'.csv'
    file = pathlib.Path.cwd() / 'Logs' / 'Daily Trade Logs' / str(today.year) / 'Individual Trades' / today.strftime('%Y-%m-%d') / file_name
    
    df_new_line.to_csv(file, index=False)
    
    return

def run_backtest(day_to_test):
    if day_to_test == 'today':
        day_to_test = today.strftime('%Y-%m-%d')
    
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print("Running Backtest")
    tickers = []
    hour_mins = []
    buy_times = []
    buy_prices = []
    sell_times = []
    sell_prices = []

    for proxy in proxies:
        update_proxy(proxy)
        sleep(3)
        
    datetime_object = datetime.strptime(day_to_test, '%Y-%m-%d')
    day_of_week = datetime_object.weekday()
    
    tups_filtered = [x for x in tups if x[1] == day_of_week] 
    tups_filtered = sorted(tups_filtered, key = lambda x: (x[2], x[3]))
        
    for tup in tups_filtered:
        co = tup[0]
        weekday = tup[1]
        hour = tup[2]
        minute = tup[3]

        print('=====================================================================================')
        print(co, ' at: ',hour,' ',minute)

        # Update Stationary Data
        update_co(co)
        sleep(4)

        df = main_dict[co]['Stationary']


        end_position_set = False
        for x in range(len(df)):
            if df['hour'][x] == hour and df['minute'][x] == minute and df['weekday'][x] == weekday:
                end_position = x

        if end_position < (390 * 3):
            for x in range(len(df)):        
                if df['hour'][x] == hour and df['minute'][x] == minute + 1 and df['weekday'][x] == weekday:
                    end_position = x

        if end_position >= (390 * 3):
            end_position_set = True

        if end_position_set == False: 
            for x in range(len(df)):            
                if df['hour'][x] == hour and df['minute'][x] == minute + 2 and df['weekday'][x] == weekday:
                    end_position = x

        if end_position >= 390 * 3:
            end_position_set = True

        if end_position_set == False:
            for x in range(len(df)):            
                if df['hour'][x] == hour and df['minute'][x] == minute + 3 and df['weekday'][x] == weekday:
                    end_position = x

        if end_position >= 390 * 3:
            end_position_set = True

        if end_position_set == False:
            for x in range(len(df)):            
                if df['hour'][x] == hour and df['minute'][x] == minute - 1 and df['weekday'][x] == weekday:
                    end_position = x

        if end_position >= 390 * 3:
            end_position_set = True

        if end_position_set == False:
            for x in range(len(df)):            
                if df['hour'][x] == hour and df['minute'][x] == minute - 2 and df['weekday'][x] == weekday:
                    end_position = x

        if end_position >= 390 * 3:
            end_position_set = True

        if end_position_set == False:
            for x in range(len(df)):            
                if df['hour'][x] == hour and df['minute'][x] == minute - 3 and df['weekday'][x] == weekday:
                    end_position = x

        df3 = df[end_position - (390*3):end_position]    

        if end_position_set == True:
            # load in scaler from dict and scale
            scaler = main_dict[co]['Scaler']
            test_data = scaler.transform(df3)

            # Make numpy array
            test_data = np.asarray(test_data)

            # add to list 
            test_data_list = []
            test_data_list.append(test_data)
            test_array = np.asarray(test_data_list)

            # load in model from dict
            model = main_dict[co]['Model']

            # Make Prediction
            try:
                y_pred = model.predict(test_array)
                pred = y_pred[0][0]
            except:
                print("Error for:",co)
                pred = 0

            if pred > .50:
                # Download Most Recent Day
                retries = 0
                while retries <= 3:
                    try:
                        #code with possible error
                        data = yf.download(tickers=co, period='7d', interval='1m',threads = False)
                    except:
                        print('error...trying again')
                        retries += 1
                        sleep(2)
                        #print('trying again')
                        continue
                    else:
                        #the rest of the code
                        break 

                data['Datetime'] = data.index
                data['hour'] = data['Datetime'].dt.hour
                # Create minute feature
                data['minute'] = data['Datetime'].dt.minute
                # Day of the week integer
                data['weekday'] = data['Datetime'].dt.weekday
                data['Date Str'] = data['Datetime'].dt.strftime('%Y-%m-%d')
                data = data.loc[(data['Date Str'] == day_to_test)]

                buy_price = data.loc[(data['weekday'] == weekday) & (data['hour'] == hour) & (data['minute'] == minute)]['Open'][0]
                buy_hour = data.loc[(data['weekday'] == weekday) & (data['hour'] == hour) & (data['minute'] == minute)]['hour'][0]
                buy_minute = data.loc[(data['weekday'] == weekday) & (data['hour'] == hour) & (data['minute'] == minute)]['minute'][0]
                buy_time = str(buy_hour) + ":" + str(buy_minute)
                try:
                    sell_price = data.loc[(data['weekday'] == weekday) & (data['hour'] == hour) & (data['minute'] == minute+29)]['Open'][0]
                    sell_hour = data.loc[(data['weekday'] == weekday) & (data['hour'] == hour) & (data['minute'] == minute+29)]['hour'][0]
                    sell_minute = data.loc[(data['weekday'] == weekday) & (data['hour'] == hour) & (data['minute'] == minute+29)]['minute'][0]

                except:
                    sell_price = data.loc[(data['weekday'] == weekday) & (data['hour'] == hour) & (data['minute'] == minute+28)]['Open'][0]
                    sell_hour = data.loc[(data['weekday'] == weekday) & (data['hour'] == hour) & (data['minute'] == minute+28)]['hour'][0]
                    sell_minute = data.loc[(data['weekday'] == weekday) & (data['hour'] == hour) & (data['minute'] == minute+28)]['minute'][0]

                sell_time = str(sell_hour) + ":" + str(sell_minute)


                percent_return = (sell_price-buy_price) / buy_price

                print('Buy At: ',buy_price)
                print('Sell At: ',sell_price)
                print('Return Of: ',percent_return)

                tickers.append(co)
                short_time = str(hour)+':'+str(minute)
                buy_times.append(buy_time)
                buy_prices.append(buy_price)
                sell_times.append(sell_time)
                sell_prices.append(sell_price)

            else:
                print("Don't Buy: ",co)

        else:
            print("Skip:",co)

    backtest_log = pd.DataFrame()
    backtest_log['Stock'] = tickers
    backtest_log['Buy Price'] = buy_prices
    backtest_log['Buy Time'] = buy_times
    backtest_log['Sale Price'] = sell_prices
    backtest_log['Sale Time'] = sell_times
    backtest_log['% Return'] = (backtest_log['Sale Price'] - backtest_log['Buy Price']) / backtest_log['Buy Price']
    
    amounts_to_invest = []

    for document in backtest_log.groupby(['Buy Time']):
        if max_capital_per_trade * len(document[1]) > max_capital:
                amount_to_invest = math.floor(max_capital / len(document[1]))
        else:
            amount_to_invest = max_capital_per_trade

        for x in range(len(document[1])):
            amounts_to_invest.append(amount_to_invest)
            
    backtest_log['Amount to Invest'] = amounts_to_invest
    backtest_log['P(L)'] = backtest_log['% Return'] * backtest_log['Amount to Invest']

    destination = pathlib.Path.cwd() / 'Logs' / 'Daily Trade Logs' / str(today.year)
    if not destination.exists():
        destination.mkdir(parents=True, exist_ok=True)

    # Set Filename
    file_name = day_to_test + ' Backtest.csv'
    file = pathlib.Path.cwd() / 'Logs' / 'Daily Trade Logs' / str(today.year) / file_name

    if not file.is_file():
        backtest_log.to_csv(file,index=False)
    
    print('###########################################################')
    print("Backtest Profit(L):", round(sum(backtest_log['P(L)']),2))
    print("Average % Return:",round(statistics.mean(backtest_log['% Return'])*100,2),"%") 

    return backtest_log

def run_passive_backtest(day_to_test):
    if day_to_test == 'today':
        day_to_test = today.strftime('%Y-%m-%d')
    
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print("Running Passive Backtest")
    tickers = []
    hour_mins = []
    buy_times = []
    buy_prices = []
    sell_times = []
    sell_prices = []
    
    datetime_object = datetime.strptime(day_to_test, '%Y-%m-%d')
    day_of_week = datetime_object.weekday()
    
    tups_filtered = [x for x in tups if x[1] == day_of_week] 
    tups_filtered = sorted(tups_filtered, key = lambda x: (x[2], x[3]))
        
    for tup in tups_filtered:
        co = tup[0]
        weekday = tup[1]
        hour = tup[2]
        minute = tup[3]

        print('=====================================================================================')
        print(co, ' at: ',hour,' ',minute)


        # Download Most Recent Day
        retries = 0
        while retries <= 3:
            try:
                #code with possible error
                data = yf.download(tickers=co, period='7d', interval='1m',threads = False)
            except:
                print('error...trying again')
                retries += 1
                sleep(2)
                #print('trying again')
                continue
            else:
                #the rest of the code
                break 

        data['Datetime'] = data.index
        data['hour'] = data['Datetime'].dt.hour
        # Create minute feature
        data['minute'] = data['Datetime'].dt.minute
        # Day of the week integer
        data['weekday'] = data['Datetime'].dt.weekday 
        data['Date Str'] = data['Datetime'].dt.strftime('%Y-%m-%d')
        data = data.loc[(data['Date Str'] == day_to_test)]

        buy_price = data.loc[(data['weekday'] == weekday) & (data['hour'] == hour) & (data['minute'] == minute)]['Open'][0]
        buy_hour = data.loc[(data['weekday'] == weekday) & (data['hour'] == hour) & (data['minute'] == minute)]['hour'][0]
        buy_minute = data.loc[(data['weekday'] == weekday) & (data['hour'] == hour) & (data['minute'] == minute)]['minute'][0]
        buy_time = str(buy_hour) + ":" + str(buy_minute)
        try:
            sell_price = data.loc[(data['weekday'] == weekday) & (data['hour'] == hour) & (data['minute'] == minute+29)]['Open'][0]
            sell_hour = data.loc[(data['weekday'] == weekday) & (data['hour'] == hour) & (data['minute'] == minute+29)]['hour'][0]
            sell_minute = data.loc[(data['weekday'] == weekday) & (data['hour'] == hour) & (data['minute'] == minute+29)]['minute'][0]

        except:
            sell_price = data.loc[(data['weekday'] == weekday) & (data['hour'] == hour) & (data['minute'] == minute+28)]['Open'][0]
            sell_hour = data.loc[(data['weekday'] == weekday) & (data['hour'] == hour) & (data['minute'] == minute+28)]['hour'][0]
            sell_minute = data.loc[(data['weekday'] == weekday) & (data['hour'] == hour) & (data['minute'] == minute+28)]['minute'][0]

        sell_time = str(sell_hour) + ":" + str(sell_minute)


        percent_return = (sell_price-buy_price) / buy_price

        print('Buy At: ',buy_price)
        print('Sell At: ',sell_price)
        print('Return Of: ',percent_return)

        tickers.append(co)
        short_time = str(hour)+':'+str(minute)
        buy_times.append(buy_time)
        buy_prices.append(buy_price)
        sell_times.append(sell_time)
        sell_prices.append(sell_price)
        
        sleep(3)

    backtest_log = pd.DataFrame()
    backtest_log['Stock'] = tickers
    backtest_log['Buy Price'] = buy_prices
    backtest_log['Buy Time'] = buy_times
    backtest_log['Sale Price'] = sell_prices
    backtest_log['Sale Time'] = sell_times
    backtest_log['% Return'] = (backtest_log['Sale Price'] - backtest_log['Buy Price']) / backtest_log['Buy Price']
    #backtest_log['P(L)'] = backtest_log['% Return'] * max_capital_per_trade
    
    amounts_to_invest = []

    for document in backtest_log.groupby(['Buy Time']):
        if max_capital_per_trade * len(document[1]) > max_capital:
                amount_to_invest = math.floor(max_capital / len(document[1]))
        else:
            amount_to_invest = max_capital_per_trade

        for x in range(len(document[1])):
            amounts_to_invest.append(amount_to_invest)
            
    backtest_log['Amount to Invest'] = amounts_to_invest
    backtest_log['P(L)'] = backtest_log['% Return'] * backtest_log['Amount to Invest']

    destination = pathlib.Path.cwd() / 'Logs' / 'Daily Trade Logs' / str(today.year)
    if not destination.exists():
        destination.mkdir(parents=True, exist_ok=True)

    # Set Filename
    file_name = day_to_test + ' Passive Backtest.csv'
    file = pathlib.Path.cwd() / 'Logs' / 'Daily Trade Logs' / str(today.year) / file_name

    if not file.is_file():
        backtest_log.to_csv(file,index=False)
        
    file_name = day_to_test + ' Passive Backtest.csv'
    file = pathlib.Path.cwd() / 'Logs' / 'Daily Trade Logs' / str(today.year) / file_name
    day_backtest = pd.read_csv(file)
    
    print('###########################################################')
    print("Passive Backtest Profit(L):", round(sum(day_backtest['P(L)']),2))
    print("Average % Return:",round(statistics.mean(day_backtest['% Return'])*100,2),"%") 

    return day_backtest

def consolidate_trades():
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('Consolidating Paper Trades')
    # Get Files for each company
    destination = pathlib.Path.cwd() / 'Logs' / 'Daily Trade Logs' / str(today.year) / 'Individual Trades'/ today.strftime('%Y-%m-%d')
    p = destination.glob('**/*')
    files = [x for x in p if x.is_file()]
    main_df = pd.DataFrame()

    # Loop through files and combine into one big DF
    for f in files:
        test_df = pd.read_csv(f)
        main_df = pd.concat([main_df,test_df],ignore_index=True)

    buy_df = main_df.loc[(main_df['Buy/Sell'] == 'Buy')]
    sell_df = main_df.loc[(main_df['Buy/Sell'] == 'Sell')]

    buy_df['Buy'] = buy_df['Buy/Sell']
    buy_df['Purch Price'] = buy_df['Price']
    buy_df['Purch Time'] = buy_df['Time']
    buy_df['Shares Purch'] = buy_df['Shares']

    sell_df['Sale'] = sell_df['Buy/Sell']
    sell_df['Sale Price'] = sell_df['Price']
    sell_df['Sale Time'] = sell_df['Time']
    sell_df['Shares Sold'] = sell_df['Shares']

    buy_df = buy_df.drop(['Buy/Sell','Price','Time','Shares'], axis=1)
    sell_df = sell_df.drop(['Buy/Sell','Price','Time','Shares'], axis=1)

    combined_df = pd.merge(buy_df, sell_df,  how='left', on = ['Ticker','Hour','Minute'])
    combined_df = combined_df.drop(['Buy','Sale'], axis=1)
    
    combined_df['% Return'] = (combined_df['Sale Price'] - combined_df['Purch Price']) / combined_df['Purch Price']
    combined_df['P(L)'] = combined_df['% Return'] * (combined_df['Shares Purch'] * combined_df['Purch Price'])
    
    combined_df.sort_values("Purch Time", inplace = True, ignore_index=True)


    destination = pathlib.Path.cwd() / 'Logs' / 'Daily Trade Logs' / str(today.year) 
    if not destination.exists():
        destination.mkdir(parents=True, exist_ok=True)

    # Set Filename
    file_name = today.strftime('%Y-%m-%d') + ' Trade Log.csv'
    file = pathlib.Path.cwd() / 'Logs' / 'Daily Trade Logs' / str(today.year) / file_name

    combined_df.to_csv(file, index=False)
    
    file_name = today.strftime('%Y-%m-%d') + ' Trade Log.csv'
    file = pathlib.Path.cwd() / 'Logs' / 'Daily Trade Logs' / str(today.year) / file_name
    day_trade_log = pd.read_csv(file)

    print("Paper Profit(L):", round(np.nansum(day_trade_log['P(L)']),2))
    print("Average % Return:",round(np.nanmean(day_trade_log['% Return'])*100,2),"%")
    
    return combined_df

def clear_jobs():
    schedule.clear()
    print('Jobs Cleared')
    
    return

'''
##################################################################################
################# Order Book Functionality
##################################################################################
''' 

def make_order_book(filtered_tups):
    order_book = pd.DataFrame()

    tickers = []
    hours = []
    minutes = []

    for tup in filtered_tups:
        tickers.append(tup[0])
        hours.append(tup[2])
        minutes.append(tup[3])

    order_book['Ticker'] = tickers
    order_book['Hour'] = hours
    order_book['Minute'] = minutes
    order_book['Action'] = 0

    return order_book

def execute_order_book(hour, minute):
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('Executing Order Book')
    # Get stocks for the relevant window
    relevant_tup = [x for x in filtered_tups if x[2] == hour and x[3] == minute]
    cos_to_buy = []
    
    # If stocks exist for this window
    if len(relevant_tup) > 0:
    
        for tup in relevant_tup:
            # Get list of stock's to buy (1) from order book
            co = tup[0]
            try:
                action = list(order_book.loc[(order_book.Ticker == co) & (order_book.Hour == hour) & (order_book.Minute == minute)]['Action'])[0]
            except:
                action = 0
                
            if action == 1:
                cos_to_buy.append(co)

        if len(cos_to_buy) == 0:
            print('No Stocks for this window: ', hour,':',minute,' were predicted > .50')
        
        # Determine amount to invest in each stock
        if max_capital_per_trade * len(cos_to_buy) > max_capital:
            amount_to_invest = math.floor(max_capital / len(cos_to_buy))
        else:
            amount_to_invest = max_capital_per_trade

        # Go through list of stocks to buy
        for ticker in cos_to_buy:
            # Buy and log result
            try:
                shares = buy_job(ticker,hour,minute,amount_to_invest)
            except:
                pass
            
            if shares > 0:
            
                # Schedule Sale
                sell_minute = minute + 29
                sell_minute = str(sell_minute)

                # tz_offset
                sell_hour = hour + tz_offset
                #sell_hour = hour - 2
                sell_hour = str(sell_hour)

                if len(sell_hour) == 1:
                    sell_hour = '0'+sell_hour

                sell_time = sell_hour+":"+sell_minute+":00"
                sell_time2 = sell_hour+":"+sell_minute+":45"

                schedule.every().day.at(sell_time).do(jobqueue.put, (sell_job, [ticker,hour,minute,shares]))
                schedule.every().day.at(sell_time2).do(jobqueue.put, (sell_job, [ticker,hour,minute,shares]))

    else:
        print('No Jobs to Execute at:',hour,":",minute)
        
    print('Order Book Executed')
            
    return

def secondary_order_book_execution(hour, minute):
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('Secondary Order Book Execution')
    
    get_current_positions()
    
    # Get stocks for the relevant window
    relevant_tup = [x for x in filtered_tups if x[2] == hour and x[3] == minute]
    cos_to_buy = []
    
    # If stocks exist for this window
    if len(relevant_tup) > 0:
    
        for tup in relevant_tup:
            # Get list of stock's to buy (1) from order book
            co = tup[0]
            try:
                action = list(order_book.loc[(order_book.Ticker == co) & (order_book.Hour == hour) & (order_book.Minute == minute)]['Action'])[0]
            except:
                action = 0
                
            if action == 1:
                cos_to_buy.append(co)

        if len(cos_to_buy) == 0:
            print('No Stocks for this window: ', hour,':',minute,' were predicted > .50')
            
        # Determine amount to invest in each stock
        if max_capital_per_trade * len(cos_to_buy) > max_capital:
            amount_to_invest = math.floor(max_capital / len(cos_to_buy))
        else:
            amount_to_invest = max_capital_per_trade
            
        # Make list of stocks currently held
        cos_currently_held = list(current_positions.keys())
        
        # Buy cos that should have been bought the first time, but weren't
        for ticker in cos_to_buy:
            if ticker not in cos_currently_held:
                # Buy and log result
                try:
                    shares = buy_job(ticker,hour,minute,amount_to_invest)
                except:
                    shares = 0

                if shares > 0:

                    # Schedule Sale
                    sell_minute = minute + 29
                    sell_minute = str(sell_minute)

                    # tz_offset
                    sell_hour = hour + tz_offset
                    #sell_hour = hour - 2
                    sell_hour = str(sell_hour)

                    if len(sell_hour) == 1:
                        sell_hour = '0'+sell_hour

                    sell_time = sell_hour+":"+sell_minute+":00"
                    sell_time2 = sell_hour+":"+sell_minute+":45"

                    schedule.every().day.at(sell_time).do(jobqueue.put, (sell_job, [ticker,hour,minute,shares]))
                    schedule.every().day.at(sell_time2).do(jobqueue.put, (sell_job, [ticker,hour,minute,shares]))    
            
    else:
        print('No Jobs to Execute at:',hour,":",minute)
        
    print('Order Book Executed')

def schedule_order_book_execution():
    minute_list = [0,30]
    for h in range(9,16):
        for m in minute_list:
            if (h,m) != (9,0):
                relevant_tup = [x for x in filtered_tups if x[2] == h and x[3] == m]

                # tz_offset
                sched_hour = str(h + tz_offset)
                if len(sched_hour) == 1:
                    sched_hour = "0"+sched_hour

                if m == 0:
                    sched_minute = "00"
                else:
                    sched_minute = '30'

                sched_time = sched_hour+":"+sched_minute+":40"

                schedule.every().day.at(sched_time).do(jobqueue.put, (execute_order_book, [h,m]))

    return

def rh_logout():
    r.authentication.logout()
    print("Logged Out")
    return

def get_current_positions():
    
    print('++++++++++++++++++++++++++++++++++++++++++++++++')
    #print('Updating Positions')
    current_stocks = []
    
    # Current Positions
    current_holdings = r.build_holdings()

    for symbol_, info in current_holdings.items():
        current_positions[symbol_] = info['quantity']
        current_stocks.append(symbol_)
    
    stocks_to_remove = []
    for symbol_, info in current_positions.items():
        if symbol_ not in current_stocks:
            current_positions[symbol_] = 0
            stocks_to_remove.append(symbol_)
    
    for symbol_ in stocks_to_remove:
        del current_positions[symbol_]
        
    print('Positions Updated')
        
    return

def clear_rh_orders():
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('Clearing Open Orders')
    open_orders = r.get_all_open_stock_orders()
    
    open_order_ids = []

    for item in open_orders:
        open_order_ids.append(item['id'])
        
    # Set Filename
    file_name = today.strftime('%Y-%m-%d') + ' Trade IDs.csv'
    file = pathlib.Path.cwd() / 'Logs' / 'RH Logs' / str(today.year) / 'Trade IDs' / 'Buy' / file_name
    buy_orders_df = pd.read_csv(file)
    buy_orders = list(buy_orders_df['Trade ID'])
    
    file_name = today.strftime('%Y-%m-%d') + ' Trade IDs.csv'
    file = pathlib.Path.cwd() / 'Logs' / 'RH Logs' / str(today.year) / 'Trade IDs' / 'Sell' / file_name
    sell_orders_df = pd.read_csv(file)
    sell_orders = list(sell_orders_df['Trade ID'])
    
    # cancel all orders in the lists at the end of the day 
    for order in buy_orders:
        if order in open_order_ids:
            r.cancel_stock_order(order)
    for order in sell_orders:
        if order in open_order_ids:
            r.cancel_stock_order(order)
    
    print('Orders Cleared')
    
    return

def rh_trade_id_logger(BuyorSell, trade_id):
    destination = pathlib.Path.cwd() / 'Logs' / 'RH Logs' / str(today.year) / 'Trade IDs' / BuyorSell

    if not destination.exists():
        destination.mkdir(parents=True, exist_ok=True)
        
    # Set Filename
    file_name = today.strftime('%Y-%m-%d') + ' Trade IDs.csv'
    file = pathlib.Path.cwd() / 'Logs' / 'RH Logs' / str(today.year) / 'Trade IDs' / BuyorSell / file_name
    
    
    cols = ['Trade ID']
    df_new_line = pd.DataFrame([[trade_id]], columns=cols)
    
    if not file.is_file():
        df_new_line.to_csv(file, index=False)     
        
    else:
        main_df = pd.read_csv(file)
        main_df = pd.concat([main_df,df_new_line],ignore_index=True)
        main_df.to_csv(file, index=False)  
        
    return

def download_and_consolidate_rh_trades(period_beg,period_end, day_to_log):
    
    if day_to_log == 'today':
        day_to_log = today.strftime('%Y-%m-%d')
        
    all_orders = r.get_all_stock_orders()
    
    all_buy_orders = []
    all_sell_orders = []

    for order in all_orders:
        if order['state'] == 'filled':
            if order['side'] == 'buy':
                all_buy_orders.append(order)
            elif order['side'] == 'sell':
                all_sell_orders.append(order)

    period_buy_orders = []
    period_sell_orders = []

    for order in all_buy_orders:
        if order['last_transaction_at'].split("T")[0] >= period_beg and order['last_transaction_at'].split("T")[0] <= period_end:
            period_buy_orders.append(order)

    for order in all_sell_orders:
        if order['last_transaction_at'].split("T")[0] >= period_beg and order['last_transaction_at'].split("T")[0] <= period_end:
            period_sell_orders.append(order)

    period_buy_ids = []
    period_sale_ids = []

    for buy_order in period_buy_orders:
        period_buy_ids.append(buy_order['id'])

    for sell_order in period_sell_orders:
        period_sale_ids.append(sell_order['id'])

    matched_orders = []

    buy_order_info = []
    sell_order_info = []

    for order in period_buy_orders:
        order_dict = {}
        order_dict['id'] = order['id']
        order_dict['instrument_id'] = order['instrument_id']
        order_dict['price'] = float(order['average_price'])
        order_dict['quantity'] = float(order['cumulative_quantity'])
        order_dict['rounded_notional'] = float(order['executed_notional']['amount'])
        order_dict['timestamp'] = order['last_transaction_at']
        order_dict['Short Date'] = order['last_transaction_at'].split('T')[0]
        order_dict['side'] = order['side']

        buy_order_info.append(order_dict)

    for order in period_sell_orders:
        order_dict = {}
        order_dict['id'] = order['id']
        order_dict['instrument_id'] = order['instrument_id']
        order_dict['price'] = float(order['average_price'])
        order_dict['quantity'] = float(order['cumulative_quantity'])
        order_dict['rounded_notional'] = float(order['executed_notional']['amount'])
        order_dict['timestamp'] = order['last_transaction_at']
        order_dict['Short Date'] = order['last_transaction_at'].split('T')[0]
        order_dict['side'] = order['side']

        sell_order_info.append(order_dict)


    trade_df = pd.DataFrame(columns=['Short Date','Ticker','InstrumentID','Buy ID','Buy Price','Buy Quantity','Buy Notional','Buy Timestamp','Sale ID',
                                    'Sale Price','Sale Quantity','Sale Notional','Sale Timestamp'])
    # First Pass

    for buy_order in buy_order_info:
        for sell_order in sell_order_info:
            if buy_order['instrument_id'] == sell_order['instrument_id'] and buy_order['Short Date'] == sell_order['Short Date'] and buy_order['quantity'] == sell_order['quantity']:
                # if buy order and sell order id still not matched yet
                if buy_order['id'] in period_buy_ids and sell_order['id'] in period_sale_ids:

                    # if buy timestamp < sale timestamp
                    if buy_order['timestamp'] < sell_order['timestamp']:

                        stock_quote = r.get_stock_quote_by_id(buy_order['instrument_id'])
                        ticker = stock_quote['symbol']


                        data = [[buy_order['Short Date'], ticker, buy_order['instrument_id'],buy_order['id'],
                        buy_order['price'],buy_order['quantity'],buy_order['rounded_notional'],
                        buy_order['timestamp'],sell_order['id'],sell_order['price'],
                        sell_order['quantity'],sell_order['rounded_notional'],sell_order['timestamp']]]

                        new_trade = pd.DataFrame(data, columns=['Short Date','Ticker','InstrumentID','Buy ID','Buy Price','Buy Quantity','Buy Notional','Buy Timestamp','Sale ID',
                                                'Sale Price','Sale Quantity','Sale Notional','Sale Timestamp'])

                        trade_df = pd.concat([trade_df,new_trade], ignore_index=True)
                        matched_orders.append(buy_order['id'])
                        matched_orders.append(sell_order['id'])

                        period_buy_ids.remove(buy_order['id'])
                        period_sale_ids.remove(sell_order['id'])

    trade_df.sort_values("Buy Timestamp", inplace = True, ignore_index=True)

    # Second Pass

    for sale_order in period_sale_ids:
        sale_order_info = r.get_stock_order_info(sale_order)
        sale_instrument = sale_order_info['instrument_id']
        sale_shares = float(sale_order_info['cumulative_quantity'])
        short_date = sale_order_info['last_transaction_at'].split('T')[0]
        sale_price = float(sale_order_info['average_price'])

        stock_quote = r.get_stock_quote_by_id(sale_instrument)
        ticker = stock_quote['symbol']

        possible_buys = []
        for buy_order in period_buy_ids:
            buy_order_info = r.get_stock_order_info(buy_order)

            if buy_order_info['instrument_id'] == sale_instrument:
                possible_buys.append(buy_order)

        shares_bought = 0
        notional = 0

        for buy_order in possible_buys:
            buy_order_info = r.get_stock_order_info(buy_order)
            shares_bought += round(float(buy_order_info['cumulative_quantity']),4)
            notional += float(buy_order_info['executed_notional']['amount'])

        if round(shares_bought,2) == round(sale_shares,2):

            data = [[short_date, ticker, buy_order_info['instrument_id'],buy_order_info['id'],
                        notional/shares_bought,shares_bought,notional,
                        buy_order_info['last_transaction_at'],sale_order_info['id'],sale_price,
                        sale_order_info['cumulative_quantity'],float(sale_order_info['executed_notional']['amount']),sale_order_info['last_transaction_at']]]

            new_trade = pd.DataFrame(data, columns=['Short Date','Ticker','InstrumentID','Buy ID','Buy Price','Buy Quantity','Buy Notional','Buy Timestamp','Sale ID',
                                                'Sale Price','Sale Quantity','Sale Notional','Sale Timestamp'])

            trade_df = pd.concat([trade_df,new_trade], ignore_index=True)



        for buy_order in possible_buys:
            period_buy_ids.remove(buy_order)

        period_sale_ids.remove(sale_order)

    trade_df.sort_values("Buy Timestamp", inplace = True, ignore_index=True)
    
    trade_df['% Return'] = (trade_df['Sale Notional'] - trade_df['Buy Notional']) / trade_df['Buy Notional']
    trade_df['P(L)'] = trade_df['Sale Notional'] - trade_df['Buy Notional']
    
    destination = pathlib.Path.cwd() / 'Logs' / 'Daily Trade Logs' / str(today.year) 
    if not destination.exists():
        destination.mkdir(parents=True, exist_ok=True)

    # Set Filename
    file_name = day_to_log + ' RH Trade Log Downloaded.csv'
    file = pathlib.Path.cwd() / 'Logs' / 'RH Logs' / str(today.year) / file_name

    trade_df.to_csv(file, index=False)
    
    print("RH Downloaded Trades Profit(L):", round(np.nansum(trade_df['P(L)']),2))
    print("Average % Return:",round(np.nanmean(trade_df['% Return'])*100,2),"%")

    return trade_df

def rh_trade_consolidator_local(day_to_log):
    
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('Consolidating Local RH Trades')
    
    if day_to_log == 'today':
        day_to_log = today.strftime('%Y-%m-%d')

    matched_orders = []

    buy_order_info = []
    sell_order_info = []

    # Set Filename
    file_name = day_to_log + ' Trade IDs.csv'
    #file_name = today.strftime('%Y-%m-%d') + ' Trade IDs.csv'
    file = pathlib.Path.cwd() / 'Logs' / 'RH Logs' / str(today.year) / 'Trade IDs' / 'Buy' / file_name
    buy_orders_df = pd.read_csv(file)
    buy_orders = list(buy_orders_df['Trade ID'])

    file_name = day_to_log + ' Trade IDs.csv'
    #file_name = today.strftime('%Y-%m-%d') + ' Trade IDs.csv'
    file = pathlib.Path.cwd() / 'Logs' / 'RH Logs' / str(today.year) / 'Trade IDs' / 'Sell' / file_name
    sell_orders_df = pd.read_csv(file)
    sell_orders = list(sell_orders_df['Trade ID'])

    for order in buy_orders:
        order_dict = {}

        order_info = r.get_stock_order_info(order)
        
        if order_info['state'] == 'filled':

            order_dict['id'] = order_info['id']
            order_dict['instrument_id'] = order_info['instrument_id']
            order_dict['price'] = float(order_info['average_price'])
            order_dict['quantity'] = float(order_info['cumulative_quantity'])
            order_dict['rounded_notional'] = float(order_info['executed_notional']['amount'])
            order_dict['timestamp'] = order_info['last_transaction_at']
            order_dict['Short Date'] = order_info['last_transaction_at'].split('T')[0]
            order_dict['side'] = order_info['side']

            buy_order_info.append(order_dict)
            
        else:
            pass

    for order in sell_orders:
        order_dict = {}

        order_info = r.get_stock_order_info(order)
        
        if order_info['state'] == 'filled':

            order_dict['id'] = order_info['id']
            order_dict['instrument_id'] = order_info['instrument_id']
            order_dict['price'] = float(order_info['average_price'])
            order_dict['quantity'] = float(order_info['cumulative_quantity'])
            order_dict['rounded_notional'] = float(order_info['executed_notional']['amount'])
            order_dict['timestamp'] = order_info['last_transaction_at']
            order_dict['Short Date'] = order_info['last_transaction_at'].split('T')[0]
            order_dict['side'] = order_info['side']

            sell_order_info.append(order_dict)
        
        else:
            pass

    trade_df = pd.DataFrame(columns=['Short Date','Ticker','InstrumentID','Buy ID','Buy Price','Buy Quantity','Buy Notional','Buy Timestamp','Sale ID',
                                    'Sale Price','Sale Quantity','Sale Notional','Sale Timestamp'])
    # First Pass

    for buy_order in buy_order_info:
        for sell_order in sell_order_info:
            if buy_order['instrument_id'] == sell_order['instrument_id'] and buy_order['Short Date'] == sell_order['Short Date'] and buy_order['quantity'] == sell_order['quantity']:
                # if buy order and sell order id still not matched yet
                if buy_order['id'] in buy_orders and sell_order['id'] in sell_orders:

                    # if buy timestamp < sale timestamp
                    if buy_order['timestamp'] < sell_order['timestamp']:

                        stock_quote = r.get_stock_quote_by_id(buy_order['instrument_id'])
                        ticker = stock_quote['symbol']


                        data = [[buy_order['Short Date'], ticker, buy_order['instrument_id'],buy_order['id'],
                        buy_order['price'],buy_order['quantity'],buy_order['rounded_notional'],
                        buy_order['timestamp'],sell_order['id'],sell_order['price'],
                        sell_order['quantity'],sell_order['rounded_notional'],sell_order['timestamp']]]

                        new_trade = pd.DataFrame(data, columns=['Short Date','Ticker','InstrumentID','Buy ID','Buy Price','Buy Quantity','Buy Notional','Buy Timestamp','Sale ID',
                                                'Sale Price','Sale Quantity','Sale Notional','Sale Timestamp'])

                        trade_df = pd.concat([trade_df,new_trade], ignore_index=True)
                        matched_orders.append(buy_order['id'])
                        matched_orders.append(sell_order['id'])

                        buy_orders.remove(buy_order['id'])
                        sell_orders.remove(sell_order['id'])

    trade_df.sort_values("Buy Timestamp", inplace = True, ignore_index=True)

    # Second Pass

    for sale_order in sell_orders:
        sale_order_info = r.get_stock_order_info(sale_order)
        sale_instrument = sale_order_info['instrument_id']
        sale_shares = float(sale_order_info['cumulative_quantity'])
        short_date = sale_order_info['last_transaction_at'].split('T')[0]
        sale_price = float(sale_order_info['average_price'])

        stock_quote = r.get_stock_quote_by_id(sale_instrument)
        ticker = stock_quote['symbol']

        possible_buys = []
        for buy_order in buy_orders:
            buy_order_info = r.get_stock_order_info(buy_order)

            if buy_order_info['instrument_id'] == sale_instrument:
                possible_buys.append(buy_order)

        shares_bought = 0
        notional = 0

        for buy_order in possible_buys:
            buy_order_info = r.get_stock_order_info(buy_order)
            shares_bought += round(float(buy_order_info['cumulative_quantity']),4)
            notional += float(buy_order_info['executed_notional']['amount'])

        if round(shares_bought,2) == round(sale_shares,2):

            data = [[short_date, ticker, buy_order_info['instrument_id'],buy_order_info['id'],
                        notional/shares_bought,shares_bought,notional,
                        buy_order_info['last_transaction_at'],sale_order_info['id'],sale_price,
                        sale_order_info['cumulative_quantity'],float(sale_order_info['executed_notional']['amount']),sale_order_info['last_transaction_at']]]

            new_trade = pd.DataFrame(data, columns=['Short Date','Ticker','InstrumentID','Buy ID','Buy Price','Buy Quantity','Buy Notional','Buy Timestamp','Sale ID',
                                                'Sale Price','Sale Quantity','Sale Notional','Sale Timestamp'])

            trade_df = pd.concat([trade_df,new_trade], ignore_index=True)       


        for buy_order in possible_buys:
            buy_orders.remove(buy_order)

        sell_orders.remove(sale_order)

    trade_df.sort_values("Buy Timestamp", inplace = True, ignore_index=True)

    trade_df['% Return'] = (trade_df['Sale Notional'] - trade_df['Buy Notional']) / trade_df['Buy Notional']
    trade_df['P(L)'] = trade_df['Sale Notional'] - trade_df['Buy Notional']

    destination = pathlib.Path.cwd() / 'Logs' / 'Daily Trade Logs' / str(today.year) 
    if not destination.exists():
        destination.mkdir(parents=True, exist_ok=True)

    # Set Filename
    file_name = day_to_log + ' RH Trade Log.csv'
    file = pathlib.Path.cwd() / 'Logs' / 'RH Logs' / str(today.year) / file_name

    trade_df.to_csv(file, index=False)

    day_trade_log = pd.read_csv(file)

    print("RH Local Trades Profit(L):", round(np.nansum(day_trade_log['P(L)']),2))
    print("Average % Return:",round(np.nanmean(day_trade_log['% Return'])*100,2),"%")
    
    return day_trade_log

###################################################################################################
# Setup
###################################################################################################

# Get the day's potential trades
tups = get_stocks_periods()

# Filter by weekday
filtered_tups = filter_tup_by_weekday(tups)
filtered_tups = sorted(filtered_tups, key = lambda x: (x[2], x[3]))

# Separate into first period and later periods
first_period_tups = [x for x in filtered_tups if x[2] == 9 and  x[3] == 30]
later_period_tups = [x for x in filtered_tups if x not in first_period_tups]
later_period_tups = sorted(later_period_tups, key = lambda x: (x[2], x[3]))

later_CoList = [x[0] for x in later_period_tups]
later_CoList = sorted(list(set(list(later_CoList))))

all_CoList = [x[0] for x in filtered_tups]
all_CoList = sorted(list(set(list(all_CoList))))

order_book = make_order_book(filtered_tups)

###################################################################################################
# Setup Holding Dict
###################################################################################################

main_dict = {}

for co in tqdm(all_CoList):
    temp_dict = {}
    temp_dict['Model'] = load_a_model(delay, co)
    temp_dict['Scaler'] = pickle_load_scaler(delay, co)
    temp_dict['Cleaned_Combined'] = load_last_5_days_of_combined_data(co)
    temp_dict['Streaming'] = True
    if co in proxies:
        temp_dict['Proxy'] = True
    else:
        temp_dict['Proxy'] = False
    
    main_dict[co] = temp_dict
    
proxy_dict = {}

for co in tqdm(proxies):
    temp_dict = {}
    temp_dict['Model'] = load_a_model(delay, co)
    temp_dict['Scaler'] = pickle_load_scaler(delay, co)
    temp_dict['Cleaned_Combined'] = load_last_5_days_of_combined_data(co)
    temp_dict['Streaming'] = True
    if co in proxies:
        temp_dict['Proxy'] = True
    else:
        temp_dict['Proxy'] = False
    
    proxy_dict[co] = temp_dict

clear_jobs()

jobqueue = queue.Queue()

###################################################################################################
# Daily Task Scheduling
###################################################################################################

# Schedule First Period Buys
for tup in tqdm(first_period_tups):
    co = tup[0]
    weekday = tup[1]
    hour = tup[2]
    minute = tup[3]

    mongo_df_stationary = load_last_3_days_of_stationary_data(co)
    test_data = mongo_df_stationary.to_numpy()
    pred = make_early_prediction(co, delay, test_data)

    if pred > .50:
        print("Buy",co)
        order_book.loc[(order_book.Ticker == co) & (order_book.Hour == tup[2]) & (order_book.Minute == tup[3]), 'Action'] = 1

    else:
        print("Don't buy", co)

########################### Schedule Proxy Updater ########################################
for symbol in proxies:
    # tz_offset 
    for h in range(first_hour_local + 1, last_hour_local + 1):
        sched_hour = str(h)
        if len(sched_hour) == 1:
            sched_hour = '0'+sched_hour
        sched_time = sched_hour+":00:20"
        schedule.every().day.at(sched_time).do(jobqueue.put, (proxy_job, [symbol]))

    # tz_offset
    for h in range(first_hour_local + 1, last_hour_local + 1):
        sched_hour = str(h)
        if len(sched_hour) == 1:
            sched_hour = '0'+sched_hour
        sched_time = sched_hour+":30:20"
        schedule.every().day.at(sched_time).do(jobqueue.put, (proxy_job, [symbol]))

############################## Schedule Later Period Jobs ####################################
for tup in later_period_tups:
    
    symbol = tup[0]
    day = tup[1]
    hour = tup[2]
    minute = tup[3]
    
    # tz_offset
    hour = str(hour + tz_offset)
    if len(hour) == 1:
        hour = "0"+hour
    
    if minute == 0:
        minute = "00"
    else:
        minute = '30'
    
    sched_time = hour+":"+minute+":30"

    schedule.every().day.at(sched_time).do(jobqueue.put, (later_period_job, [tup]))

####################### Schedule Order Book Execution #################################
schedule_order_book_execution()

####################### Schedule Secondary Order Book Execution #################################
minute_list = [0,30]
for h in range(9,16):
    for m in minute_list:
        if (h,m) != (9,0):
            relevant_tup = [x for x in filtered_tups if x[2] == h and x[3] == m]

            # tz_offset
            sched_hour = str(h + tz_offset)
            #sched_hour = str(h - 2)
            if len(sched_hour) == 1:
                sched_hour = "0"+sched_hour

            if m == 0:
                sched_minute = "02"
            else:
                sched_minute = "32"

            sched_time = sched_hour+":"+sched_minute+":00"

            schedule.every().day.at(sched_time).do(jobqueue.put, (secondary_order_book_execution, [h,m]))

############################# Schedule the update of current positions ################################
current_positions = {}
get_current_positions()

for h in range(first_hour_local, last_hour_local+1):
    '''
    if h == first_hour_local:
        sched_hour = str(h)
        if len(sched_hour) == 1:
            sched_hour = '0' + sched_hour
        sched_time = sched_hour + ':31:30'
        schedule.every().day.at(sched_time).do(jobqueue.put,(get_current_positions, []))
    '''
    # else:
    sched_hour = str(h)
    if len(sched_hour) == 1:
        sched_hour = '0' + sched_hour
    
    # Update Before Second Sale to see if any shares failed to sell on the first pass
    sched_time = sched_hour + ':29:20'
    sched_time1 = sched_hour + ':59:20'

    # Schedule after all order book executions, to update positions for firt sale attempt
    sched_time2 = sched_hour + ':05:00'
    sched_time3 = sched_hour + ':35:00'

    schedule.every().day.at(sched_time).do(jobqueue.put,(get_current_positions, []))
    schedule.every().day.at(sched_time1).do(jobqueue.put,(get_current_positions, []))
    schedule.every().day.at(sched_time2).do(jobqueue.put,(get_current_positions, []))
    schedule.every().day.at(sched_time3).do(jobqueue.put,(get_current_positions, []))


################  Schedule Post Trading Day Functions (After 2:00 PM Mountain) ##########################

# Run Trade Consolidation For Paper Trades and Summarize Results
sched_hour = str(last_hour_local + 1)
if len(sched_hour) == 1:
    sched_hour = "0" + sched_hour
sched_time = sched_hour + ':02'
schedule.every().day.at(sched_time).do(jobqueue.put,(consolidate_trades, []))

# Run Trade Consolidation For Robinhood and Summarize Results
sched_time = sched_hour + ':03'
schedule.every().day.at(sched_time).do(jobqueue.put,(rh_trade_consolidator_local, ['today']))

# Run Back test and Summarize Results
sched_time = sched_hour + ':05'
schedule.every().day.at(sched_time).do(jobqueue.put,(run_backtest, ['today']))

# Run Passive Back test and Summarize Results
sched_time = sched_hour + ':10'
schedule.every().day.at(sched_time).do(jobqueue.put,(run_passive_backtest, ['today']))

# Download and Consolidate RH Trades
sched_time = sched_hour + ':12'
today_str = today.strftime('%Y-%m-%d')
schedule.every().day.at(sched_time).do(jobqueue.put,(download_and_consolidate_rh_trades, [today_str,today_str,today_str]))

# Clear any of the session's open orders on Robinhood
sched_time = sched_hour + ':14'
schedule.every().day.at(sched_time).do(jobqueue.put,(clear_rh_orders, []))

# Logout of Robinhood
sched_time = sched_hour + ':14:30'
schedule.every().day.at(sched_time).do(jobqueue.put,(rh_logout, []))

# Clear Jobs
sched_time = sched_hour + ':15'
schedule.every().day.at(sched_time).do(jobqueue.put,(clear_jobs, []))

all_jobs = schedule.get_jobs()

###################################################################################################
# Main Function: Job Running
###################################################################################################
worker_thread = threading.Thread(target=worker_main)
worker_thread.start()

while True:
    try:
        schedule.run_pending()
        time.sleep(1)
    except Exception as e:
        print(e)
        exit