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
import pytz
from pytz import timezone
from datetime import datetime, timedelta
from time import sleep
from tqdm import tqdm
from pymongo import MongoClient,InsertOne
from keras.models import load_model
from real_time_trading import run_backtest
from real_time_trading import run_passive_backtest
from real_time_trading import consolidate_trades
from real_time_trading import rh_trade_consolidator_local
from real_time_trading import download_and_consolidate_rh_trades

###################################################################################################
# Day Strategy Backtest
###################################################################################################
df = run_backtest("today")

###################################################################################################
# Day Passive Backtest
###################################################################################################
df1 = run_passive_backtest("today")

###################################################################################################
# Consolidate Paper Trades
###################################################################################################
df2 = consolidate_trades()

###################################################################################################
# Consolidate Local RH Trades
###################################################################################################
login = r.login(rh_info['User'][0],rh_info['Pass'][0])
df3 = rh_trade_consolidator_local('today')
rh_logout()

###################################################################################################
# Download and Consolidate RH Trades
###################################################################################################
login = r.login(rh_info['User'][0],rh_info['Pass'][0])
df4 = download_and_consolidate_rh_trades(today.strftime('%Y-%m-%d'),today.strftime('%Y-%m-%d'),today.strftime('%Y-%m-%d'))
rh_logout()

###################################################################################################
# Summarize Results
###################################################################################################
first_day = 1 
last_day = 31
first_month = 1
last_month = 2
year = 2022

dates = []
paper_trades_profits = []
local_rh_trades_profits = []
dl_rh_trades_profits = []
backtest_profits = []
passive_backtest_profits = []

for month in range(first_month,last_month+1):
    for day in range(first_day,last_day+1):
        if len(str(day)) == 1:
            str_day = '0'+str(day)
        else:
            str_day = str(day)
        if len(str(month)) == 1:
            str_month = '0'+str(month)
        else:
            str_month = str(month)

        day_to_log = str(year) + "-" + str_month + "-" + str_day
        dates.append(day_to_log)

        # Paper Trades
        try:
            file_name = day_to_log + ' Trade Log.csv'
            file = pathlib.Path.cwd() / 'Logs' / 'Daily Trade Logs' / str(today.year) / file_name
            day_trade_log = pd.read_csv(file)
            profit = round(np.nansum(day_trade_log['P(L)']),2)
        except:
            profit = float('NaN')
        paper_trades_profits.append(profit)

        # Local RH Trades
        try:
            file_name = day_to_log + ' RH Trade Log.csv'
            file = pathlib.Path.cwd() / 'Logs' / 'RH Logs' / str(today.year) / file_name
            day_trade_log = pd.read_csv(file)
            profit = round(np.nansum(day_trade_log['P(L)']),2)
        except:
            profit = float('NaN')
        local_rh_trades_profits.append(profit)

        # Downloaded RH Trades
        try:
            file_name = day_to_log + ' RH Trade Log Downloaded.csv'
            file = pathlib.Path.cwd() / 'Logs' / 'RH Logs' / str(today.year) / file_name
            day_trade_log = pd.read_csv(file)
            profit = round(np.nansum(day_trade_log['P(L)']),2)
        except:
            profit = float('NaN')
        dl_rh_trades_profits.append(profit)

        # Strategy Backtest Trades
        try:
            file_name = day_to_log + ' Backtest.csv'
            file = pathlib.Path.cwd() / 'Logs' / 'Daily Trade Logs' / str(today.year) / file_name
            day_backtest = pd.read_csv(file)
            profit = round(sum(day_backtest['P(L)']),2)
        except:
            profit = float('NaN')
        backtest_profits.append(profit)

        # Passive Back Test Trades
        try:
            file_name = day_to_log + ' Passive Backtest.csv'
            file = pathlib.Path.cwd() / 'Logs' / 'Daily Trade Logs' / str(today.year) / file_name
            day_backtest = pd.read_csv(file)
            profit = round(sum(day_backtest['P(L)']),2)
        except:
            profit = float('NaN')
        passive_backtest_profits.append(profit)
        
results = pd.DataFrame()
results['Date'] = dates
results['Paper P(L)'] = paper_trades_profits
results['Local RH P(L)'] = local_rh_trades_profits
results['DL RH P(L)'] = dl_rh_trades_profits
results['Backtest P(L)'] = backtest_profits
results['Pass. Backtest P(L)'] = passive_backtest_profits
results['Active vs Pass'] = results['Backtest P(L)'] - results['Pass. Backtest P(L)']

print_cols = [x for x in results.columns if x != 'Date']

print('++++++++++++++++++++++++++++++++++++++++++')

for col in print_cols:
    total = round(np.nansum(results[col]),2)
    print(col,'Total:',total)
    
print('++++++++++++++++++++++++++++++++++++++++++')

for col in print_cols:
    average = round(np.nanmean(results[col]),2)
    print(col,'Average:',average)
    
print('++++++++++++++++++++++++++++++++++++++++++')

results.dropna()