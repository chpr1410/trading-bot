# Algorithmic Trading Using Deep Learning and Robinhood's Unofficial API

## File Structure
- [real_time_trading.py](https://github.com/chpr1410/trading-bot/blob/main/real_time_trading.py)
- [results.py](https://github.com/chpr1410/trading-bot/blob/main/results.py)
- [update_db_daily.py](https://github.com/chpr1410/trading-bot/blob/main/update_db_daily.py)
- [update_db_weekly.py](https://github.com/chpr1410/trading-bot/blob/main/update_db_weekly.py)

## Project Overview

In this project, I extend one of my [earlier capstone projects](https://github.com/chpr1410/MSDS696-Practicum).  That project used deep learning to find historical pricing patterns in 54 Exchange Traded Funds (ETFs).  This project takes the models from that capstone projects and implements them in a real time trading strategy.  This "Trading Bot", combines the previously trained models, a stream of current stock price data, and the unofficial API for the e-brokerage Robhinhood in one application.  This application could be used to deploy a real time algorithmic stock trading strategy.

Additional functionality of the "Trading Bot" includes logging trades, backtesting both the models' strategy and a passive strategy, and displaying results daily, including profit (loss) and average return per trade.  This trade logging is crucial to understanding the results and the trades that contributed to the overall performance.

** The information in this document is for informational and educational purposes only. Nothing in this document may be construed as financial, legal or tax advice. The content of this document is solely the opinion of the author, who is not a licensed financial advisor or registered investment advisor.

This document is not an offer to buy or sell financial instruments. Never invest more than you can afford to lose. You should consult a registered professional advisor before making any investment. **

## Data Acquisition and Processing

Real time stock pricing data is aquired via the python package [yFinance](https://pypi.org/project/yfinance/) and via API from data provider [TwelveData](https://twelvedata.com/)

Daily and weekly updating of the MongoDB Database utilizes stock price data from [Alpha Vantage](https://www.alphavantage.co/) via API calls.  

## Process Map

Hypothetically, the "Trading Bot" could be initialized each morning before the stock market opens.  This would be contained in the real_time_trading.py file.  The "Trading Bot" utilizes the [unofficial API of Robinhood](https://robin-stocks.readthedocs.io/en/latest/index.html)

Each morning, the MongoDB Database would need to be updated to include the prior days' data for the relevant stocks for that day.  This is contained in the "update_db_daily.py" file.  At the end of each trading week, data for all companies would be updated using the "update_db_weekly.py" file.  

Finally, strategy results can be viewed using the "results.py" file.  This functionality displays a number of investment return metrics and compares the actual results compared to different back test strategies.  

## Additional Considerations

To understand the foundation for this project, one would need an understanding of the project that served as its [oringal inspiration](https://github.com/chpr1410/MSDS696-Practicum)

Some changes have been made to the structure of this project.  The main change includes only using closing price data rather than open, high, low, close pricing data.  This made streaming the data more feasible for some data providers.  

Other changes made MongoDB storage more tenable and efficient.  

The deep learning models that underlie the "Trading Bot's" strategy would need to be re-trained periodically to capture as much data as possible, as well as recent trends.
