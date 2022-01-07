# Tradebot
Tradebot is a stock trading algorithm that currently relies on Support Vector Regression to produce a 24-hour ensemble stock forecast for a variety of U.S.-based stocks. Additional ML algorithms will be explored in the future. This bot aims to avoid Day Trader Pattern rules by holding positions for at least a 24 hour period. Future versions will explore methods around PDT rules while increasing the frequency of trades, including the possibility of crypto trading. 

This tool uses a combination of the Alpaca API (https://alpaca.markets/) for brokerage purposes and the yfinance API (https://github.com/ranaroussi/yfinance) to retrive stock data. 

Dependency requirements can be found in the trade_bot_env.txt file. 

Currently analyzed stocks can be found in the stock_list.json file, but can be edited for user-preference. This file will likely be omitted in future versions in favor of an SQL database to aid in tracking when stocks were bought/sold. 

This is still a work in progress; future changes are likely.
