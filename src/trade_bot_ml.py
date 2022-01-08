import datetime as DT
import numpy as np
import pandas as pd
import json
from svr import SVR_forecast_train_diagnosis
from alpaca import accountStatus, get_positions, buy, sell, get_position_quantity, checkCash, check_market_hours
from functions import determine_quantity, yahoo_price
from stock_db import update_stock, get_stock_info, get_all_stock_info

# Set tunable parameters
# Minimum allowed cash value before selling stocks to recoup losses
min_cash_limit = 90000.0
# Minimum amount of time (in seconds) that much elapse before a stock can be bought/sold again.
min_wait_time = 72000.0
# Maximum percentage of total cash willing to risk on a single buy
max_risk = 0.05
# For testing
cash = 10.0

# Read in stock list from database
stock_info = get_all_stock_info()[0:]
stocks = [s[1] for s in stock_info]

# Get and state current time
currentTime = DT.datetime.utcnow()
print("\n")
print(f'Starting trade routine at time {DT.datetime.strftime(currentTime,"%m/%d/%Y %H:%M")} UTC\n')


# Check to see if market is open
# if check_market_hours():
# Check for valid account status and sufficient funds
if accountStatus():
    # First, get a list of all currently owned stocks/positions
    positions = get_positions()
    symbols = [position.symbol for position in positions]
    for stock in stocks:
        # Check to see if it's been at least 20 hours since our last transaction with this stock
        last_transaction_time = DT.datetime.strptime(get_stock_info(stock)[3],"%m/%d/%Y %H:%M")
        time_delta = DT.datetime.utcnow() - last_transaction_time
        if time_delta.total_seconds() > min_wait_time:
            # Check to see if we have enough cash to buy. If not, sell off stocks.
            if checkCash() > min_cash_limit:
                print(f'Valid conditions for trading {stock}. Getting forecast.')
                # Proceed to generate a forecast.
                recommendation, reason, net = SVR_forecast_train_diagnosis(stock)
                print(f'{stock} Recommendation: {recommendation} with reason: {reason}')
                # For testing
                cash = cash + net
                print(f'NET CASH: ${cash}')
                # end of testing
                last_price = yahoo_price(stock, period='1d', interval='1m')[-1]
                transaction_time = DT.datetime.strftime(DT.datetime.utcnow(), "%m/%d/%Y %H:%M")
                if recommendation == "BUY" and stock not in symbols:
                    # cash = checkCash()
                    qnty = determine_quantity(stock,cash,max_risk)
                    print(f'Purchasing {qnty} shares of {stock} at price ${last_price} per share at {transaction_time} UTC.\n')
                    # Update the stock database
                    # update_stock(stock,1,"BUY",transaction_time,last_price,reason)
                    # buy(stock,qnty)
                elif recommendation == "SELL" and stock in symbols:
                    print(f'Selling all shares of {stock} at price ${last_price} per share at {transaction_time} UTC.\n')
                    # Update the stock database
                    # update_stock(stock,0,"SELL",transaction_time,last_price,reason)
                    # sell(stock,get_position_quantity(stock))
                else:
                    print(f'Passing on {stock} at price ${last_price} per share at {transaction_time} UTC.\n')
                    # Update the stock database
                    # if stock in symbols:
                    #     update_stock(stock,1,"PASS",transaction_time,last_price,reason)
                    # else:
                    #     update_stock(stock,0,"PASS",transaction_time,last_price,reason)
                    # Maybe do a check here to make sure the price of our positions
                    # isn't falling too much and is missed due to a bad forecast.
            else:
                # Sell assets until fund are above the minimum
                print("Not enough cash! Selling off assets to raise funds...")
                # cash = checkCash()
                # while cash < min_cash_limit:
                #     positions = get_positions()
                #     sell(positions[0].symbol,positions[0].qnty)
                #     last_price = yahoo_price(positions[0].symbol, period='1d', interval='1m')[-1]
                #     transaction_time = DT.datetime.strftime(DT.datetime.utcnow(), "%m/%d/%Y %H:%M")
                #     # Update the stock database
                #     reason = "Raising funds to baseline."
                #     update_stock(stock,0,"SELL",transaction_time,last_price,reason)
                #     print(f'Selling all shares of {stock} at price ${last_price} per share at {transaction_time} UTC.')
