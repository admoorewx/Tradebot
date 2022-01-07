import datetime as DT
import numpy as np
import pandas as pd
import json
from svr import SVR_forecast_train
from alpaca import accountStatus, get_positions, buy, sell, get_position_quantity, checkCash, check_market_hours
from functions import determine_quantity

# Set tunable parameters
# Minimum allowed cash value before selling stocks to recoup losses
min_cash_limit = 90000.0
# Maximum percentage of total cash willing to risk on a single buy
max_risk = 0.05
cash = 10.0

# Read in stock list
with open("stock_list.json") as json_file:
    stocks = json.load(json_file).keys()

# Get and state current time
currentTime = DT.datetime.utcnow()
print(f'Starting trade routine at time {DT.datetime.strftime(currentTime,"%m/%d/%Y %H:%M")} UTC')

# Check to see if market is open
# if check_market_hours():
# Check for valid account status and sufficient funds
if accountStatus():
    # First, get a list of all currently owned stocks/positions
    positions = get_positions()
    symbols = [position.symbol for position in positions]
    for stock in stocks:
        # TO DO: Add a check to see the last time we bought this stock. If < 24 hours, skip.
        if checkCash() > min_cash_limit:
            print("Enough Cash to trade!")
            # Proceed to generate a forecast.
            recommendation, net = SVR_forecast_train(stock)
            # For testing
            cash = cash + net
            print(f'{stock} Recommendation: {recommendation}')
            print(f'NET CASH: ${cash}\n')
            # end of testing
            if recommendation == "BUY" and stock not in symbols:
                # cash = checkCash()
                qnty = determine_quantity(stock,cash,max_risk)
                print(qnty)
            #     #buy(stock,qnty)
            # elif recommendation == "SELL" and stock in positions:
            #     #sell(stock,get_position_quantity(stock))
            # else:
            #     pass
                # Maybe do a check here to make sure the price of our positions
                # isn't falling too much and is missed due to a bad forecast.
        else:
            print("Not enough cash! Selling off assets to raise funds...")
            # Sell assets until fund are above the minimum
            # cash = checkCash()
            # while cash < min_cash_limit:
            #     positions = get_positions()
            #     sell(positions[0].symbol,positions[0].qnty)
