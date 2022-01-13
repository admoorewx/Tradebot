import datetime as DT
from svr import SVR_forecast
from alpaca import accountStatus, get_positions, buy, sell, get_position_quantity, checkCash, check_market_hours
from functions import determine_quantity, yahoo_price, high_low_check, normalize, currentTime
from stock_db import update_stock, get_stock_info, get_all_stock_info

# Set tunable parameters
# Minimum allowed cash value before selling stocks to recoup losses
min_cash_limit = 90000.0
# Minimum amount of time (in seconds) that much elapse before a stock can be bought/sold again.
min_wait_time = 72000.0
# Maximum percentage of total cash willing to risk on a single buy
max_risk = 0.05

########################################################################################################################
def high_low_trade(stock,symbols):
    normed = normalize(yahoo_price(stock, period='1mo', interval='1h'))
    signal = high_low_check(normed, 5)
    if signal == 1 and stock not in symbols:
        cash = checkCash()
        qnty = determine_quantity(stock,cash,max_risk)
        last_price = yahoo_price(stock, period='1d', interval='1m')[-1]
        transaction_time = DT.datetime.strftime(DT.datetime.utcnow(), "%m/%d/%Y %H:%M")
        print(f'{currentTime()}: High/Low Check: Purchasing {qnty} shares of {stock} at price ${last_price} per share at {transaction_time} UTC.\n')
        # Update the stock database
        update_stock(stock,1,"BUY",transaction_time,last_price,"Relative Min")
        buy(stock,qnty)
    elif signal == -1 and stock in symbols:
        last_price = yahoo_price(stock, period='1d', interval='1m')[-1]
        transaction_time = DT.datetime.strftime(DT.datetime.utcnow(), "%m/%d/%Y %H:%M")
        print(f'{currentTime()}: High/Low Check: Selling all shares of {stock} at price ${last_price} per share at {transaction_time} UTC.\n')
        # Update the stock database
        update_stock(stock,0,"SELL",transaction_time,last_price,"Relative Max")
        sell(stock,get_position_quantity(stock))
    else:
        pass
    return signal
########################################################################################################################
def SVR_trade(stock,symbols):
    # Proceed to generate a forecast.
    recommendation, reason = SVR_forecast(stock)
    print(f'{currentTime()}: SVR forecast {stock} Recommendation: {recommendation} with reason: {reason}')
    # end of testing
    last_price = yahoo_price(stock, period='1d', interval='1m')[-1]
    transaction_time = DT.datetime.strftime(DT.datetime.utcnow(), "%m/%d/%Y %H:%M")
    if recommendation == "BUY" and stock not in symbols:
        cash = checkCash()
        qnty = determine_quantity(stock, cash, max_risk)
        print(f'{currentTime()}: Purchasing {qnty} shares of {stock} at price ${last_price} per share at {transaction_time} UTC.\n')
        # Update the stock database
        update_stock(stock,1,"BUY",transaction_time,last_price,reason)
        buy(stock,qnty)
    elif recommendation == "SELL" and stock in symbols:
        print(f'{currentTime()}: Selling all shares of {stock} at price ${last_price} per share at {transaction_time} UTC.\n')
        # Update the stock database
        update_stock(stock,0,"SELL",transaction_time,last_price,reason)
        sell(stock,get_position_quantity(stock))
    else:
        print(f'{currentTime()}: Passing on {stock} at price ${last_price} per share at {transaction_time} UTC.\n')
        # Update the stock database
        if stock in symbols:
            update_stock(stock,1,"PASS",transaction_time,last_price,reason)
        else:
            update_stock(stock,0,"PASS",transaction_time,last_price,reason)
########################################################################################################################


# Read in stock list from database
stock_info = get_all_stock_info()[0:]
stocks = [s[1] for s in stock_info]

# Get and state current time
print(f'{currentTime()}: Starting trade routine.')
# Check to see if market is open
if check_market_hours():
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
                    print(f'{currentTime()}: Valid conditions for trading {stock}. Getting forecast.')
                    # Do a min/max check first
                    signal = high_low_trade(stock,symbols)
                    # If no trades made, do a SVR Forecast
                    if signal == 0:
                        SVR_trade(stock,symbols)
                else:
                    # Sell assets until funds are above the minimum
                    print(f'{currentTime()}: Not enough cash! Selling off assets to raise funds...')
                    cash = checkCash()
                    while cash < min_cash_limit:
                        positions = get_positions()
                        sell(positions[0].symbol,positions[0].qnty)
                        last_price = yahoo_price(positions[0].symbol, period='1d', interval='1m')[-1]
                        transaction_time = DT.datetime.strftime(DT.datetime.utcnow(), "%m/%d/%Y %H:%M")
                        # Update the stock database
                        reason = "Raising funds to baseline."
                        update_stock(stock,0,"SELL",transaction_time,last_price,reason)
                        print(f'{currentTime()}: Selling all shares of {stock} at price ${last_price} per share at {transaction_time} UTC.')
            else:
                print(f'{currentTime()}: Stock {stock} has been traded within the past 20 hours. Will check again later.')
    else:
        print(f'{currentTime()}: Account has been blocked! Check Alpaca account for details.')
else:
    print(f'{currentTime()}: Market Closed. No trades made.')

print(f'{currentTime()}: COMPLETE.')