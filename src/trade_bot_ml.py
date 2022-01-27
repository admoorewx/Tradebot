import datetime as DT
from svr import SVR_forecast
from alpaca import accountStatus, get_positions, buy, sell, get_position_quantity, checkCash, check_market_hours
from functions import determine_quantity, yahoo_price, high_low_check, normalize, currentTime, yahoo_current_price
from stocks_db import update_stock_transaction, update_bought_price, update_sold_price, get_stock_info, get_all_stock_info

# Set tunable parameters
# Minimum allowed cash value before selling stocks to recoup losses
min_cash_limit = 80000.0
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
        last_price = yahoo_current_price(stock)
        transaction_time = DT.datetime.strftime(DT.datetime.utcnow(), "%m/%d/%Y %H:%M")
        # Update the stock database
        update_stock_transaction(stock, "BUY",transaction_time,"Relative Min")
        update_bought_price(stock,last_price,False)
        # Send the "buy" command to alpaca
        buy(stock,qnty)
        # Log this transaction
        print(f'{currentTime()}: High/Low Check: Purchasing {qnty} shares of {stock} at price ${last_price} per share at {transaction_time} UTC.')

    elif signal == -1 and stock in symbols:
        last_price = yahoo_current_price(stock)
        transaction_time = DT.datetime.strftime(DT.datetime.utcnow(), "%m/%d/%Y %H:%M")
        # Update the stock database
        update_stock_transaction(stock, "SELL",transaction_time,"Relative Max")
        update_bought_price(stock,last_price,True)
        # Send the "sell" command to alpaca
        sell(stock,get_position_quantity(stock))
        # Log the transaction
        print(f'{currentTime()}: High/Low Check: Selling all shares of {stock} at price ${last_price} per share at {transaction_time} UTC.')
    else:
        pass
    return signal
########################################################################################################################
def SVR_trade(stock):
    # Proceed to generate a forecast.
    recommendation, reason = SVR_forecast(stock)
    print(f'{currentTime()}: SVR forecast {stock} Recommendation: {recommendation} with reason: {reason}')
    # Get the current position of the stock
    position = get_stock_info(stock)[1]
    last_price = yahoo_current_price(stock)
    transaction_time = DT.datetime.strftime(DT.datetime.utcnow(), "%m/%d/%Y %H:%M")
    if recommendation == "BUY" and position == "NONE":
        # Open a LONG position
        cash = checkCash()
        qnty = determine_quantity(stock, cash, max_risk)
        # Update the stock database
        update_stock_transaction(stock, "BUY",transaction_time,reason)
        update_bought_price(stock,last_price,False)
        # Send the "buy" command to alpaca
        buy(stock,qnty)
        # Log this transaction
        print(f'{currentTime()}: Purchasing {qnty} shares of {stock} at price ${last_price} per share at {transaction_time} UTC.')
    elif recommendation == "SELL" and position == "LONG":
        # Close a LONG position
        # Update the stock database
        update_stock_transaction(stock, "SELL", transaction_time, reason)
        update_sold_price(stock, last_price, True)
        # Send the "sell" command to alpaca
        sell(stock,get_position_quantity(stock))
        # Log this transaction
        print(f'{currentTime()}: Selling all shares of {stock} at price ${last_price} per share at {transaction_time} UTC.')
    elif recommendation == "SELL" and position == "NONE":
        # Open a SHORT Position
        cash = checkCash()
        qnty = determine_quantity(stock, cash, max_risk)
        # Update the stock database
        update_stock_transaction(stock, "SELL", transaction_time, reason)
        update_sold_price(stock, last_price, False)
        # Send the "sell" command to alpaca
        sell(stock,qnty)
        # Log this transaction
        print(f'{currentTime()}: Selling {qnty} shares of {stock} at price ${last_price} per share at {transaction_time} UTC.')
    elif recommendation == "SELL" and position == "SHORT":
        # Close a SHORT position
        # Update the stock database
        update_stock_transaction(stock, "BUY",transaction_time,reason)
        update_bought_price(stock,last_price,True)
        # Send the "buy" command to alpaca
        buy(stock,get_position_quantity(stock))
        # Log this transaction
        print(f'{currentTime()}: Purchasing {get_position_quantity(stock)} shares of {stock} at price ${last_price} per share at {transaction_time} UTC.')
    else:
        # Log the "pass" transaction
        print(f'{currentTime()}: Passing on {stock} at price ${last_price} per share at {transaction_time} UTC.')

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
                    print("\n")
                    print(f'{currentTime()}: Valid conditions for trading {stock}. Getting forecast.')
                    # Do a min/max check first
                    signal = high_low_trade(stock,symbols)
                    # If no trades made, do a SVR Forecast
                    if signal == 0:
                        SVR_trade(stock)
                else:
                    # Close positions until funds are above the minimum
                    print("\n")
                    print(f'{currentTime()}: Not enough cash! Closing positions to raise funds...')
                    cash = checkCash()
                    print(cash)
                    while cash < min_cash_limit:
                        for position in positions:
                            # Check to see if it's been at least 20 hours since our last transaction with this stock
                            last_transaction_time = DT.datetime.strptime(get_stock_info(position.symbol)[3], "%m/%d/%Y %H:%M")
                            time_delta = DT.datetime.utcnow() - last_transaction_time
                            if time_delta.total_seconds() > min_wait_time:
                                if position.side == "long":
                                    # Send sell command
                                    sell(positions[0].symbol,positions[0].qnty)
                                    # Update the stock database
                                    last_price = yahoo_current_price(position.symbol)
                                    transaction_time = DT.datetime.strftime(DT.datetime.utcnow(), "%m/%d/%Y %H:%M")
                                    reason = "Raising funds to baseline."
                                    update_stock_transaction(stock, "SELL", transaction_time, reason)
                                    update_sold_price(position.symbol, last_price, True)
                                    # Log transaction
                                    print(f'{currentTime()}: Selling all shares of {position.symbol} at price ${last_price} per share at {transaction_time} UTC.')
                                elif positions.side == "short":
                                    # Send buy command
                                    buy(positions[0].symbol,positions[0].qnty)
                                    # Update the stock database
                                    last_price = yahoo_current_price(positions.symbol)
                                    transaction_time = DT.datetime.strftime(DT.datetime.utcnow(), "%m/%d/%Y %H:%M")
                                    reason = "Raising funds to baseline."
                                    update_stock_transaction(stock, "BUY", transaction_time, reason)
                                    update_bought_price(positions.symbol, last_price, True)
                                    # Log transaction
                                    print(f'{currentTime()}: Buying all shares of {positions.symbol} at price ${last_price} per share at {transaction_time} UTC.')

            else:
                print(f'{currentTime()}: Stock {stock} has been traded within the past 20 hours. Will check again later.')
    else:
        print(f'{currentTime()}: Account has been blocked! Check Alpaca account for details.')
else:
    print(f'{currentTime()}: Market Closed. No trades made.')

print(f'{currentTime()}: COMPLETE.')