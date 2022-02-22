from stocks_db import get_stock_info, update_stock_transaction, update_bought_price, update_sold_price
from alpaca import buy, sell, get_position_quantity, get_position, get_positions, checkCash
from functions import currentTime, high_low_check, yahoo_current_price, yahoo_price, normalize, determine_quantity, position_check
from svr import SVR_forecast
import datetime

########################################################################################################################
def position_pop(min_wait_time):
    # get the first position held
    positions = get_positions()
    sold_position = False
    ind = 0
    while sold_position == False:
        position = positions[ind]
        # Check to see if it's been at least 20 hours since our last transaction with this stock
        last_transaction_time = datetime.datetime.strptime(get_stock_info(position.symbol)[3], "%m/%d/%Y %H:%M")
        time_delta = datetime.datetime.utcnow() - last_transaction_time
        if time_delta.total_seconds() > min_wait_time:
            # Begin buy/sell procedure
            last_price = float(position.current_price)
            transaction_time = datetime.datetime.strftime(datetime.datetime.utcnow(), "%m/%d/%Y %H:%M")
            reason = "Raising funds to baseline."
            # get rid of it
            if position.side == "long":
                sell(position.symbol, int(position.qty))
                update_sold_price(position.symbol, last_price, True)
                update_stock_transaction(position.symbol, "SELL", transaction_time, reason)
                print(f'{currentTime()}: Selling all shares of {position.symbol} at price ${last_price} per share at {transaction_time} UTC.')
                sold_position = True
            else:
                buy(position.symbol, abs(int(position.qty)))
                update_bought_price(position.symbol, last_price, True)
                update_stock_transaction(position.symbol, "BUY", transaction_time, reason)
                print(f'{currentTime()}: Buying all shares of {position.symbol} at price ${last_price} per share at {transaction_time} UTC.')
                sold_position = True
        else:
            # move on to the next position
            ind = ind + 1
            print(f'{currentTime()}: Position_pop(): Can not buy/sell position {position.symbol}, assest traded too recently.')
########################################################################################################################
def high_low_trade(stock,symbols,max_risk):
    normed = normalize(yahoo_price(stock, period='1mo', interval='1h'))
    signal = high_low_check(normed, 5)
    # if there is a min and we don't own it yet - buy
    if signal == -1 and stock not in symbols:
        cash = checkCash()
        qnty = determine_quantity(stock,cash,max_risk)
        last_price = yahoo_current_price(stock)
        transaction_time = datetime.datetime.strftime(datetime.datetime.utcnow(), "%m/%d/%Y %H:%M")
        # Update the stock database
        update_stock_transaction(stock, "BUY",transaction_time,"Relative Min")
        update_bought_price(stock,last_price,False)
        # Send the "buy" command to alpaca
        buy(stock,qnty)
        # Log this transaction
        print(f'{currentTime()}: High/Low Check: Purchasing {qnty} shares of {stock} at price ${last_price} per share at {transaction_time} UTC.')
    elif signal == -1 and stock in symbols:
        position = get_position(stock).side
        # if the position is short, buy
        if position == 'short':
            qnty = abs(float(position.qty))
            last_price = yahoo_current_price(stock)
            transaction_time = datetime.datetime.strftime(datetime.datetime.utcnow(), "%m/%d/%Y %H:%M")
            # Update the stock database
            update_stock_transaction(stock, "BUY", transaction_time, "Relative Min")
            update_bought_price(stock, last_price, False)
            # Send the "buy" command to alpaca
            buy(stock, qnty)
            # Log this transaction
            print(f'{currentTime()}: High/Low Check: Purchasing {qnty} shares of {stock} at price ${last_price} per share at {transaction_time} UTC.')
        else: # if the position is long, just wait by resetting signal to zero
            signal = 0
    elif signal == 1 and stock in symbols:
        position = get_position(stock).side
        # if there is a max and position is long, sell
        if position == 'long':
            last_price = float(get_position(stock).current_price)
            transaction_time = datetime.datetime.strftime(datetime.datetime.utcnow(), "%m/%d/%Y %H:%M")
            # Update the stock database
            update_stock_transaction(stock, "SELL",transaction_time,"Relative Max")
            update_bought_price(stock,last_price,True)
            # Send the "sell" command to alpaca
            sell(stock,get_position_quantity(stock))
            # Log the transaction
            print(f'{currentTime()}: High/Low Check: Selling all shares of {stock} at price ${last_price} per share at {transaction_time} UTC.')
        else: # if the position is short and there's a max - just hold by resetting signal to 0
            signal = 0
    elif signal == 1 and stock not in symbols:
        # if there is a max and we don't own it - short it
        cash = checkCash()
        qnty = determine_quantity(stock,cash,max_risk)
        last_price = yahoo_current_price(stock)
        transaction_time = datetime.datetime.strftime(datetime.datetime.utcnow(), "%m/%d/%Y %H:%M")
        # Update the stock database
        update_stock_transaction(stock, "SELL",transaction_time,"Relative Max")
        update_bought_price(stock,last_price,True)
        # Send the "sell" command to alpaca
        sell(stock,qnty)
        # Log the transaction
        print(f'{currentTime()}: High/Low Check: Selling {qnty} shares of {stock} at price ${last_price} per share at {transaction_time} UTC.')
    else:
        pass
    return signal
########################################################################################################################
def SVR_trade(stock, max_risk):
    # Proceed to generate a forecast.
    recommendation, reason = SVR_forecast(stock)
    print(f'{currentTime()}: SVR forecast {stock} Recommendation: {recommendation} with reason: {reason}')
    # Get the current position of the stock
    position = get_stock_info(stock)[1]
    try:
        last_price = yahoo_current_price(stock)
    except:
        last_price = yahoo_price(stock,period="1d",interval="1m")["Close"][-1]
    transaction_time = datetime.datetime.strftime(datetime.datetime.utcnow(), "%m/%d/%Y %H:%M")
    if recommendation == "BUY" and position != "SHORT":
        # Open or extend a LONG position
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
    elif recommendation == "SELL" and position != "LONG":
        # Open or extend a SHORT Position
        cash = checkCash()
        qnty = determine_quantity(stock, cash, max_risk)
        # Update the stock database
        update_stock_transaction(stock, "SELL", transaction_time, reason)
        update_sold_price(stock, last_price, False)
        # Send the "sell" command to alpaca
        sell(stock,qnty)
        # Log this transaction
        print(f'{currentTime()}: Selling {qnty} shares of {stock} at price ${last_price} per share at {transaction_time} UTC.')
    elif recommendation == "BUY" and position == "SHORT":
        # Close a SHORT position
        # Update the stock database
        update_stock_transaction(stock, "BUY",transaction_time,reason)
        update_bought_price(stock,last_price,True)
        # Send the "buy" command to alpaca
        qnty = abs(float(get_position(stock).qty))
        buy(stock,qnty)
        # Log this transaction
        print(f'{currentTime()}: Purchasing {get_position_quantity(stock)} shares of {stock} at price ${last_price} per share at {transaction_time} UTC.')
    else:
        # Log the "pass" transaction
        print(f'{currentTime()}: Passing on {stock} at price ${last_price} per share at {transaction_time} UTC.')
########################################################################################################################
def port_trade(stock):
    """
    Check to see if we can close any open positions in our portfolio.
    We'll only close positions if we have gains or if losses are too high.
    Return:
        1 - closed an open position
        0 - did not close an open position
    """
    # Check to see if we can close any open positions in our portfolio.
    # We'll only close positions if we have gains or if losses are too high.
    position = get_position(stock)
    rec, reason = position_check(stock)
    transaction_time = datetime.datetime.strftime(datetime.datetime.utcnow(), "%m/%d/%Y %H:%M")
    if rec == "BUY":
        qnty = abs(float(position.qty))
        last_price = float(position.current_price)
        # Update the stock database
        update_stock_transaction(stock, "BUY",transaction_time,reason)
        update_bought_price(stock,last_price,True)
        # Send the "buy" command to alpaca
        buy(stock,qnty)
        # Log this transaction
        print(f'{currentTime()}: Portfolio Check: Purchasing {qnty} shares of {stock} at price ${last_price} per share at {transaction_time} UTC.')
        return 1
    elif rec == "SELL":
        qnty = float(position.qty)
        last_price = float(position.current_price)
        # Update the stock database
        update_stock_transaction(stock, "SELL",transaction_time,reason)
        update_sold_price(stock,last_price,True)
        # Send the "buy" command to alpaca
        buy(stock,qnty)
        # Log this transaction
        print(f'{currentTime()}: Portfolio Check: Selling {qnty} shares of {stock} at price ${last_price} per share at {transaction_time} UTC.')
        return 1
    else:
        print(f'{currentTime()}: Portfolio Check: position losses manageable, holding position.')
        return 0
########################################################################################################################