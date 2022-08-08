import numpy as np
import tensorflow as tf
import datetime as DT
from alpaca import *
from functions import yahoo_hist, determine_quantity, currentTime
from sklearn.preprocessing import MinMaxScaler


period = "1mo"
interval = "1h"
history = 120 # number of hours in the past to use as input for the RNN.
testing_std = 0.25 # Standard deviation of the errors during RNN testing.
loss_limit = 50.0 # USD, maximum allowed amount of loss on a single asset.
min_cash = 200.0 # USD, the minimum allowed amount of cash in Alpaca account.
risk_tolerance = 0.05 # Percent of cash to spend on a single trade.
yf_crypto = "BTC-USD"
crypto = "BTCUSD"
model_path = "/home/icebear/Tradebot/btc_nn"
log_path = "/home/icebear/Tradebot/btc_nn/btc_nn.log"

##############################################################################################
def get_price_inputs(crypto,period,interval):
    # Get latest BTC hourly data
    hist = yahoo_hist(crypto,period=period,interval=interval,prepost=True)
    close = hist["Close"].values
    close = np.asarray(close[-history:])
    return close
##############################################################################################
def data_prep_and_last_price(data):
    # Normalize the data
    scaler = MinMaxScaler()
    data = data.reshape(len(data),1)
    data= scaler.fit_transform(data)
    scaled_latest_price = data[-1][0]
    # Reshape the data to the expected input format (need to be of shape (1,history,1)
    data = data.reshape(1,len(data),1)
    return data, scaled_latest_price
##############################################################################################
def forecast(inputs,scaled_latest_price,min_req_diff=0.0,use_min_diff=False):
    model = tf.keras.models.load_model(model_path)
    # Use model to predict the next hours price
    scaled_forecast = model.predict(inputs)[0][0]
    if use_min_diff:
        if abs(scaled_forecast - scaled_latest_price) > min_req_diff:
            if scaled_forecast > scaled_latest_price:
                recommendation = True
            else:
                recommendation = False
        else:
            # If this check fails, then the forecast signal is not strong enough.
            recommendation = False
    else:
        if scaled_forecast > scaled_latest_price:
            recommendation= True
        else:
            recommendation = False
    return recommendation
##############################################################################################
def check_for_profit(sym):
    unrealized_pl = get_position(sym).unrealized_pl
    if unrealized_pl > 0:
        return True
    else:
        return False
##############################################################################################
def check_for_position(sym):
    positions = get_positions()
    symbols = [position.symbol for position in positions]
    if sym in symbols:
        holding = True
    else:
        holding = False
    return holding
##############################################################################################
def check_loss(sym,loss_limit):
    unrealized_pl = get_position(sym).unrealized_pl
    if unrealized_pl <= (-1*loss_limit):
        return True
    else:
        return False
##############################################################################################
def check_funds(min_cash):
    cash = checkCash()
    if cash > min_cash:
        return True
    else:
        return False
##############################################################################################
def buy_action(sym,risk_tolerance):
    # Determin the quantity based on available cash and risk tolerance
    cash = checkCash()
    qnty = determine_quantity(sym, cash, risk_tolerance)
    buy(sym,qnty)
##############################################################################################
def sell_action(sym):
    # Assumes we're selling all shares.
    qnty = get_position_quantity(sym)
    sell(sym,qnty)
##############################################################################################

with open(log_path,'a') as file:
    file.write(f'{currentTime()}: Starting run.\n')

    # Get the latest price
    inputs = get_price_inputs(yf_crypto,period,interval)
    # Data prep
    inputs, last_price = data_prep_and_last_price(inputs)
    # Use model to predict the next hours price
    recommendation = forecast(inputs,last_price,min_req_diff=testing_std,use_min_diff=True)

    # Make a buy/sell decision based on:
    # 1) Sufficient funds
    # 2) Current positions
    # 3) Forecast price

    # Check current funds
    sufficient_funds = check_funds(min_cash)
    if sufficient_funds:
        # Get current position
        holding = check_for_position(crypto)
        if holding:
            # Check for profit/loss
            profit = check_for_profit(crypto)
            if profit:
                if recommendation:
                    file.write(f'{currentTime()}: Position open, buying more\n')
                    buy_action(crypto,risk_tolerance)
                else:
                    file.write(f'{currentTime()}: Position open, selling to close.\n')
                    sell_action(crypto)
            else:
                if recommendation:
                    # Do a loss check
                    loss_too_high = check_loss(crypto,loss_limit)
                    if loss_too_high:
                        file.write(f'{currentTime()}: Losses too high, selling to minimize damage.\n')
                        sell_action(crypto)
                    else:
                        file.write(f'{currentTime()}: Current loss, but forecast to improve. Buying more.\n')
                        buy_action(crypto, risk_tolerance)
                else:
                    file.write(f'{currentTime()}: Current loss, forecast to go down. Selling.\n')
                    sell_action(crypto)
        else:
            if recommendation:
                file.write(f'{currentTime()}:No position open. Buying to open.\n')
                buy_action(crypto, risk_tolerance)
            else:
                file.write(f'{currentTime()}:No position open, but sell recommended. No action taken.\n')
    else:
        file.write(f'{currentTime()}:Insufficient funds. Taking no action.\n')
    file.write(f'{currentTime()}:Run Complete.\n')
file.close()
